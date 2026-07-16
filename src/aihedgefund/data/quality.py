"""Configured quality checks for canonical market data."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from aihedgefund.core.bus import MessageBus
from aihedgefund.core.config import QualitySettings
from aihedgefund.core.runtime import Clock, SystemClock
from aihedgefund.core.schemas import (
    ExtremeReturnFlag,
    ExtremeReturnFlagged,
    QualityFailure,
    QualityGateFailed,
    QualityReport,
    QualityReportProduced,
)


class DataQualityError(RuntimeError):
    """Raised when any mandatory market-data quality rule fails."""


class DataQualityGate:
    """Validate one canonical symbol frame and emit typed outcomes."""

    def __init__(
        self,
        settings: QualitySettings,
        bus: MessageBus,
        *,
        clock: Clock | None = None,
    ) -> None:
        self._settings = settings
        self._bus = bus
        self._clock = clock or SystemClock()

    def validate(
        self,
        frame: pd.DataFrame,
        symbol: str,
        *,
        now: datetime | None = None,
    ) -> QualityReport:
        """Return a report when every hard-fail check passes (z-score is soft-flagged)."""
        checked_at = now or self._clock.now()
        try:
            report = self._validate(frame, symbol, checked_at)
        except DataQualityError as exc:
            self._bus.publish_event(
                QualityGateFailed(
                    timestamp=checked_at,
                    payload=QualityFailure(symbol=symbol, reason=str(exc)),
                )
            )
            raise
        except Exception as exc:
            failure = DataQualityError(f"{symbol} quality evaluation failed: {exc}")
            self._bus.publish_event(
                QualityGateFailed(
                    timestamp=checked_at,
                    payload=QualityFailure(symbol=symbol, reason=str(failure)),
                )
            )
            raise failure from exc

        self._bus.publish_event(QualityReportProduced(timestamp=checked_at, report=report))
        return report

    def _validate(
        self,
        frame: pd.DataFrame,
        symbol: str,
        checked_at: datetime,
    ) -> QualityReport:
        if frame.empty:
            raise DataQualityError(f"{symbol} has no bars")
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise DataQualityError(f"{symbol} index is not a DatetimeIndex")
        if frame.index.has_duplicates:
            raise DataQualityError(f"{symbol} has duplicate timestamps")
        if not frame.index.is_monotonic_increasing:
            raise DataQualityError(f"{symbol} timestamps are not strictly increasing")
        if "close" not in frame:
            raise DataQualityError(f"{symbol} has no close column")

        nan_ratios = {column: float(frame[column].isna().mean()) for column in frame.columns}
        excessive_nan = {
            column: ratio
            for column, ratio in nan_ratios.items()
            if ratio > self._settings.max_nan_ratio
        }
        if excessive_nan:
            raise DataQualityError(f"{symbol} exceeds NaN ratios: {excessive_nan}")

        close = frame["close"].astype(float)
        repeated = close.eq(close.shift(1))
        flatline_length = self._settings.stale_bars - 1
        if repeated.rolling(flatline_length).sum().ge(flatline_length).any():
            raise DataQualityError(
                f"{symbol} close is flat for {self._settings.stale_bars} consecutive bars"
            )

        last_timestamp = frame.index[-1]
        checked_timestamp = pd.Timestamp(checked_at)
        if checked_timestamp.tzinfo is None:
            checked_timestamp = checked_timestamp.tz_localize("UTC")
        else:
            checked_timestamp = checked_timestamp.tz_convert("UTC")
        if last_timestamp > checked_timestamp:
            raise DataQualityError(f"{symbol} last bar is in the future")
        age = checked_timestamp - last_timestamp
        if age > pd.Timedelta(days=self._settings.max_last_bar_age_days):
            raise DataQualityError(f"{symbol} last bar is stale by {age}")

        log_returns = np.log(close / close.shift(1)).dropna()
        max_abs_log_return = float(log_returns.abs().max()) if not log_returns.empty else 0.0
        if max_abs_log_return > self._settings.max_abs_logret:
            raise DataQualityError(
                f"{symbol} absolute log return {max_abs_log_return:.6f} exceeds cap"
            )

        return_std = float(log_returns.std(ddof=0)) if not log_returns.empty else 0.0
        if return_std == 0.0:
            max_return_zscore = 0.0
            abs_zscores = pd.Series(dtype=float)
        else:
            abs_zscores = ((log_returns - log_returns.mean()) / return_std).abs()
            max_return_zscore = float(abs_zscores.max())
        # Statistical extremes relative to own history are soft-flagged only.
        # Physical plausibility remains hard-fail via max_abs_logret above.
        if max_return_zscore > self._settings.zscore_cap:
            flagged = abs_zscores[abs_zscores > self._settings.zscore_cap]
            for bar_ts, z_score in flagged.items():
                bar_timestamp = pd.Timestamp(bar_ts)
                if bar_timestamp.tzinfo is None:
                    bar_timestamp = bar_timestamp.tz_localize("UTC")
                else:
                    bar_timestamp = bar_timestamp.tz_convert("UTC")
                self._bus.publish_event(
                    ExtremeReturnFlagged(
                        timestamp=checked_at,
                        payload=ExtremeReturnFlag(
                            symbol=symbol,
                            bar_timestamp=bar_timestamp.to_pydatetime(),
                            log_return=float(log_returns.loc[bar_ts]),
                            z_score=float(z_score),
                        ),
                    )
                )

        return QualityReport(
            symbol=symbol,
            rows=len(frame),
            nan_ratios=nan_ratios,
            max_abs_log_return=max_abs_log_return,
            max_return_zscore=max_return_zscore,
            last_timestamp=last_timestamp.to_pydatetime(),
        )
