"""Causal per-symbol feature pipeline with a tidy timestamp/symbol output index."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite

import pandas as pd

from aihedgefund.core.bus import MessageBus
from aihedgefund.core.schemas import (
    BarFrame,
    FeaturesComputed,
    FeatureValue,
    FeatureVector,
)
from aihedgefund.data.corporate_actions import adjust_corporate_actions
from aihedgefund.features.indicators import (
    atr,
    log_return,
    macd,
    momentum,
    moving_average_ratio,
    realized_volatility,
    rolling_zscore,
    rsi,
)
from aihedgefund.features.pit import assert_no_lookahead


@dataclass(frozen=True)
class FeatureParameters:
    """Explicit indicator windows for a reproducible feature set."""

    volatility_span: int = 20
    momentum_periods: int = 20
    moving_average_window: int = 20
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    zscore_window: int = 20


def compute_symbol_features(
    frame: pd.DataFrame,
    parameters: FeatureParameters,
) -> pd.DataFrame:
    """Compute indicators whose row at t uses no observation after t."""
    close = frame["close"]
    macd_frame = macd(
        close,
        parameters.macd_fast,
        parameters.macd_slow,
        parameters.macd_signal,
    )
    return pd.concat(
        (
            log_return(close),
            realized_volatility(close, parameters.volatility_span),
            momentum(close, parameters.momentum_periods),
            moving_average_ratio(close, parameters.moving_average_window),
            rsi(close, parameters.rsi_period),
            macd_frame,
            atr(frame, parameters.atr_period),
            rolling_zscore(close, parameters.zscore_window),
        ),
        axis=1,
    )


class FeaturePipeline:
    """Compute and announce a tidy ``(timestamp, symbol)`` feature matrix."""

    def __init__(
        self,
        bus: MessageBus,
        parameters: FeatureParameters | None = None,
    ) -> None:
        self._bus = bus
        self._parameters = parameters or FeatureParameters()

    def compute(self, bars: BarFrame) -> pd.DataFrame:
        """Return a sorted MultiIndex matrix without feeding labels into features."""
        per_symbol = {
            symbol: compute_symbol_features(
                _adjusted_feature_frame(
                    frame,
                    bars.splits[symbol],
                    bars.dividends[symbol],
                ),
                self._parameters,
            )
            for symbol, frame in bars.bars.items()
        }
        matrix = pd.concat(per_symbol, names=("symbol", "timestamp"))
        matrix = matrix.reorder_levels(("timestamp", "symbol")).sort_index()

        timestamps = matrix.index.get_level_values("timestamp")
        provenance = pd.DataFrame(
            {"anchor_timestamp": timestamps, "source_timestamp": timestamps},
            index=matrix.index,
        )
        assert_no_lookahead(provenance, "anchor_timestamp")
        self._bus.publish_event(
            FeaturesComputed(
                timestamp=datetime.now(UTC),
                symbols=tuple(per_symbol),
                rows=len(matrix),
            )
        )
        return matrix


def to_feature_vectors(
    matrix: pd.DataFrame,
    *,
    feature_set_version: str,
) -> tuple[FeatureVector, ...]:
    """Convert complete finite matrix rows into existing Phase 0 DTOs."""
    vectors: list[FeatureVector] = []
    for (timestamp, symbol), row in matrix.iterrows():
        values = tuple(
            FeatureValue(name=str(name), value=float(value))
            for name, value in row.items()
            if pd.notna(value) and isfinite(float(value))
        )
        if len(values) != len(row):
            continue
        vectors.append(
            FeatureVector(
                timestamp=pd.Timestamp(timestamp).to_pydatetime(),
                symbol=str(symbol),
                features=values,
                feature_set_version=feature_set_version,
            )
        )
    return tuple(vectors)


def _adjusted_feature_frame(
    frame: pd.DataFrame,
    splits: pd.Series,
    dividends: pd.Series,
) -> pd.DataFrame:
    adjusted_close = adjust_corporate_actions(frame["close"], splits, dividends)
    total_return_close = adjusted_close["total_return_adjusted"]
    adjustment_factor = frame["close"].astype(float) / total_return_close
    adjusted = frame.copy()
    for column in ("open", "high", "low", "close"):
        adjusted[column] = frame[column].astype(float) / adjustment_factor
    return adjusted
