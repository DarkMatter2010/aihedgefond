"""Causal per-symbol feature pipeline with a tidy timestamp/symbol output index."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

import pandas as pd

from aihedgefund.core.bus import MessageBus
from aihedgefund.core.runtime import Clock, IdProvider, SystemClock, Uuid4IdProvider
from aihedgefund.core.schemas import (
    BarFrame,
    CorporateActionInput,
    FeatureMatrixPayload,
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

FEATURE_COLUMNS = (
    "log_return",
    "realized_vol_20",
    "momentum_20",
    "ma_ratio_20",
    "rsi_14",
    "macd",
    "macd_signal",
    "atr_14",
    "close_zscore_20",
)
FEATURE_DTYPES: tuple[Literal["float64"], ...] = (
    "float64",
    "float64",
    "float64",
    "float64",
    "float64",
    "float64",
    "float64",
    "float64",
    "float64",
)


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
        *,
        clock: Clock | None = None,
    ) -> None:
        self._bus = bus
        self._parameters = parameters or FeatureParameters()
        if self._parameters != FeatureParameters():
            msg = "FeaturePipeline requires the fixed Phase 1 feature parameters"
            raise ValueError(msg)
        self._clock = clock or SystemClock()

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
        matrix = matrix.loc[:, list(FEATURE_COLUMNS)].astype(
            {column: "float64" for column in FEATURE_COLUMNS}
        )

        timestamps = matrix.index.get_level_values("timestamp")
        provenance = pd.DataFrame(
            {"anchor_timestamp": timestamps, "source_timestamp": timestamps},
            index=matrix.index,
        )
        assert_no_lookahead(provenance, "anchor_timestamp")
        self._bus.publish_event(
            FeaturesComputed(
                timestamp=self._clock.now(),
                payload=FeatureMatrixPayload(
                    symbols=tuple(per_symbol),
                    rows=len(matrix),
                    columns=FEATURE_COLUMNS,
                    dtypes=FEATURE_DTYPES,
                ),
            )
        )
        return matrix


def to_feature_vectors(
    matrix: pd.DataFrame,
    *,
    feature_set_version: str,
    id_provider: IdProvider | None = None,
) -> tuple[FeatureVector, ...]:
    """Convert complete finite matrix rows into existing Phase 0 DTOs."""
    ids = id_provider or Uuid4IdProvider()
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
                feature_vector_id=ids.new_id(),
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
    adjustment = adjust_corporate_actions(
        CorporateActionInput(
            raw_close=frame["close"],
            splits=splits,
            dividends=dividends,
        )
    )
    adjustment_factor = adjustment.as_of_adjusted / adjustment.raw_close
    adjusted = frame.copy()
    for column in ("open", "high", "low", "close"):
        adjusted[column] = frame[column].astype(float) * adjustment_factor
    return adjusted
