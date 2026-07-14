"""Typed labeling orchestration and event emission."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, cast

import pandas as pd

from aihedgefund.core.bus import MessageBus
from aihedgefund.core.config import LabelSettings
from aihedgefund.core.runtime import Clock, SystemClock
from aihedgefund.core.schemas import (
    BarFrame,
    CorporateActionInput,
    LabelBatch,
    LabeledSample,
    LabelsComputed,
)
from aihedgefund.data.corporate_actions import adjust_corporate_actions
from aihedgefund.labels.labeling import (
    cusum_filter,
    daily_volatility,
    sample_weights,
    triple_barrier,
)


class LabelPipeline:
    """Create weighted, purge-friendly DTOs from configured barriers."""

    def __init__(
        self,
        settings: LabelSettings,
        bus: MessageBus,
        *,
        clock: Clock | None = None,
    ) -> None:
        self._settings = settings
        self._bus = bus
        self._clock = clock or SystemClock()

    def compute(
        self,
        close: pd.Series,
        events: Iterable[pd.Timestamp] | None = None,
        *,
        side: pd.Series | None = None,
        vol: pd.Series | None = None,
    ) -> tuple[LabeledSample, ...]:
        """Return immutable labels and announce successful computation."""
        event_times = (
            cusum_filter(close, self._settings.cusum_threshold)
            if events is None
            else tuple(events)
        )
        volatility = (
            daily_volatility(close, self._settings.vol_span) if vol is None else vol
        )
        labels = triple_barrier(
            close,
            event_times,
            self._settings.pt,
            self._settings.sl,
            self._settings.vertical_bars,
            side=side,
            vol=volatility,
        )
        if labels.empty:
            msg = "label pipeline produced no samples"
            raise ValueError(msg)
        weights = sample_weights(labels.set_index("t0")["t1"], close.index)
        samples = tuple(
            LabeledSample(
                label=cast(Literal[-1, 0, 1], int(row["label"])),
                t0=pd.Timestamp(row["t0"]).to_pydatetime(),
                t1=pd.Timestamp(row["t1"]).to_pydatetime(),
                ret=float(row["ret"]),
                weight=float(weights.loc[pd.Timestamp(row["t0"])]),
            )
            for _, row in labels.iterrows()
        )
        self._bus.publish_event(
            LabelsComputed(timestamp=self._clock.now(), payload=LabelBatch(samples=samples))
        )
        return samples

    def compute_from_bars(
        self,
        bars: BarFrame,
        symbol: str,
        events: Iterable[pd.Timestamp] | None = None,
        *,
        side: pd.Series | None = None,
    ) -> tuple[LabeledSample, ...]:
        """Label an explicitly total-return-adjusted canonical symbol series."""
        if symbol not in bars.bars:
            msg = f"symbol {symbol!r} is not present in BarFrame"
            raise ValueError(msg)
        adjustment = adjust_corporate_actions(
            CorporateActionInput(
                raw_close=bars.bars[symbol]["close"],
                splits=bars.splits[symbol],
                dividends=bars.dividends[symbol],
            )
        )
        return self.compute(
            adjustment.as_of_adjusted,
            events,
            side=side,
        )
