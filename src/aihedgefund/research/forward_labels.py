"""Dense leak-free forward-return labels for the Phase-2 baseline."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from aihedgefund.core.schemas import (
    BarFrame,
    CorporateActionInput,
    ForwardReturnLabelMeta,
)
from aihedgefund.data.corporate_actions import adjust_corporate_actions

# Phase-2 binds ``close_adj`` to the Phase-1 causal total-return series.
CLOSE_ADJ_SOURCE: Literal["as_of_adjusted"] = "as_of_adjusted"


def make_forward_return_labels(
    bars: BarFrame,
    *,
    horizon: int,
) -> tuple[pd.Series, ForwardReturnLabelMeta]:
    """Build ``close_adj[t+h]/close_adj[t]-1`` per ``(timestamp, symbol)``.

    ``close_adj`` is the Phase-1 ``as_of_adjusted`` series from
    ``adjust_corporate_actions`` (documented binding; not Yahoo ``adj_close``).

    The last ``horizon`` rows per symbol are left as NaN by the shift and are
    removed from the returned series (never zero-filled).
    """
    if horizon < 1:
        msg = "horizon must be >= 1"
        raise ValueError(msg)

    per_symbol: list[pd.Series] = []
    for symbol, frame in bars.bars.items():
        if len(frame) <= horizon:
            msg = f"{symbol}: need more than {horizon} bars to build forward returns"
            raise ValueError(msg)
        close_adj = adjust_corporate_actions(
            CorporateActionInput(
                raw_close=frame["close"],
                splits=bars.splits[symbol],
                dividends=bars.dividends[symbol],
            )
        ).as_of_adjusted
        forward = (close_adj.shift(-horizon) / close_adj - 1.0).rename(
            f"forward_return_{horizon}"
        )
        if not forward.iloc[-horizon:].isna().all():
            msg = f"{symbol}: last {horizon} forward-return rows must be NaN before drop"
            raise ValueError(msg)
        forward = forward.iloc[:-horizon]
        if forward.isna().any():
            msg = f"{symbol}: unexpected NaNs remain after dropping the final {horizon} rows"
            raise ValueError(msg)
        labeled = forward.copy()
        labeled.index = pd.MultiIndex.from_arrays(
            [labeled.index, [symbol] * len(labeled)],
            names=("timestamp", "symbol"),
        )
        per_symbol.append(labeled)

    if not per_symbol:
        msg = "bars must contain at least one symbol"
        raise ValueError(msg)

    labels = pd.concat(per_symbol).sort_index()
    labels = labels.astype("float64")
    meta = ForwardReturnLabelMeta(
        horizon=horizon,
        rows=len(labels),
        symbols=tuple(bars.bars),
        close_adj_source=CLOSE_ADJ_SOURCE,
    )
    return labels, meta
