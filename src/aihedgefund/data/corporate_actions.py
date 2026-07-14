"""Pure corporate-action transforms for raw daily closes.

Point-in-time caveat
--------------------
Back-adjustment retroactively rewrites pre-event history using split and
dividend facts that were not known before their ex-dates. The daily Phase 1 MVP
accepts this convention for feature and label computation, but exposes the raw
close alongside both adjusted series so a future PIT-strict path can preserve
the information set that was actually available at each historical timestamp.
"""

from __future__ import annotations

import pandas as pd


def adjust_corporate_actions(
    raw_close: pd.Series,
    splits: pd.Series,
    dividends: pd.Series,
) -> pd.DataFrame:
    """Return raw, split-adjusted, and total-return-adjusted close series."""
    if raw_close.empty:
        msg = "raw_close must not be empty"
        raise ValueError(msg)
    if not raw_close.index.equals(splits.index) or not raw_close.index.equals(dividends.index):
        msg = "close, splits, and dividends must have identical indexes"
        raise ValueError(msg)
    if raw_close.isna().any() or splits.isna().any() or dividends.isna().any():
        msg = "corporate-action inputs must not contain NaNs"
        raise ValueError(msg)
    if (raw_close <= 0).any() or (splits < 0).any() or (dividends < 0).any():
        msg = "prices must be positive and actions must be non-negative"
        raise ValueError(msg)

    split_events = splits.mask(splits == 0.0, 1.0).astype(float)
    inclusive_future_factor = split_events.iloc[::-1].cumprod().iloc[::-1]
    future_split_factor = inclusive_future_factor.shift(-1, fill_value=1.0)
    split_adjusted = raw_close.astype(float) / future_split_factor
    adjusted_dividends = dividends.astype(float) / future_split_factor

    gross_return = (split_adjusted + adjusted_dividends) / split_adjusted.shift(1)
    gross_return.iloc[0] = 1.0
    wealth = split_adjusted.iloc[0] * gross_return.cumprod()
    total_return_adjusted = wealth * (split_adjusted.iloc[-1] / wealth.iloc[-1])

    return pd.DataFrame(
        {
            "raw_close": raw_close.astype(float),
            "split_adjusted": split_adjusted,
            "total_return_adjusted": total_return_adjusted,
        },
        index=raw_close.index,
    )
