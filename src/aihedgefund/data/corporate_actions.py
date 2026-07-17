"""Pure point-in-time corporate-action transforms for vendor daily closes."""

from __future__ import annotations

from aihedgefund.core.schemas import CorporateActionAdjustment, CorporateActionInput


def adjust_corporate_actions(
    data: CorporateActionInput,
) -> CorporateActionAdjustment:
    """Build start-anchored total-return prices from split-continuous closes.

    Vendor ``close`` from Yahoo (``auto_adjust=False``) is already continuous
    across splits. Multiplying again by the split factor double-adjusts and
    injects ≈log(split) jumps into features and labels.

    Dividends are still applied: Yahoo ``close`` is not dividend-adjusted
    (unlike ``adj_close``). ``splits`` remain on the input DTO for schema/PIT
    alignment but must not rescale prices when ``raw_close`` is already
    split-continuous.

    The transition ending at ``t`` applies only the dividend whose ex-date is
    ``t``. Cumulating those adjusted returns forward prevents an action after
    ``t`` from rewriting any value at or before ``t``.
    """
    raw_close = data.raw_close.astype(float)
    dividends = data.dividends.astype(float)
    # data.splits: schema/PIT only — do not multiply into prices (already continuous).

    split_adjusted = raw_close.copy()

    total_gross_return = raw_close.add(dividends).div(raw_close.shift(1))
    total_gross_return.iloc[0] = 1.0
    as_of_adjusted = raw_close.iloc[0] * total_gross_return.cumprod()

    return CorporateActionAdjustment(
        raw_close=raw_close,
        split_adjusted=split_adjusted,
        as_of_adjusted=as_of_adjusted,
    )
