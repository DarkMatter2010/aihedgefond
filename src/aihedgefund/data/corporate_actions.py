"""Pure point-in-time corporate-action transforms for raw daily closes."""

from __future__ import annotations

from aihedgefund.core.schemas import CorporateActionAdjustment, CorporateActionInput


def adjust_corporate_actions(
    data: CorporateActionInput,
) -> CorporateActionAdjustment:
    """Build start-anchored prices using only actions known at each row.

    The transition ending at ``t`` applies only the split and dividend whose
    ex-date is ``t``. Cumulating those adjusted returns forward prevents an
    action after ``t`` from rewriting any value at or before ``t``.
    """
    raw_close = data.raw_close.astype(float)
    split_events = data.splits.mask(data.splits == 0.0, 1.0).astype(float)
    dividends = data.dividends.astype(float)

    split_gross_return = raw_close.mul(split_events).div(raw_close.shift(1))
    split_gross_return.iloc[0] = 1.0
    split_adjusted = raw_close.iloc[0] * split_gross_return.cumprod()

    total_gross_return = raw_close.add(dividends).mul(split_events).div(raw_close.shift(1))
    total_gross_return.iloc[0] = 1.0
    as_of_adjusted = raw_close.iloc[0] * total_gross_return.cumprod()

    return CorporateActionAdjustment(
        raw_close=raw_close,
        split_adjusted=split_adjusted,
        as_of_adjusted=as_of_adjusted,
    )
