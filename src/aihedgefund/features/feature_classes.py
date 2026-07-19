"""Separately identifiable OHLCV feature-class blocks for Phase-2 IC triage.

Reversal / Low-Vol / Parkinson Range-Vol are measured as distinct column sets
so each class can be scored alone before any combo. Amihud / illiquidity is
intentionally omitted (hopeless on this large-cap universe; would only burn
``n_trials`` budget).

Production ``FEATURE_COLUMNS`` (36) is **not** mutated here — baseline / gate
stay on the old stack until a class clears Rank-IC >= 0.02.

Reversal sign convention
------------------------
``reversal_k = -1 * past_k_day_return``. A **positive** Rank-IC when the model
uses these columns means the Reversal hypothesis carries (losers rank higher
than recent winners for the forward label).
"""

from __future__ import annotations

from typing import Final

import pandas as pd

from aihedgefund.core.schemas import BarFrame
from aihedgefund.features.indicators import (
    inverse_realized_vol,
    parkinson_volatility,
    reversal,
)
from aihedgefund.features.pipeline import (
    FEATURE_COLUMNS,
    NEW_RAW_FEATURE_COLUMNS,
    FeatureParameters,
    _adjusted_feature_frame,
    add_cross_sectional_features,
    compute_symbol_features,
)
from aihedgefund.features.pit import assert_no_lookahead

# ---------------------------------------------------------------------------
# Raw registries (self-describing names; CS twins derived below)
# ---------------------------------------------------------------------------

REVERSAL_RAW_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "reversal_5",
    "reversal_21",
)

LOW_VOL_RAW_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "inv_ret_std_21",
    "inv_ret_std_63",
)

RANGE_VOL_RAW_FEATURE_COLUMNS: Final[tuple[str, ...]] = ("parkinson_vol_21",)

ALL_NEW_CLASS_RAW_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    *REVERSAL_RAW_FEATURE_COLUMNS,
    *LOW_VOL_RAW_FEATURE_COLUMNS,
    *RANGE_VOL_RAW_FEATURE_COLUMNS,
)


def _with_cs(raw: tuple[str, ...]) -> tuple[str, ...]:
    """Expand raw names with ``_cs_rank`` / ``_cs_zscore`` twins."""
    return tuple(
        name for column in raw for name in (column, f"{column}_cs_rank", f"{column}_cs_zscore")
    )


REVERSAL_FEATURE_COLUMNS: Final[tuple[str, ...]] = _with_cs(REVERSAL_RAW_FEATURE_COLUMNS)
LOW_VOL_FEATURE_COLUMNS: Final[tuple[str, ...]] = _with_cs(LOW_VOL_RAW_FEATURE_COLUMNS)
RANGE_VOL_FEATURE_COLUMNS: Final[tuple[str, ...]] = _with_cs(RANGE_VOL_RAW_FEATURE_COLUMNS)

ALL_NEW_FEATURE_CLASS_COLUMNS: Final[tuple[str, ...]] = (
    *REVERSAL_FEATURE_COLUMNS,
    *LOW_VOL_FEATURE_COLUMNS,
    *RANGE_VOL_FEATURE_COLUMNS,
)

NEW_PLUS_OLD_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    *FEATURE_COLUMNS,
    *ALL_NEW_FEATURE_CLASS_COLUMNS,
)

# Longest lookback among new class features (inv_ret_std_63).
MAX_TRIAGE_FEATURE_WARMUP_BARS: Final[int] = 63

REVERSAL_SIGN_NOTE: Final[str] = (
    "reversal_k = -1 * past_k_day_return; positive OOS Rank-IC means Reversal carries"
)

# Ordered triage arms: label → column subset (raw + CS for that class / combo).
FEATURE_CLASS_CONFIGS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("reversal", REVERSAL_FEATURE_COLUMNS),
    ("low_vol", LOW_VOL_FEATURE_COLUMNS),
    ("range_vol", RANGE_VOL_FEATURE_COLUMNS),
    ("all_new", ALL_NEW_FEATURE_CLASS_COLUMNS),
    ("new_plus_old", NEW_PLUS_OLD_FEATURE_COLUMNS),
)


def compute_symbol_feature_class_raws(frame: pd.DataFrame) -> pd.DataFrame:
    """Causal Reversal / Low-Vol / Parkinson raw columns for one symbol."""
    close = frame["close"]
    return pd.concat(
        (
            reversal(close, 5),
            reversal(close, 21),
            inverse_realized_vol(close, 21),
            inverse_realized_vol(close, 63),
            parkinson_volatility(frame, 21),
        ),
        axis=1,
    )


def build_triage_feature_matrix(
    bars: BarFrame,
    parameters: FeatureParameters | None = None,
) -> pd.DataFrame:
    """Full PIT matrix: production FEATURE_COLUMNS + all new class columns.

    Cross-sectionals are computed for Phase-2 ``NEW_RAW_FEATURE_COLUMNS`` and
    for every new-class raw. Production ``FeaturePipeline.compute`` is unchanged
    and still returns only the 36 ``FEATURE_COLUMNS``.
    """
    params = parameters or FeatureParameters()
    if params != FeatureParameters():
        msg = "build_triage_feature_matrix requires the fixed Phase 1 feature parameters"
        raise ValueError(msg)

    per_symbol: dict[str, pd.DataFrame] = {}
    for symbol, frame in bars.bars.items():
        adjusted = _adjusted_feature_frame(
            frame,
            bars.splits[symbol],
            bars.dividends[symbol],
        )
        production = compute_symbol_features(adjusted, params)
        new_raws = compute_symbol_feature_class_raws(adjusted)
        per_symbol[symbol] = pd.concat((production, new_raws), axis=1)

    matrix = pd.concat(per_symbol, names=("symbol", "timestamp"))
    matrix = matrix.reorder_levels(("timestamp", "symbol")).sort_index()
    cs_bases = (*NEW_RAW_FEATURE_COLUMNS, *ALL_NEW_CLASS_RAW_FEATURE_COLUMNS)
    matrix = add_cross_sectional_features(matrix, cs_bases)
    matrix = matrix.loc[:, list(NEW_PLUS_OLD_FEATURE_COLUMNS)].astype(
        {column: "float64" for column in NEW_PLUS_OLD_FEATURE_COLUMNS}
    )

    timestamps = matrix.index.get_level_values("timestamp")
    provenance = pd.DataFrame(
        {"anchor_timestamp": timestamps, "source_timestamp": timestamps},
        index=matrix.index,
    )
    assert_no_lookahead(provenance, "anchor_timestamp")
    return matrix
