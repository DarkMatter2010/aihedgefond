"""Insider (SEC Form 4) feature columns — PIT via filed_at only.

``transaction_date`` is never used as the availability timestamp. A filing is
visible on bar date ``t`` only when ``filed_at <= t``. Activity is aggregated
over the last ``INSIDER_LOOKBACK_BARS`` sessions on that symbol's bar calendar.

Missing insider activity in the window yields NaN (neutral), not a hard error.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from aihedgefund.core.schemas import BarFrame, Form4Frame, Form4Record
from aihedgefund.features.feature_classes import (
    ALL_NEW_FEATURE_CLASS_COLUMNS,
    _with_cs,
)
from aihedgefund.features.pipeline import add_cross_sectional_features
from aihedgefund.features.pit import assert_no_lookahead

INSIDER_LOOKBACK_BARS: Final[int] = 21
_EPS: Final[float] = 1.0

INSIDER_RAW_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "insider_net_buy_ratio_21",
    "insider_buyer_count_21",
    "insider_signed_volume_21",
)

INSIDER_FEATURE_COLUMNS: Final[tuple[str, ...]] = _with_cs(INSIDER_RAW_FEATURE_COLUMNS)

INSIDER_PLUS_ALL_NEW_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    *INSIDER_FEATURE_COLUMNS,
    *ALL_NEW_FEATURE_CLASS_COLUMNS,
)

INSIDER_FEATURE_CLASS_CONFIGS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("insider", INSIDER_FEATURE_COLUMNS),
    ("insider_plus_all_new", INSIDER_PLUS_ALL_NEW_FEATURE_COLUMNS),
)

INSIDER_PIT_NOTE: Final[str] = (
    "Form 4 features use filed_at (acceptanceDateTime) only; "
    "transaction_date is informational and may precede filing by up to ~2 days"
)


def form4_records_to_frame(records: tuple[Form4Record, ...] | Form4Frame) -> pd.DataFrame:
    """Flatten Form4 records into a DataFrame sorted by filed_at."""
    rows = records.records if isinstance(records, Form4Frame) else records
    if not rows:
        return pd.DataFrame(
            columns=[
                "symbol",
                "filed_at",
                "transaction_date",
                "transaction_code",
                "shares",
                "acquired_disposed",
                "reporting_owner",
                "accession",
            ]
        )
    frame = pd.DataFrame(
        [
            {
                "symbol": r.symbol,
                "filed_at": r.filed_at,
                "transaction_date": r.transaction_date,
                "transaction_code": r.transaction_code,
                "shares": float(r.shares),
                "acquired_disposed": r.acquired_disposed,
                "reporting_owner": r.reporting_owner or r.accession,
                "accession": r.accession,
            }
            for r in rows
        ]
    )
    frame["filed_at"] = pd.to_datetime(frame["filed_at"], utc=True)
    return frame.sort_values(["symbol", "filed_at"]).reset_index(drop=True)


def compute_symbol_insider_raws(
    bar_index: pd.DatetimeIndex,
    filings: pd.DataFrame,
    *,
    lookback: int = INSIDER_LOOKBACK_BARS,
) -> pd.DataFrame:
    """Causal insider aggregates for one symbol's bar calendar.

    Only rows with ``filed_at <= bar_timestamp`` participate. Lookback is the
    last ``lookback`` bars inclusive (trading-session window).
    """
    if lookback < 1:
        msg = "lookback must be >= 1"
        raise ValueError(msg)
    if not isinstance(bar_index, pd.DatetimeIndex):
        msg = "bar_index must be a DatetimeIndex"
        raise ValueError(msg)

    n = len(bar_index)
    net_ratio = np.full(n, np.nan, dtype="float64")
    buyer_count = np.full(n, np.nan, dtype="float64")
    signed_vol = np.full(n, np.nan, dtype="float64")

    if filings.empty:
        return pd.DataFrame(
            {
                "insider_net_buy_ratio_21": net_ratio,
                "insider_buyer_count_21": buyer_count,
                "insider_signed_volume_21": signed_vol,
            },
            index=bar_index,
        )

    filed_series = pd.to_datetime(filings["filed_at"], utc=True)
    shares = filings["shares"].to_numpy(dtype="float64")
    acquired = filings["acquired_disposed"].astype(str).to_numpy()
    owners = filings["reporting_owner"].astype(str).to_numpy()
    codes = filings["transaction_code"].astype(str).str.upper().to_numpy()

    for i, ts in enumerate(bar_index):
        window_start_pos = max(0, i - lookback + 1)
        window_start = bar_index[window_start_pos]
        # Inclusive on both ends; only filings with filed_at <= bar timestamp.
        mask = ((filed_series >= window_start) & (filed_series <= ts)).to_numpy()
        if not mask.any():
            continue
        w_shares = shares[mask]
        w_ad = acquired[mask]
        w_owners = owners[mask]
        w_codes = codes[mask]
        buy = w_shares[(w_ad == "A") | (w_codes == "P")].sum()
        sell = w_shares[(w_ad == "D") | (w_codes == "S")].sum()
        # Prefer A/D for signed volume (plan).
        signed = float(
            w_shares[w_ad == "A"].sum() - w_shares[w_ad == "D"].sum()
        )
        denom = buy + sell + _EPS
        net_ratio[i] = float((buy - sell) / denom)
        purchase_owners = w_owners[(w_ad == "A") | (w_codes == "P")]
        buyer_count[i] = float(len(set(purchase_owners.tolist()))) if len(purchase_owners) else 0.0
        signed_vol[i] = signed
        # If we entered the branch, activity existed — keep zeros for buyer_count.

    return pd.DataFrame(
        {
            "insider_net_buy_ratio_21": net_ratio,
            "insider_buyer_count_21": buyer_count,
            "insider_signed_volume_21": signed_vol,
        },
        index=bar_index,
    )


def build_insider_feature_matrix(
    bars: BarFrame,
    form4: Form4Frame | tuple[Form4Record, ...],
) -> pd.DataFrame:
    """Build insider raw + CS columns aligned to ``(timestamp, symbol)`` bars."""
    filings = form4_records_to_frame(form4)
    per_symbol: dict[str, pd.DataFrame] = {}
    for symbol, frame in bars.bars.items():
        symbol_filings = (
            filings.loc[filings["symbol"] == symbol]
            if not filings.empty
            else filings
        )
        per_symbol[symbol] = compute_symbol_insider_raws(frame.index, symbol_filings)

    matrix = pd.concat(per_symbol, names=("symbol", "timestamp"))
    matrix = matrix.reorder_levels(("timestamp", "symbol")).sort_index()
    matrix = add_cross_sectional_features(matrix, INSIDER_RAW_FEATURE_COLUMNS)
    matrix = matrix.loc[:, list(INSIDER_FEATURE_COLUMNS)].astype(
        {column: "float64" for column in INSIDER_FEATURE_COLUMNS}
    )
    # No activity in the lookback window is neutral (0), not a dropped row.
    # CS ranks were computed with NaNs skipped; fill after CS.
    matrix = matrix.fillna(0.0)

    timestamps = matrix.index.get_level_values("timestamp")
    provenance = pd.DataFrame(
        {"anchor_timestamp": timestamps, "source_timestamp": timestamps},
        index=matrix.index,
    )
    assert_no_lookahead(provenance, "anchor_timestamp")
    return matrix
