"""Cross-sectional IC / Rank-IC / ICIR metrics for Phase-2 diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aihedgefund.core.schemas import ICMetricsReport

BREADTH_WARNING = (
    "median cross-sectional breadth < 30 symbols/date; "
    "IC/ICIR only weakly informative"
)


def compute_ic_metrics(
    scores: pd.Series,
    forward_returns: pd.Series,
    *,
    ic_positive_threshold: float,
    min_cs_breadth_for_reliable_ic: int = 30,
) -> ICMetricsReport:
    """Compute mean cross-sectional Pearson/Spearman IC and ICIR.

    IC: per-date Pearson correlation across symbols, then mean over dates.
    Rank IC: per-date Spearman correlation, then mean.
    ICIR: mean(IC series) / std(IC series); ``None`` when std is 0.
    """
    if not scores.index.equals(forward_returns.index):
        msg = "scores and forward_returns must share an identical index"
        raise ValueError(msg)
    if list(scores.index.names) != ["timestamp", "symbol"]:
        msg = "metric index names must be ('timestamp', 'symbol')"
        raise ValueError(msg)

    frame = pd.DataFrame({"score": scores, "fwd": forward_returns}).sort_index()
    ic_values: list[float] = []
    rank_ic_values: list[float] = []
    breadths: list[int] = []

    for _timestamp, group in frame.groupby(level="timestamp", sort=True):
        if len(group) < 2:
            continue
        breadths.append(int(len(group)))
        pearson = group["score"].corr(group["fwd"], method="pearson")
        spearman = group["score"].corr(group["fwd"], method="spearman")
        if pd.notna(pearson):
            ic_values.append(float(pearson))
        if pd.notna(spearman):
            rank_ic_values.append(float(spearman))

    if not ic_values or not rank_ic_values or not breadths:
        msg = "insufficient cross-sections to compute IC metrics"
        raise ValueError(msg)

    ic_mean = float(np.mean(ic_values))
    rank_ic_mean = float(np.mean(rank_ic_values))
    ic_std = float(np.std(ic_values, ddof=0))
    rank_ic_std = float(np.std(rank_ic_values, ddof=0))
    icir = (ic_mean / ic_std) if ic_std > 0.0 else None
    rank_icir = (rank_ic_mean / rank_ic_std) if rank_ic_std > 0.0 else None
    median_breadth = float(np.median(breadths))
    breadth_warning = median_breadth < float(min_cs_breadth_for_reliable_ic)
    warnings = (BREADTH_WARNING,) if breadth_warning else ()

    return ICMetricsReport(
        ic_mean=ic_mean,
        rank_ic_mean=rank_ic_mean,
        icir=icir,
        rank_icir=rank_icir,
        median_cs_breadth=median_breadth,
        cs_breadth_warning=breadth_warning,
        ic_materially_positive=ic_mean > ic_positive_threshold,
        ic_positive_threshold=ic_positive_threshold,
        n_dates=len(ic_values),
        warnings=warnings,
    )
