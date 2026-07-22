"""Phase-2 small-cap universe IC diagnostic — no CPCV/Gate.

Same feature stack as the best broad triage cell
(`ALL_NEW_FEATURE_CLASS_COLUMNS`) on `SMALL_CAP_CANDIDATE_UNIVERSE`.
Horizons h=2 and h=21. Counts as **one** research trial (best-horizon Grinold SR).

Interpretation (this probe only): Rank-IC >= 0.02 on **any** horizon →
candidate for a separate gate prompt; otherwise free OHLCV signal search is
exhausted (large + small universe, all feature classes, meta-labeling done).
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Final, Literal

from pydantic import Field

from aihedgefund.core.config import Settings
from aihedgefund.core.schemas import (
    BarFrame,
    BoundaryDTO,
    FiniteFloat,
    ICMetricsReport,
    NonEmptyText,
)
from aihedgefund.features.feature_classes import (
    ALL_NEW_FEATURE_CLASS_COLUMNS,
    build_triage_feature_matrix,
)
from aihedgefund.research.baseline import predict_scores, train_baseline
from aihedgefund.research.dataset import assemble_baseline_dataset
from aihedgefund.research.feature_class_triage import (
    RANK_IC_MATERIAL_THRESHOLD,
    TRIAGE_HORIZONS,
    _test_start_for_dataset,
    grinold_trial_sharpe,
    test_start_for_horizon,
)
from aihedgefund.research.forward_labels import make_forward_return_labels
from aihedgefund.research.metrics import compute_ic_metrics
from aihedgefund.research.model_hash import compute_model_hash
from aihedgefund.research.run_baseline import build_hyperparams
from aihedgefund.research.split import time_embargo_split
from aihedgefund.research.universes import (
    SMALL_CAP_CANDIDATE_UNIVERSE,
    SMALL_CAP_SURVIVORSHIP_BIAS_NOTE,
)

DIAGNOSTIC_HORIZONS: Final[tuple[int, ...]] = TRIAGE_HORIZONS
FEATURE_COLUMNS: Final[tuple[str, ...]] = ALL_NEW_FEATURE_CLASS_COLUMNS

# Broad all_new reference (P2 feature-class triage live 2026-07-19).
BROAD_ALL_NEW_H2_RANK_IC: Final[float] = 0.022127
BROAD_ALL_NEW_H21_RANK_IC: Final[float] = 0.034849

Interpretation = Literal["candidate_for_gate", "free_ohlcv_search_exhausted"]


class SmallCapUniverseRow(BoundaryDTO):
    """OOS metrics for ALL_NEW on the small-cap universe at one horizon."""

    horizon: int = Field(ge=1)
    feature_column_count: int = Field(ge=1)
    metrics: ICMetricsReport
    grinold_sr: FiniteFloat
    rank_ic_above_threshold: bool
    test_start: date
    broad_all_new_rank_ic: FiniteFloat


class SmallCapUniverseReport(BoundaryDTO):
    """Small-cap universe diagnostic table plus interpretation."""

    n_symbols_requested: int = Field(ge=1)
    n_symbols: int = Field(ge=1)
    n_symbols_dropped: int = Field(ge=0)
    drop_rate: Annotated[float, Field(ge=0.0, le=1.0)]
    seed: int = Field(ge=0)
    train_end: date
    configured_test_start: date
    horizons: tuple[int, ...] = Field(min_length=1)
    feature_columns: tuple[NonEmptyText, ...] = Field(min_length=1)
    rows: tuple[SmallCapUniverseRow, ...] = Field(min_length=1)
    best_rank_ic: FiniteFloat
    best_horizon: int = Field(ge=1)
    best_grinold_sr: FiniteFloat
    interpretation: Interpretation
    interpretation_note: NonEmptyText
    survivorship_bias_note: NonEmptyText
    counts_as_research_trial: bool = True


def settings_for_small_cap_universe_diagnostic(settings: Settings, *, horizon: int) -> Settings:
    """Small-cap universe + matched horizon/embargo; seed/train_end from limits.yaml."""
    if horizon < 1:
        msg = "horizon must be >= 1"
        raise ValueError(msg)
    test_start = test_start_for_horizon(
        settings.research.train_end,
        settings.research.test_start,
        horizon,
    )
    research = settings.research.model_copy(
        update={
            "horizon": horizon,
            "embargo_days": horizon,
            "test_start": test_start,
        }
    )
    return settings.model_copy(
        update={
            "universe": SMALL_CAP_CANDIDATE_UNIVERSE,
            "research": research,
        }
    )


def interpret_small_cap_universe(
    rows: tuple[SmallCapUniverseRow, ...],
    *,
    threshold: float = RANK_IC_MATERIAL_THRESHOLD,
) -> tuple[Interpretation, str]:
    """Rank-IC >= threshold on any horizon → gate candidate; else search stop."""
    if not rows:
        msg = "rows must be non-empty"
        raise ValueError(msg)

    above = [r for r in rows if r.metrics.rank_ic_mean >= threshold]
    best = max(rows, key=lambda r: r.metrics.rank_ic_mean)

    if above:
        hs = ", ".join(f"h={r.horizon} Rank-IC={r.metrics.rank_ic_mean:.4f}" for r in above)
        return (
            "candidate_for_gate",
            (
                f"Small-cap ALL_NEW reaches OOS Rank-IC >= {threshold} on {hs}. "
                "Next: run this config through the hardened overfitting gate "
                "(separate prompt; this diagnostic is not a gate)."
            ),
        )

    return (
        "free_ohlcv_search_exhausted",
        (
            f"No small-cap horizon reaches Rank-IC >= {threshold} "
            f"(best h={best.horizon} Rank-IC={best.metrics.rank_ic_mean:.4f}). "
            "Gratis-OHLCV signal search is exhausted: large + small universe, "
            "all feature classes, and meta-labeling are done. Do not propose "
            "further free-data OHLCV variants."
        ),
    )


def _broad_reference_rank_ic(horizon: int) -> float:
    if horizon == 2:
        return BROAD_ALL_NEW_H2_RANK_IC
    if horizon == 21:
        return BROAD_ALL_NEW_H21_RANK_IC
    msg = f"no broad all_new reference for horizon={horizon}"
    raise ValueError(msg)


def run_small_cap_universe_diagnostic(
    bars: BarFrame,
    settings: Settings,
    *,
    n_symbols_dropped: int | None = None,
) -> SmallCapUniverseReport:
    """Train/evaluate ALL_NEW on small-cap bars for both diagnostic horizons.

    `settings.universe` must be a subset of `SMALL_CAP_CANDIDATE_UNIVERSE`
    (download drops allowed). Horizon on `settings.research` is ignored —
    both `DIAGNOSTIC_HORIZONS` are always evaluated with matched embargo.

    `n_symbols_dropped` is the ingest quality/missing count from the live
    script; when omitted, inferred as `requested - kept`.
    """
    requested = set(SMALL_CAP_CANDIDATE_UNIVERSE)
    actual = set(settings.universe)
    if not actual.issubset(requested):
        msg = "settings.universe must be a subset of SMALL_CAP_CANDIDATE_UNIVERSE"
        raise ValueError(msg)

    bar_symbols = tuple(sorted(bars.bars))
    if not bar_symbols:
        msg = "bars must contain at least one symbol"
        raise ValueError(msg)

    n_requested = len(SMALL_CAP_CANDIDATE_UNIVERSE)
    n_kept = len(bar_symbols)
    if n_symbols_dropped is None:
        n_dropped = max(0, n_requested - n_kept)
    else:
        if n_symbols_dropped < 0:
            msg = "n_symbols_dropped must be >= 0"
            raise ValueError(msg)
        n_dropped = n_symbols_dropped
    drop_rate = float(n_dropped) / float(n_requested) if n_requested else 0.0

    configured_test_start = settings.research.test_start
    train_end = settings.research.train_end
    matrix = build_triage_feature_matrix(bars)
    subset = matrix.loc[:, list(FEATURE_COLUMNS)]
    rows: list[SmallCapUniverseRow] = []

    for horizon in DIAGNOSTIC_HORIZONS:
        labels, _meta = make_forward_return_labels(bars, horizon=horizon)
        dataset = assemble_baseline_dataset(
            subset,
            labels,
            horizon=horizon,
            feature_columns=FEATURE_COLUMNS,
        )
        test_start = _test_start_for_dataset(
            dataset,
            train_end=train_end,
            configured_test_start=configured_test_start,
            horizon=horizon,
        )
        research = settings.research.model_copy(
            update={
                "horizon": horizon,
                "embargo_days": horizon,
                "test_start": test_start,
            }
        )
        run_settings = settings.model_copy(update={"universe": bar_symbols, "research": research})
        train, test, _split = time_embargo_split(
            dataset,
            train_end=research.train_end,
            test_start=research.test_start,
            embargo_days=research.embargo_days,
            horizon=horizon,
        )
        hyperparams = build_hyperparams(research)
        model_hash = compute_model_hash(
            features=FEATURE_COLUMNS,
            hyperparameters=hyperparams,
            universe=run_settings.universe,
            start=run_settings.start,
            end=run_settings.end,
            frequency=run_settings.frequency,
            seed=research.seed,
        )
        model = train_baseline(
            train,
            params={k: v for k, v in hyperparams.items() if k != "num_boost_round"},
            num_boost_round=research.num_boost_round,
        )
        predictions = predict_scores(model, test.features, model_hash=model_hash)
        metrics = compute_ic_metrics(
            predictions.scores,
            test.label,
            ic_positive_threshold=research.ic_positive_threshold,
            min_cs_breadth_for_reliable_ic=research.min_cs_breadth_for_reliable_ic,
        )
        grinold = grinold_trial_sharpe(
            rank_ic=metrics.rank_ic_mean,
            median_cs_breadth=metrics.median_cs_breadth,
        )
        rows.append(
            SmallCapUniverseRow(
                horizon=horizon,
                feature_column_count=len(FEATURE_COLUMNS),
                metrics=metrics,
                grinold_sr=grinold,
                rank_ic_above_threshold=metrics.rank_ic_mean >= RANK_IC_MATERIAL_THRESHOLD,
                test_start=test_start,
                broad_all_new_rank_ic=_broad_reference_rank_ic(horizon),
            )
        )

    row_tuple = tuple(rows)
    best = max(row_tuple, key=lambda r: r.metrics.rank_ic_mean)
    interpretation, note = interpret_small_cap_universe(row_tuple)
    return SmallCapUniverseReport(
        n_symbols_requested=n_requested,
        n_symbols=n_kept,
        n_symbols_dropped=n_dropped,
        drop_rate=drop_rate,
        seed=settings.research.seed,
        train_end=train_end,
        configured_test_start=configured_test_start,
        horizons=DIAGNOSTIC_HORIZONS,
        feature_columns=FEATURE_COLUMNS,
        rows=row_tuple,
        best_rank_ic=best.metrics.rank_ic_mean,
        best_horizon=best.horizon,
        best_grinold_sr=best.grinold_sr,
        interpretation=interpretation,
        interpretation_note=note,
        survivorship_bias_note=SMALL_CAP_SURVIVORSHIP_BIAS_NOTE,
        counts_as_research_trial=True,
    )
