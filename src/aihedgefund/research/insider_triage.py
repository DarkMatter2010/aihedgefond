"""Phase-2 insider Form 4 IC triage — alone and + ALL_NEW; no CPCV/Gate.

Measures OOS Rank-IC / IC / ICIR on ``BROAD_LIQUID_CANDIDATE_UNIVERSE`` for
``INSIDER_FEATURE_CLASS_CONFIGS``. Production ``FEATURE_COLUMNS`` is unchanged.

Every config × horizon counts as a research trial for DSR ``n_trials``.
PIT: insider columns use ``filed_at`` only (see ``INSIDER_PIT_NOTE``).
"""

from __future__ import annotations

from datetime import date
from typing import Final, Literal

import pandas as pd
from pydantic import Field

from aihedgefund.core.config import Settings
from aihedgefund.core.schemas import (
    BarFrame,
    BoundaryDTO,
    FiniteFloat,
    Form4Frame,
    ICMetricsReport,
    NonEmptyText,
)
from aihedgefund.features.feature_classes import (
    ALL_NEW_FEATURE_CLASS_COLUMNS,
    build_triage_feature_matrix,
)
from aihedgefund.features.insider import (
    INSIDER_FEATURE_CLASS_CONFIGS,
    INSIDER_FEATURE_COLUMNS,
    INSIDER_PIT_NOTE,
    INSIDER_PLUS_ALL_NEW_FEATURE_COLUMNS,
    build_insider_feature_matrix,
)
from aihedgefund.features.pipeline import FEATURE_COLUMNS
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
    BROAD_LIQUID_CANDIDATE_UNIVERSE,
    SURVIVORSHIP_BIAS_NOTE,
)

Interpretation = Literal["candidate_for_gate", "free_data_levers_exhausted"]


class InsiderTriageRow(BoundaryDTO):
    """One OOS metric row: insider config × horizon."""

    class_label: NonEmptyText
    horizon: int = Field(ge=1)
    feature_column_count: int = Field(ge=1)
    metrics: ICMetricsReport
    grinold_sr: FiniteFloat
    rank_ic_above_threshold: bool
    test_start: date


class InsiderTriageReport(BoundaryDTO):
    """Full insider triage table plus interpretation."""

    n_symbols_requested: int = Field(ge=1)
    n_symbols: int = Field(ge=1)
    n_form4_records: int = Field(ge=0)
    n_symbols_without_filings: int = Field(ge=0)
    seed: int = Field(ge=0)
    train_end: date
    configured_test_start: date
    horizons: tuple[int, ...] = Field(min_length=1)
    rows: tuple[InsiderTriageRow, ...] = Field(min_length=1)
    best_rank_ic: FiniteFloat
    best_class_label: NonEmptyText
    best_horizon: int = Field(ge=1)
    interpretation: Interpretation
    interpretation_note: NonEmptyText
    survivorship_bias_note: NonEmptyText
    insider_pit_note: NonEmptyText
    counts_as_research_trial: bool = True
    n_configs_measured: int = Field(ge=1)
    prior_best_all_new_rank_ic_h2: FiniteFloat = 0.0221
    prior_best_all_new_rank_ic_h21: FiniteFloat = 0.0348


def settings_for_insider_triage(settings: Settings, *, horizon: int) -> Settings:
    """Broad universe + matched horizon/embargo; seed/train_end from limits.yaml."""
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
            "universe": BROAD_LIQUID_CANDIDATE_UNIVERSE,
            "research": research,
        }
    )


def interpret_insider_triage(
    rows: tuple[InsiderTriageRow, ...],
    *,
    threshold: float = RANK_IC_MATERIAL_THRESHOLD,
) -> tuple[Interpretation, str]:
    """Rank-IC >= threshold on any cell → gate candidate; else free-data exhausted."""
    if any(row.metrics.rank_ic_mean >= threshold for row in rows):
        best = max(rows, key=lambda r: r.metrics.rank_ic_mean)
        return (
            "candidate_for_gate",
            (
                f"{best.class_label} h={best.horizon} reaches OOS Rank-IC "
                f"{best.metrics.rank_ic_mean:.4f} >= {threshold}. "
                "Next: run that config through the hardened overfitting gate."
            ),
        )
    best = max(rows, key=lambda r: r.metrics.rank_ic_mean)
    return (
        "free_data_levers_exhausted",
        (
            f"No insider config reaches Rank-IC >= {threshold} "
            f"(best={best.class_label} h={best.horizon} "
            f"Rank-IC={best.metrics.rank_ic_mean:.4f}). All identified free-data "
            "levers are exhausted (OHLCV large+small, feature classes, "
            "meta-labeling, Form 4 insider). End search or discuss paid data."
        ),
    )


def _build_combined_matrix(bars: BarFrame, form4: Form4Frame) -> pd.DataFrame:
    """Insider columns joined with ALL_NEW (and production stack unused here)."""
    insider = build_insider_feature_matrix(bars, form4)
    triage = build_triage_feature_matrix(bars)
    all_new = triage.loc[:, list(ALL_NEW_FEATURE_CLASS_COLUMNS)]
    combined = insider.join(all_new, how="inner")
    expected = INSIDER_PLUS_ALL_NEW_FEATURE_COLUMNS
    missing = [c for c in expected if c not in combined.columns]
    if missing:
        msg = f"combined insider matrix missing columns: {missing}"
        raise ValueError(msg)
    return combined.loc[:, list(expected)]


def run_insider_form4_triage(
    bars: BarFrame,
    form4: Form4Frame,
    settings: Settings,
) -> InsiderTriageReport:
    """Train/evaluate insider alone and insider+all_new × triage horizons."""
    requested = set(BROAD_LIQUID_CANDIDATE_UNIVERSE)
    actual = set(settings.universe)
    if not actual.issubset(requested):
        msg = "settings.universe must be a subset of BROAD_LIQUID_CANDIDATE_UNIVERSE"
        raise ValueError(msg)

    bar_symbols = tuple(sorted(bars.bars))
    if not bar_symbols:
        msg = "bars must contain at least one symbol"
        raise ValueError(msg)

    _ = FEATURE_COLUMNS  # production stack untouched; referenced for honesty
    configured_test_start = settings.research.test_start
    train_end = settings.research.train_end

    matrices: dict[str, pd.DataFrame] = {
        "insider": build_insider_feature_matrix(bars, form4),
        "insider_plus_all_new": _build_combined_matrix(bars, form4),
    }
    config_columns = dict(INSIDER_FEATURE_CLASS_CONFIGS)
    rows: list[InsiderTriageRow] = []

    for class_label, feature_columns in INSIDER_FEATURE_CLASS_CONFIGS:
        matrix = matrices[class_label]
        subset = matrix.loc[:, list(feature_columns)]
        for horizon in TRIAGE_HORIZONS:
            labels, _meta = make_forward_return_labels(bars, horizon=horizon)
            dataset = assemble_baseline_dataset(
                subset,
                labels,
                horizon=horizon,
                feature_columns=feature_columns,
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
            run_settings = settings.model_copy(
                update={"universe": bar_symbols, "research": research}
            )
            train, test, _split = time_embargo_split(
                dataset,
                train_end=research.train_end,
                test_start=research.test_start,
                embargo_days=research.embargo_days,
                horizon=horizon,
            )
            hyperparams = build_hyperparams(research)
            model_hash = compute_model_hash(
                features=feature_columns,
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
                InsiderTriageRow(
                    class_label=class_label,
                    horizon=horizon,
                    feature_column_count=len(feature_columns),
                    metrics=metrics,
                    grinold_sr=grinold,
                    rank_ic_above_threshold=metrics.rank_ic_mean >= RANK_IC_MATERIAL_THRESHOLD,
                    test_start=test_start,
                )
            )

    row_tuple = tuple(rows)
    best = max(row_tuple, key=lambda r: r.metrics.rank_ic_mean)
    interpretation, note = interpret_insider_triage(row_tuple)
    assert config_columns["insider"] == INSIDER_FEATURE_COLUMNS
    assert config_columns["insider_plus_all_new"] == INSIDER_PLUS_ALL_NEW_FEATURE_COLUMNS
    return InsiderTriageReport(
        n_symbols_requested=len(BROAD_LIQUID_CANDIDATE_UNIVERSE),
        n_symbols=len(bar_symbols),
        n_form4_records=len(form4.records),
        n_symbols_without_filings=len(form4.symbols_without_filings),
        seed=settings.research.seed,
        train_end=train_end,
        configured_test_start=configured_test_start,
        horizons=TRIAGE_HORIZONS,
        rows=row_tuple,
        best_rank_ic=best.metrics.rank_ic_mean,
        best_class_label=best.class_label,
        best_horizon=best.horizon,
        interpretation=interpretation,
        interpretation_note=note,
        survivorship_bias_note=SURVIVORSHIP_BIAS_NOTE,
        insider_pit_note=INSIDER_PIT_NOTE,
        counts_as_research_trial=True,
        n_configs_measured=len(row_tuple),
    )
