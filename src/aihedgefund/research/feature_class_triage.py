"""Phase-2 feature-class IC triage: Reversal / Low-Vol / Range — no CPCV/Gate.

Measures OOS Rank-IC / IC / ICIR per feature class (and combos) on
``BROAD_LIQUID_CANDIDATE_UNIVERSE``. Production ``FEATURE_COLUMNS`` is unchanged;
this path builds an extended triage matrix and subsets columns per config.

Every config × horizon counts as a research trial for DSR ``n_trials``.

Split note: ``limits.yaml`` seed / ``train_end`` / configured ``test_start`` are
used. When the calendar / trading-bar gap is too short for the horizon embargo
(e.g. h=21 vs an 11-day gap), ``test_start`` is pushed to
``train_end + (horizon + 1)`` business days so label windows cannot overlap —
mechanical, not a feature retune.
"""

from __future__ import annotations

from datetime import date
from math import sqrt
from typing import Final, Literal

import pandas as pd
from pandas.tseries.offsets import BDay
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
    FEATURE_CLASS_CONFIGS,
    REVERSAL_SIGN_NOTE,
    build_triage_feature_matrix,
)
from aihedgefund.research.baseline import predict_scores, train_baseline
from aihedgefund.research.dataset import assemble_baseline_dataset
from aihedgefund.research.forward_labels import make_forward_return_labels
from aihedgefund.research.metrics import compute_ic_metrics
from aihedgefund.research.model_hash import compute_model_hash
from aihedgefund.research.run_baseline import build_hyperparams
from aihedgefund.research.split import time_embargo_split
from aihedgefund.research.universes import (
    BROAD_LIQUID_CANDIDATE_UNIVERSE,
    SURVIVORSHIP_BIAS_NOTE,
)

TRIAGE_HORIZONS: Final[tuple[int, ...]] = (2, 21)
RANK_IC_MATERIAL_THRESHOLD: Final[float] = 0.02

Interpretation = Literal["candidate_for_gate", "ohlcv_features_dead"]


class FeatureClassTriageRow(BoundaryDTO):
    """One OOS metric row: feature-class config × horizon."""

    class_label: NonEmptyText
    horizon: int = Field(ge=1)
    feature_column_count: int = Field(ge=1)
    metrics: ICMetricsReport
    grinold_sr: FiniteFloat
    rank_ic_above_threshold: bool
    test_start: date


class FeatureClassTriageReport(BoundaryDTO):
    """Full triage table plus interpretation for the next phase."""

    n_symbols_requested: int = Field(ge=1)
    n_symbols: int = Field(ge=1)
    seed: int = Field(ge=0)
    train_end: date
    configured_test_start: date
    horizons: tuple[int, ...] = Field(min_length=1)
    rows: tuple[FeatureClassTriageRow, ...] = Field(min_length=1)
    best_rank_ic: FiniteFloat
    best_class_label: NonEmptyText
    best_horizon: int = Field(ge=1)
    interpretation: Interpretation
    interpretation_note: NonEmptyText
    survivorship_bias_note: NonEmptyText
    reversal_sign_note: NonEmptyText
    counts_as_research_trial: bool = True
    n_configs_measured: int = Field(ge=1)


def grinold_trial_sharpe(*, rank_ic: float, median_cs_breadth: float) -> float:
    """Grinold IR proxy ``IC * sqrt(N)`` using Rank-IC and that run's median breadth."""
    if median_cs_breadth < 0.0:
        msg = "median_cs_breadth must be >= 0"
        raise ValueError(msg)
    return float(rank_ic) * sqrt(float(median_cs_breadth))


def test_start_for_horizon(
    train_end: date,
    configured_test_start: date,
    horizon: int,
    *,
    trading_dates: tuple[date, ...] | None = None,
) -> date:
    """Keep configured test_start unless embargo / label-gap requires later.

    ``time_embargo_split`` needs (1) calendar gap >= embargo_days (== horizon)
    and (2) trading-bar gap > horizon so label windows do not overlap.

    When ``trading_dates`` (sorted unique session dates) is supplied, the
    minimum start is taken from the real calendar: first session strictly after
    the last train session by more than ``horizon`` bars. Otherwise fall back
    to ``train_end + (horizon + 5)`` business days (holiday buffer).
    """
    if horizon < 1:
        msg = "horizon must be >= 1"
        raise ValueError(msg)
    if configured_test_start <= train_end:
        msg = "configured_test_start must be later than train_end"
        raise ValueError(msg)

    if trading_dates is not None:
        if len(trading_dates) < horizon + 2:
            msg = "trading_dates too short for requested horizon"
            raise ValueError(msg)
        train_positions = [i for i, d in enumerate(trading_dates) if d <= train_end]
        if not train_positions:
            msg = "no trading dates on or before train_end"
            raise ValueError(msg)
        last_train_pos = train_positions[-1]
        # Need bar_gap = test_pos - last_train_pos > horizon.
        min_test_pos = last_train_pos + horizon + 1
        if min_test_pos >= len(trading_dates):
            msg = "not enough trading dates after train_end for horizon"
            raise ValueError(msg)
        min_start = trading_dates[min_test_pos]
    else:
        min_start = (pd.Timestamp(train_end) + BDay(horizon + 5)).date()

    if configured_test_start >= min_start:
        return configured_test_start
    return min_start


def _trading_dates_from_index(timestamps: pd.Index) -> tuple[date, ...]:
    """Unique session dates from a pandas DatetimeIndex-like, sorted."""
    return tuple(sorted({pd.Timestamp(ts).date() for ts in timestamps}))


def _test_start_for_dataset(
    dataset: object,
    *,
    train_end: date,
    configured_test_start: date,
    horizon: int,
) -> date:
    """Earliest test_start that keeps per-symbol bar_gap > horizon on ``dataset``.

    Takes the max over symbols of each symbol's calendar-derived minimum so a
    sparse name cannot violate ``_assert_no_label_window_overlap``. Symbols with
    no pre-``train_end`` rows are skipped (they never enter the train split).
    """
    from datetime import timedelta

    from aihedgefund.core.schemas import BaselineDataset

    if not isinstance(dataset, BaselineDataset):
        msg = "dataset must be a BaselineDataset"
        raise TypeError(msg)

    symbols = sorted(set(dataset.features.index.get_level_values("symbol")))
    required: list[date] = [
        configured_test_start,
        train_end + timedelta(days=horizon),
    ]
    for symbol in symbols:
        symbol_ts = dataset.features.xs(symbol, level="symbol").index
        dates = _trading_dates_from_index(symbol_ts)
        if not any(d <= train_end for d in dates):
            continue
        required.append(
            test_start_for_horizon(
                train_end,
                configured_test_start,
                horizon,
                trading_dates=dates,
            )
        )
    return max(required)


def settings_for_feature_class_triage(settings: Settings, *, horizon: int) -> Settings:
    """Broad universe + matched horizon/embargo; seed/train_end from limits.yaml.

    ``test_start`` may be pushed later so the calendar gap satisfies embargo.
    """
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


def interpret_feature_class_triage(
    rows: tuple[FeatureClassTriageRow, ...],
    *,
    threshold: float = RANK_IC_MATERIAL_THRESHOLD,
) -> tuple[Interpretation, str]:
    """Apply the registered interpretation rule (relative diagnosis only)."""
    by_class: dict[str, dict[int, float]] = {}
    for row in rows:
        by_class.setdefault(row.class_label, {})[row.horizon] = row.metrics.rank_ic_mean

    stable_candidates: list[str] = []
    for label, horizon_ics in by_class.items():
        if all(h in horizon_ics for h in TRIAGE_HORIZONS) and all(
            horizon_ics[h] >= threshold for h in TRIAGE_HORIZONS
        ):
            stable_candidates.append(label)

    if stable_candidates:
        names = ", ".join(stable_candidates)
        return (
            "candidate_for_gate",
            (
                f"Class(es) {names} reach OOS Rank-IC >= {threshold} on both "
                f"h={list(TRIAGE_HORIZONS)}. Next: run that config through the "
                "hardened overfitting gate (DSR >= 0.95)."
            ),
        )

    best = max(rows, key=lambda r: r.metrics.rank_ic_mean)
    return (
        "ohlcv_features_dead",
        (
            f"No class/combo reaches Rank-IC >= {threshold} on both horizons "
            f"(best={best.class_label} h={best.horizon} "
            f"Rank-IC={best.metrics.rank_ic_mean:.4f}). Free OHLCV/large-cap "
            "looks dead; recommend pipeline Durchstich with the best weak "
            "signal instead of further feature hunting."
        ),
    )


def run_feature_class_triage(
    bars: BarFrame,
    settings: Settings,
) -> FeatureClassTriageReport:
    """Train/evaluate each feature-class config × triage horizon; return table.

    ``settings.universe`` must be a subset of ``BROAD_LIQUID_CANDIDATE_UNIVERSE``
    (download drops allowed). Horizon on ``settings.research`` is ignored —
    both ``TRIAGE_HORIZONS`` are always evaluated with matched embargo.
    """
    requested = set(BROAD_LIQUID_CANDIDATE_UNIVERSE)
    actual = set(settings.universe)
    if not actual.issubset(requested):
        msg = "settings.universe must be a subset of BROAD_LIQUID_CANDIDATE_UNIVERSE"
        raise ValueError(msg)

    bar_symbols = tuple(sorted(bars.bars))
    if not bar_symbols:
        msg = "bars must contain at least one symbol"
        raise ValueError(msg)

    configured_test_start = settings.research.test_start
    train_end = settings.research.train_end
    matrix = build_triage_feature_matrix(bars)
    rows: list[FeatureClassTriageRow] = []

    for class_label, feature_columns in FEATURE_CLASS_CONFIGS:
        subset = matrix.loc[:, list(feature_columns)]
        for horizon in TRIAGE_HORIZONS:
            # Labels/features first so test_start uses the post-NaN session calendar
            # that ``time_embargo_split`` will validate (per-symbol bar gaps).
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
                FeatureClassTriageRow(
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
    interpretation, note = interpret_feature_class_triage(row_tuple)
    return FeatureClassTriageReport(
        n_symbols_requested=len(BROAD_LIQUID_CANDIDATE_UNIVERSE),
        n_symbols=len(bar_symbols),
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
        reversal_sign_note=REVERSAL_SIGN_NOTE,
        counts_as_research_trial=True,
        n_configs_measured=len(row_tuple),
    )
