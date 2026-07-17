"""Multi-horizon IC sweep over a single loaded bar/feature set."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, timedelta

import pandas as pd

from aihedgefund.core.config import Settings
from aihedgefund.core.schemas import (
    BarFrame,
    BaselineDataset,
    HorizonSweepReport,
    HorizonSweepRow,
    ModelArtifactMetadata,
    Phase2Sidecar,
)
from aihedgefund.features.pipeline import FEATURE_COLUMNS
from aihedgefund.research.adapters.filesystem import (
    MODEL_FILENAME,
    FilesystemModelArtifactAdapter,
)
from aihedgefund.research.baseline import predict_scores, train_baseline
from aihedgefund.research.dataset import assemble_baseline_dataset
from aihedgefund.research.forward_labels import make_forward_return_labels
from aihedgefund.research.metrics import compute_ic_metrics
from aihedgefund.research.model_hash import compute_model_hash
from aihedgefund.research.run_baseline import (
    build_hyperparams,
    lib_versions,
    resolve_git_commit,
    write_sidecar,
)
from aihedgefund.research.split import time_embargo_split

DEFAULT_HORIZONS: tuple[int, ...] = (1, 2, 5, 10, 20)
_MAX_TEST_START_ADVANCE_DAYS = 180


def resolve_leakage_safe_test_start(
    dataset: BaselineDataset,
    *,
    train_end: date,
    configured_test_start: date,
    horizon: int,
    max_advance_days: int = _MAX_TEST_START_ADVANCE_DAYS,
) -> date:
    """Return the earliest ``test_start`` that satisfies embargo and bar-gap guards.

    Keeps the configured ``test_start`` when it already works; otherwise advances
    day-by-day until ``time_embargo_split`` accepts the calendar (hard-fail if no
    safe date exists within ``max_advance_days``).
    """
    if horizon < 1:
        msg = "horizon must be >= 1"
        raise ValueError(msg)
    embargo_days = horizon
    min_by_calendar = train_end + timedelta(days=embargo_days)
    candidate = max(configured_test_start, min_by_calendar)

    last_error: Exception | None = None
    for offset in range(max_advance_days + 1):
        test_start = candidate + timedelta(days=offset)
        try:
            time_embargo_split(
                dataset,
                train_end=train_end,
                test_start=test_start,
                embargo_days=embargo_days,
                horizon=horizon,
            )
        except ValueError as exc:
            last_error = exc
            continue
        return test_start

    detail = f": {last_error}" if last_error is not None else ""
    msg = (
        f"no leakage-safe test_start for horizon={horizon} within "
        f"{max_advance_days} days after {candidate}{detail}"
    )
    raise ValueError(msg)


def settings_for_horizon(
    settings: Settings,
    *,
    horizon: int,
    test_start: date,
) -> Settings:
    """Copy settings with horizon/embargo/test_start and a unique strategy_id."""
    research = settings.research.model_copy(
        update={
            "horizon": horizon,
            "embargo_days": horizon,
            "test_start": test_start,
            "strategy_id": f"{settings.research.strategy_id}-h{horizon}",
        }
    )
    return settings.model_copy(update={"research": research})


def evaluate_horizon(
    bars: BarFrame,
    feature_matrix: pd.DataFrame,
    settings: Settings,
    *,
    horizon: int,
    artifact_adapter: FilesystemModelArtifactAdapter,
    created_at: datetime,
    persist_artifact: bool = True,
) -> HorizonSweepRow:
    """Train/evaluate one horizon on precomputed features (labels recomputed)."""
    if horizon < 1:
        msg = "horizon must be >= 1"
        raise ValueError(msg)

    labels, _meta = make_forward_return_labels(bars, horizon=horizon)
    dataset = assemble_baseline_dataset(
        feature_matrix,
        labels,
        horizon=horizon,
        feature_columns=FEATURE_COLUMNS,
    )
    test_start = resolve_leakage_safe_test_start(
        dataset,
        train_end=settings.research.train_end,
        configured_test_start=settings.research.test_start,
        horizon=horizon,
    )
    horizon_settings = settings_for_horizon(
        settings,
        horizon=horizon,
        test_start=test_start,
    )
    research = horizon_settings.research
    train, test, split_def = time_embargo_split(
        dataset,
        train_end=research.train_end,
        test_start=research.test_start,
        embargo_days=research.embargo_days,
        horizon=research.horizon,
    )

    hyperparams = build_hyperparams(research)
    model_hash = compute_model_hash(
        features=FEATURE_COLUMNS,
        hyperparameters=hyperparams,
        universe=horizon_settings.universe,
        start=horizon_settings.start,
        end=horizon_settings.end,
        frequency=horizon_settings.frequency,
        seed=research.seed,
    )
    # strategy_id distinguishes otherwise-identical hyperparam hashes across horizons.
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

    if persist_artifact:
        metadata = ModelArtifactMetadata(
            model_hash=model_hash,
            strategy_id=research.strategy_id,
            created_at=created_at,
            universe=horizon_settings.universe,
            features=FEATURE_COLUMNS,
            hyperparameters=hyperparams,
            seed=research.seed,
            start=horizon_settings.start,
            end=horizon_settings.end,
            frequency=horizon_settings.frequency,
            model_format="lightgbm_native",
            model_file=MODEL_FILENAME,
            phase=2,
        )
        artifact_dir = artifact_adapter.save_booster(model, metadata)
        sidecar = Phase2Sidecar(
            model_hash=model_hash,
            git_commit=resolve_git_commit(),
            universe=horizon_settings.universe,
            feature_list=FEATURE_COLUMNS,
            hyperparams=hyperparams,
            horizon=research.horizon,
            split_def=split_def,
            data_range={
                "start": horizon_settings.start.isoformat(),
                "end": horizon_settings.end.isoformat(),
                "frequency": horizon_settings.frequency,
            },
            lib_versions=lib_versions(),
            seed=research.seed,
            metrics=metrics,
            strategy_id=research.strategy_id,
        )
        write_sidecar(artifact_dir, sidecar)

    return HorizonSweepRow(
        horizon=horizon,
        ic_mean=metrics.ic_mean,
        rank_ic_mean=metrics.rank_ic_mean,
        icir=metrics.icir,
        ic_materially_positive=metrics.ic_materially_positive,
        n_symbols=len(bars.bars),
        embargo_days=research.embargo_days,
        test_start=research.test_start,
        train_end=research.train_end,
    )


def run_horizon_sweep(
    bars: BarFrame,
    feature_matrix: pd.DataFrame,
    settings: Settings,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    artifact_adapter: FilesystemModelArtifactAdapter,
    created_at: datetime,
    persist_artifact: bool = True,
) -> HorizonSweepReport:
    """Evaluate each horizon once; bars/features must already be loaded."""
    if not horizons:
        msg = "horizons must not be empty"
        raise ValueError(msg)
    ordered = tuple(int(h) for h in horizons)
    if any(h < 1 for h in ordered):
        msg = "all horizons must be >= 1"
        raise ValueError(msg)
    if len(ordered) != len(set(ordered)):
        msg = "horizons must be unique"
        raise ValueError(msg)

    rows = [
        evaluate_horizon(
            bars,
            feature_matrix,
            settings,
            horizon=horizon,
            artifact_adapter=artifact_adapter,
            created_at=created_at,
            persist_artifact=persist_artifact,
        )
        for horizon in ordered
    ]
    return HorizonSweepReport(
        rows=tuple(rows),
        n_symbols=len(bars.bars),
        seed=settings.research.seed,
    )


def format_sweep_table(report: HorizonSweepReport) -> str:
    """Render a fixed-column text table for Slack / stdout."""
    header = (
        f"{'horizon':>8}  {'ic_mean':>10}  {'rank_ic_mean':>12}  "
        f"{'icir':>10}  {'ic_materially_positive':>22}  {'n_symbols':>9}"
    )
    lines = [header, "-" * len(header)]
    for row in report.rows:
        icir_text = "None" if row.icir is None else f"{row.icir:.6f}"
        lines.append(
            f"{row.horizon:>8}  {row.ic_mean:>10.6f}  {row.rank_ic_mean:>12.6f}  "
            f"{icir_text:>10}  {str(row.ic_materially_positive):>22}  {row.n_symbols:>9}"
        )
    return "\n".join(lines)
