"""Phase-2 end-to-end baseline orchestration and artifact sidecar I/O."""

from __future__ import annotations

import subprocess
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from aihedgefund.core.config import ResearchSettings, Settings
from aihedgefund.core.schemas import (
    BarFrame,
    BaselineDataset,
    ICMetricsReport,
    ModelArtifactMetadata,
    Phase2Sidecar,
    PredictionOutput,
    SplitDefinition,
)
from aihedgefund.features.pipeline import FEATURE_COLUMNS, FeaturePipeline
from aihedgefund.research.adapters.filesystem import (
    METADATA_FILENAME,
    MODEL_FILENAME,
    FilesystemModelArtifactAdapter,
)
from aihedgefund.research.baseline import (
    build_lgbm_params,
    predict_scores,
    train_baseline,
)
from aihedgefund.research.dataset import assemble_baseline_dataset
from aihedgefund.research.forward_labels import make_forward_return_labels
from aihedgefund.research.metrics import compute_ic_metrics
from aihedgefund.research.model_hash import compute_model_hash
from aihedgefund.research.split import time_embargo_split

SIDECAR_FILENAME = "phase2_sidecar.json"


def resolve_git_commit(*, fallback: str = "unknown") -> str:
    """Return the current HEAD SHA, or ``fallback`` when git is unavailable."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return fallback
    commit = completed.stdout.strip()
    return commit or fallback


def lib_versions() -> dict[str, str]:
    """Pin-relevant library versions recorded in the Phase-2 sidecar."""
    names = ("lightgbm", "pandas", "numpy", "scipy", "pydantic")
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            msg = f"required package {name!r} is not installed"
            raise RuntimeError(msg) from exc
    return versions


def build_hyperparams(research: ResearchSettings) -> dict[str, Any]:
    """Flatten research LightGBM settings into the artifact hyperparam map."""
    params = build_lgbm_params(
        seed=research.seed,
        learning_rate=research.learning_rate,
        num_leaves=research.num_leaves,
        min_data_in_leaf=research.min_data_in_leaf,
        feature_fraction=research.feature_fraction,
        bagging_fraction=research.bagging_fraction,
        bagging_freq=research.bagging_freq,
    )
    params["num_boost_round"] = research.num_boost_round
    return params


def run_baseline(
    bars: BarFrame,
    settings: Settings,
    *,
    feature_pipeline: FeaturePipeline,
    artifact_adapter: FilesystemModelArtifactAdapter,
    created_at: datetime,
    git_commit: str | None = None,
) -> tuple[
    BaselineDataset,
    BaselineDataset,
    SplitDefinition,
    PredictionOutput,
    ICMetricsReport,
    Path,
    Phase2Sidecar,
]:
    """Train, evaluate, and persist the Phase-2 LightGBM baseline.

    Returns train/test datasets, split definition, OOS predictions, metrics,
    artifact directory, and the Phase-2 JSON sidecar payload.
    """
    research = settings.research
    feature_matrix = feature_pipeline.compute(bars)
    labels, _label_meta = make_forward_return_labels(bars, horizon=research.horizon)
    dataset = assemble_baseline_dataset(
        feature_matrix,
        labels,
        horizon=research.horizon,
        feature_columns=FEATURE_COLUMNS,
    )
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
        universe=settings.universe,
        start=settings.start,
        end=settings.end,
        frequency=settings.frequency,
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

    metadata = ModelArtifactMetadata(
        model_hash=model_hash,
        strategy_id=research.strategy_id,
        created_at=created_at,
        universe=settings.universe,
        features=FEATURE_COLUMNS,
        hyperparameters=hyperparams,
        seed=research.seed,
        start=settings.start,
        end=settings.end,
        frequency=settings.frequency,
        model_format="lightgbm_native",
        model_file=MODEL_FILENAME,
        phase=2,
    )
    artifact_dir = artifact_adapter.save_booster(model, metadata)

    sidecar = Phase2Sidecar(
        model_hash=model_hash,
        git_commit=git_commit if git_commit is not None else resolve_git_commit(),
        universe=settings.universe,
        feature_list=FEATURE_COLUMNS,
        hyperparams=hyperparams,
        horizon=research.horizon,
        split_def=split_def,
        data_range={
            "start": settings.start.isoformat(),
            "end": settings.end.isoformat(),
            "frequency": settings.frequency,
        },
        lib_versions=lib_versions(),
        seed=research.seed,
        metrics=metrics,
        strategy_id=research.strategy_id,
    )
    write_sidecar(artifact_dir, sidecar)
    return train, test, split_def, predictions, metrics, artifact_dir, sidecar


def write_sidecar(artifact_dir: Path, sidecar: Phase2Sidecar) -> Path:
    """Persist ``phase2_sidecar.json`` next to ``model.txt`` / ``metadata.json``."""
    path = artifact_dir / SIDECAR_FILENAME
    if path.exists():
        msg = f"phase-2 sidecar already exists: {path}"
        raise FileExistsError(msg)
    path.write_text(sidecar.model_dump_json(), encoding="utf-8")
    return path


def load_sidecar(artifact_dir: Path) -> Phase2Sidecar:
    """Load and validate the Phase-2 sidecar from an artifact directory."""
    path = artifact_dir / SIDECAR_FILENAME
    if not path.is_file():
        msg = f"phase-2 sidecar not found: {path}"
        raise FileNotFoundError(msg)
    model_path = artifact_dir / MODEL_FILENAME
    metadata_path = artifact_dir / METADATA_FILENAME
    if not model_path.is_file() or not metadata_path.is_file():
        msg = f"incomplete model artifact at {artifact_dir}"
        raise FileNotFoundError(msg)
    return Phase2Sidecar.model_validate_json(path.read_text(encoding="utf-8"))
