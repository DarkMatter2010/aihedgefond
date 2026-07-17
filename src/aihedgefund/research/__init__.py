"""Research domain; concrete adapters live in ``research.adapters``."""

from aihedgefund.research.adapters import FilesystemModelArtifactAdapter
from aihedgefund.research.baseline import build_lgbm_params, predict_scores, train_baseline
from aihedgefund.research.cpcv import combinatorial_purged_splits, subset_by_positions
from aihedgefund.research.dataset import assemble_baseline_dataset
from aihedgefund.research.deflated_sharpe import (
    deflated_sharpe,
    expected_max_sharpe,
    sharpe_ratio,
)
from aihedgefund.research.forward_labels import make_forward_return_labels
from aihedgefund.research.gate import (
    aggregate_cpcv_path_returns,
    run_overfitting_gate,
    scores_to_strategy_returns,
)
from aihedgefund.research.metrics import compute_ic_metrics
from aihedgefund.research.model_hash import compute_model_hash
from aihedgefund.research.run_baseline import load_sidecar, run_baseline, write_sidecar
from aihedgefund.research.split import time_embargo_split
from aihedgefund.research.trial_meta import research_var_trial_sharpes

__all__ = [
    "FilesystemModelArtifactAdapter",
    "aggregate_cpcv_path_returns",
    "assemble_baseline_dataset",
    "build_lgbm_params",
    "combinatorial_purged_splits",
    "compute_ic_metrics",
    "compute_model_hash",
    "deflated_sharpe",
    "expected_max_sharpe",
    "load_sidecar",
    "make_forward_return_labels",
    "predict_scores",
    "research_var_trial_sharpes",
    "run_baseline",
    "run_overfitting_gate",
    "scores_to_strategy_returns",
    "sharpe_ratio",
    "subset_by_positions",
    "time_embargo_split",
    "train_baseline",
    "write_sidecar",
]
