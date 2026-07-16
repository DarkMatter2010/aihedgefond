"""Research domain; concrete adapters live in ``research.adapters``."""

from aihedgefund.research.adapters import FilesystemModelArtifactAdapter
from aihedgefund.research.baseline import build_lgbm_params, predict_scores, train_baseline
from aihedgefund.research.dataset import assemble_baseline_dataset
from aihedgefund.research.forward_labels import make_forward_return_labels
from aihedgefund.research.metrics import compute_ic_metrics
from aihedgefund.research.model_hash import compute_model_hash
from aihedgefund.research.run_baseline import load_sidecar, run_baseline, write_sidecar
from aihedgefund.research.split import time_embargo_split

__all__ = [
    "FilesystemModelArtifactAdapter",
    "assemble_baseline_dataset",
    "build_lgbm_params",
    "compute_ic_metrics",
    "compute_model_hash",
    "load_sidecar",
    "make_forward_return_labels",
    "predict_scores",
    "run_baseline",
    "time_embargo_split",
    "train_baseline",
    "write_sidecar",
]
