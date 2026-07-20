"""Research domain; concrete adapters live in ``research.adapters``."""

from aihedgefund.research.adapters import FilesystemModelArtifactAdapter
from aihedgefund.research.all_new_gate import (
    assemble_all_new_dataset,
    run_all_new_gate,
)
from aihedgefund.research.baseline import build_lgbm_params, predict_scores, train_baseline
from aihedgefund.research.cpcv import combinatorial_purged_splits, subset_by_positions
from aihedgefund.research.dataset import assemble_baseline_dataset
from aihedgefund.research.deflated_sharpe import (
    deflated_sharpe,
    expected_max_sharpe,
    sharpe_ratio,
)
from aihedgefund.research.feature_class_triage import (
    run_feature_class_triage,
    settings_for_feature_class_triage,
)
from aihedgefund.research.forward_labels import make_forward_return_labels
from aihedgefund.research.gate import (
    merge_cpcv_path_returns,
    run_overfitting_gate,
    scores_to_strategy_returns,
)
from aihedgefund.research.meta_labeling import run_meta_labeling_triage
from aihedgefund.research.meta_labeling_gate import run_meta_labeling_gate
from aihedgefund.research.metrics import compute_ic_metrics
from aihedgefund.research.model_hash import compute_model_hash
from aihedgefund.research.research_trials import (
    N_RESEARCH_TRIALS,
    RESEARCH_TRIAL_SHARPES,
    research_trial_sharpe_variance,
)
from aihedgefund.research.run_baseline import load_sidecar, run_baseline, write_sidecar
from aihedgefund.research.split import time_embargo_split
from aihedgefund.research.universe_breadth_diagnostic import (
    run_universe_breadth_diagnostic,
    settings_for_breadth_diagnostic,
)
from aihedgefund.research.universes import (
    BROAD_LIQUID_CANDIDATE_UNIVERSE,
    SURVIVORSHIP_BIAS_NOTE,
)

__all__ = [
    "BROAD_LIQUID_CANDIDATE_UNIVERSE",
    "FilesystemModelArtifactAdapter",
    "N_RESEARCH_TRIALS",
    "RESEARCH_TRIAL_SHARPES",
    "SURVIVORSHIP_BIAS_NOTE",
    "assemble_all_new_dataset",
    "assemble_baseline_dataset",
    "build_lgbm_params",
    "combinatorial_purged_splits",
    "compute_ic_metrics",
    "compute_model_hash",
    "deflated_sharpe",
    "expected_max_sharpe",
    "load_sidecar",
    "make_forward_return_labels",
    "merge_cpcv_path_returns",
    "predict_scores",
    "research_trial_sharpe_variance",
    "run_all_new_gate",
    "run_baseline",
    "run_feature_class_triage",
    "run_meta_labeling_gate",
    "run_meta_labeling_triage",
    "run_overfitting_gate",
    "run_universe_breadth_diagnostic",
    "scores_to_strategy_returns",
    "settings_for_breadth_diagnostic",
    "settings_for_feature_class_triage",
    "sharpe_ratio",
    "subset_by_positions",
    "time_embargo_split",
    "train_baseline",
    "write_sidecar",
]
