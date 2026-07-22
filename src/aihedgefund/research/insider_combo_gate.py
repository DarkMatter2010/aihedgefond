"""Hardened CPCV/DSR gate for `insider_plus_all_new` (last free-data candidate).

Validates the already-logged triage trial on `BROAD_LIQUID_CANDIDATE_UNIVERSE`.
Does **not** increment `N_RESEARCH_TRIALS` (already counted as trials 28–29).

Primary horizon is h=21 (best triage Rank-IC 0.0403); h=2 is secondary context.
CPCV knobs match the prior hardened gate (N=6, k=2) — no new hyper-variation.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any, Final

import numpy as np
import pandas as pd

from aihedgefund.core.schemas import (
    BarFrame,
    BaselineDataset,
    CPCVConfig,
    Form4Frame,
    GateVerdict,
)
from aihedgefund.features.feature_classes import ALL_NEW_FEATURE_CLASS_COLUMNS
from aihedgefund.features.insider import (
    INSIDER_FEATURE_COLUMNS,
    INSIDER_PLUS_ALL_NEW_FEATURE_COLUMNS,
    build_insider_feature_matrix,
)
from aihedgefund.features.feature_classes import build_triage_feature_matrix
from aihedgefund.features.pipeline import FEATURE_COLUMNS
from aihedgefund.research.all_new_gate import (
    interpret_corrected_verdict,
    permute_labels_within_dates,
)
from aihedgefund.research.dataset import assemble_baseline_dataset
from aihedgefund.research.forward_labels import make_forward_return_labels
from aihedgefund.research.gate import run_overfitting_gate
from aihedgefund.research.research_trials import (
    N_RESEARCH_TRIALS,
    research_trial_sharpe_variance,
)

PRIMARY_HORIZON: Final[int] = 21
SECONDARY_HORIZON: Final[int] = 2
SEED: Final[int] = 42
N_TRIALS: Final[int] = N_RESEARCH_TRIALS
N_BLOCKS: Final[int] = 6
N_TEST_BLOCKS: Final[int] = 2
N_PERMUTATIONS: Final[int] = 100
PERM_SEED: Final[int] = 20260722
DSR_THRESHOLD: Final[float] = 0.95

GATE_CANDIDATE_FEATURE_COLUMNS: Final[tuple[str, ...]] = INSIDER_PLUS_ALL_NEW_FEATURE_COLUMNS

COUNTS_AS_NEW_RESEARCH_TRIAL: Final[bool] = False

CPCV_PARAM_NOTE: Final[str] = (
    "CPCV N=6 k=2 unchanged from prior hardened gate; no new hyper-variation"
)

UNIVERSE_NOTE: Final[str] = (
    "BROAD_LIQUID_CANDIDATE_UNIVERSE — same universe that produced triage Rank-IC 0.0403"
)


def assert_insider_combo_feature_set() -> None:
    """Hard-fail if the gate candidate is not exactly insider_plus_all_new."""
    if GATE_CANDIDATE_FEATURE_COLUMNS != INSIDER_PLUS_ALL_NEW_FEATURE_COLUMNS:
        msg = "gate candidate must equal INSIDER_PLUS_ALL_NEW_FEATURE_COLUMNS"
        raise RuntimeError(msg)
    if len(GATE_CANDIDATE_FEATURE_COLUMNS) != 24:
        msg = (
            f"expected 24 insider_plus_all_new columns, "
            f"got {len(GATE_CANDIDATE_FEATURE_COLUMNS)}"
        )
        raise RuntimeError(msg)
    if set(INSIDER_FEATURE_COLUMNS) & set(FEATURE_COLUMNS):
        msg = "insider columns must not overlap production FEATURE_COLUMNS"
        raise RuntimeError(msg)
    expected = (*INSIDER_FEATURE_COLUMNS, *ALL_NEW_FEATURE_CLASS_COLUMNS)
    if GATE_CANDIDATE_FEATURE_COLUMNS != expected:
        msg = "insider_plus_all_new schema drift vs insider + ALL_NEW"
        raise RuntimeError(msg)


def build_insider_combo_feature_matrix(
    bars: BarFrame,
    form4: Form4Frame,
) -> pd.DataFrame:
    """Insider (filed_at PIT) joined with ALL_NEW — exact 24-col gate schema."""
    assert_insider_combo_feature_set()
    insider = build_insider_feature_matrix(bars, form4)
    triage = build_triage_feature_matrix(bars)
    all_new = triage.loc[:, list(ALL_NEW_FEATURE_CLASS_COLUMNS)]
    combined = insider.join(all_new, how="inner")
    missing = [c for c in GATE_CANDIDATE_FEATURE_COLUMNS if c not in combined.columns]
    if missing:
        msg = f"insider combo matrix missing columns: {missing}"
        raise ValueError(msg)
    return combined.loc[:, list(GATE_CANDIDATE_FEATURE_COLUMNS)].astype(
        {column: "float64" for column in GATE_CANDIDATE_FEATURE_COLUMNS}
    )


def assemble_insider_combo_dataset(
    bars: BarFrame,
    form4: Form4Frame,
    *,
    horizon: int,
) -> BaselineDataset:
    """Causal insider_plus_all_new features + forward labels for one horizon."""
    if horizon < 1:
        msg = "horizon must be >= 1"
        raise ValueError(msg)
    feature_matrix = build_insider_combo_feature_matrix(bars, form4)
    labels, _meta = make_forward_return_labels(bars, horizon=horizon)
    return assemble_baseline_dataset(
        feature_matrix,
        labels,
        horizon=horizon,
        feature_columns=GATE_CANDIDATE_FEATURE_COLUMNS,
    )


def bar_timestamps_from_bars(bars: BarFrame) -> pd.DatetimeIndex:
    """Union trading calendar for CPCV label-end resolution."""
    return pd.DatetimeIndex(sorted({ts for frame in bars.bars.values() for ts in frame.index}))


def cpcv_config_for_horizon(horizon: int) -> CPCVConfig:
    """Fixed hardened CPCV knobs with matched embargo."""
    return CPCVConfig(
        n_blocks=N_BLOCKS,
        n_test_blocks=N_TEST_BLOCKS,
        embargo_days=horizon,
        horizon=horizon,
    )


def run_insider_combo_gate(
    dataset: BaselineDataset,
    *,
    model_params: Mapping[str, Any],
    num_boost_round: int,
    seed: int,
    universe: tuple[str, ...],
    start: date,
    end: date,
    frequency: str,
    bar_timestamps: pd.DatetimeIndex,
) -> GateVerdict:
    """Run `run_overfitting_gate` with research-trial variance (n_trials=29)."""
    if dataset.feature_columns != GATE_CANDIDATE_FEATURE_COLUMNS:
        msg = "dataset.feature_columns must be INSIDER_PLUS_ALL_NEW_FEATURE_COLUMNS"
        raise ValueError(msg)
    horizon = dataset.horizon
    return run_overfitting_gate(
        dataset,
        cpcv_config=cpcv_config_for_horizon(horizon),
        model_params=model_params,
        num_boost_round=num_boost_round,
        n_trials=N_TRIALS,
        var_trial_sharpes=research_trial_sharpe_variance(),
        seed=seed,
        universe=universe,
        start=start,
        end=end,
        frequency=frequency,
        bar_timestamps=bar_timestamps,
    )


__all__ = [
    "COUNTS_AS_NEW_RESEARCH_TRIAL",
    "CPCV_PARAM_NOTE",
    "DSR_THRESHOLD",
    "GATE_CANDIDATE_FEATURE_COLUMNS",
    "N_BLOCKS",
    "N_PERMUTATIONS",
    "N_TEST_BLOCKS",
    "N_TRIALS",
    "PERM_SEED",
    "PRIMARY_HORIZON",
    "SECONDARY_HORIZON",
    "SEED",
    "UNIVERSE_NOTE",
    "assemble_insider_combo_dataset",
    "assert_insider_combo_feature_set",
    "bar_timestamps_from_bars",
    "build_insider_combo_feature_matrix",
    "cpcv_config_for_horizon",
    "interpret_corrected_verdict",
    "permute_labels_within_dates",
    "run_insider_combo_gate",
]
