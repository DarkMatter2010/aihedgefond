"""Hardened CPCV/DSR gate wiring for the ``all_new`` feature-class candidate.

Validates the already-logged triage trial (``ALL_NEW_FEATURE_CLASS_COLUMNS`` on
the broad universe). Does **not** increment ``N_RESEARCH_TRIALS``.

Primary horizon is h=21 (best triage Rank-IC); h=2 is secondary context only.
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
    GateVerdict,
)
from aihedgefund.features.feature_classes import (
    ALL_NEW_FEATURE_CLASS_COLUMNS,
    build_triage_feature_matrix,
)
from aihedgefund.features.pipeline import FEATURE_COLUMNS
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
PERM_SEED: Final[int] = 20260717
DSR_THRESHOLD: Final[float] = 0.95

GATE_CANDIDATE_FEATURE_COLUMNS: Final[tuple[str, ...]] = ALL_NEW_FEATURE_CLASS_COLUMNS

COUNTS_AS_NEW_RESEARCH_TRIAL: Final[bool] = False

CPCV_PARAM_NOTE: Final[str] = (
    "CPCV N=6 k=2 unchanged from prior hardened gate; no new hyper-variation"
)


def assert_all_new_feature_set() -> None:
    """Hard-fail if the gate candidate is not exactly the 15-col all_new set."""
    if GATE_CANDIDATE_FEATURE_COLUMNS != ALL_NEW_FEATURE_CLASS_COLUMNS:
        msg = "gate candidate must equal ALL_NEW_FEATURE_CLASS_COLUMNS"
        raise RuntimeError(msg)
    if len(GATE_CANDIDATE_FEATURE_COLUMNS) != 15:
        msg = f"expected 15 all_new columns, got {len(GATE_CANDIDATE_FEATURE_COLUMNS)}"
        raise RuntimeError(msg)
    overlap = set(GATE_CANDIDATE_FEATURE_COLUMNS) & set(FEATURE_COLUMNS)
    if overlap:
        msg = f"all_new must not mix production FEATURE_COLUMNS; overlap={sorted(overlap)}"
        raise RuntimeError(msg)


def build_all_new_feature_matrix(bars: BarFrame) -> pd.DataFrame:
    """Triage matrix sliced to the 15 all_new columns (exact schema for assemble)."""
    assert_all_new_feature_set()
    full = build_triage_feature_matrix(bars)
    return full.loc[:, list(GATE_CANDIDATE_FEATURE_COLUMNS)].astype(
        {column: "float64" for column in GATE_CANDIDATE_FEATURE_COLUMNS}
    )


def assemble_all_new_dataset(bars: BarFrame, *, horizon: int) -> BaselineDataset:
    """Causal all_new features + forward labels for one horizon."""
    if horizon < 1:
        msg = "horizon must be >= 1"
        raise ValueError(msg)
    feature_matrix = build_all_new_feature_matrix(bars)
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


def permute_labels_within_dates(
    label: pd.Series,
    *,
    rng: np.random.Generator,
) -> pd.Series:
    """Shuffle forward returns inside each timestamp's cross-section."""
    if list(label.index.names) != ["timestamp", "symbol"]:
        msg = "label index names must be ('timestamp', 'symbol')"
        raise ValueError(msg)
    parts: list[pd.Series] = []
    for _ts, group in label.groupby(level="timestamp", sort=True):
        values = group.to_numpy(dtype=np.float64).copy()
        rng.shuffle(values)
        parts.append(pd.Series(values, index=group.index, dtype="float64"))
    out = pd.concat(parts).sort_index()
    out.name = label.name
    return out


def run_all_new_gate(
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
    """Run ``run_overfitting_gate`` with research-trial variance (n_trials=23)."""
    if dataset.feature_columns != GATE_CANDIDATE_FEATURE_COLUMNS:
        msg = "dataset.feature_columns must be ALL_NEW_FEATURE_CLASS_COLUMNS"
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


def interpret_corrected_verdict(
    *,
    real_dsr: float,
    null_q95: float,
) -> tuple[str, bool, bool, bool]:
    """Return (corrected_verdict, gate_ja, beats_null, null_broken)."""
    gate_ja = real_dsr >= DSR_THRESHOLD
    beats_null = real_dsr > null_q95
    null_broken = null_q95 >= 0.9
    corrected = "JA" if gate_ja and beats_null and not null_broken else "NEIN"
    return corrected, gate_ja, beats_null, null_broken
