"""Permutation null for the CPCV/DSR gate (manual live path; not CI).

Loads Yahoo once (horizon=2, 50 names, feature-set #17, seed=42), runs the
real candidate through the corrected gate, then destroys the signal with
M≥100 cross-sectional label permutations (shuffle forward returns within
each date) and re-runs the full gate for each permutation.

Reports null DSR quantiles (50%, 95%), the real candidate's DSR + percentile
against the null, and the Slice-2 acceptance decision:

    JA only if real DSR > null 95% quantile.
    If the null itself sits near ~0.9+, the gate is still broken — report loudly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, time
from typing import Any

import numpy as np
import pandas as pd

from aihedgefund.core.bus import InProcessMessageBus
from aihedgefund.core.config import load_settings
from aihedgefund.core.runtime import FrozenClock
from aihedgefund.core.schemas import BaselineDataset, CPCVConfig, MarketDataRequest
from aihedgefund.data.adapters import YFinanceProvider
from aihedgefund.data.quality import DataQualityGate
from aihedgefund.features.pipeline import FEATURE_COLUMNS, FeaturePipeline
from aihedgefund.research.baseline import build_lgbm_params
from aihedgefund.research.dataset import assemble_baseline_dataset
from aihedgefund.research.forward_labels import make_forward_return_labels
from aihedgefund.research.gate import run_overfitting_gate
from aihedgefund.research.trial_meta import RESEARCH_TRIAL_ICS, research_var_trial_sharpes

HORIZON = 2
SEED = 42
N_TRIALS = 12
N_BLOCKS = 6
N_TEST_BLOCKS = 2
N_PERMUTATIONS = 100
PERM_SEED = 20260717


def _permute_labels_cross_section(
    label: pd.Series,
    *,
    rng: np.random.Generator,
) -> pd.Series:
    """Shuffle forward-return labels within each timestamp (destroy signal)."""
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


def _run_gate(
    dataset: BaselineDataset,
    *,
    cpcv_config: CPCVConfig,
    params: Mapping[str, Any],
    num_boost_round: int,
    var_trial: float,
    settings_universe: Sequence[str],
    start: date,
    end: date,
    frequency: str,
    bar_timestamps: pd.DatetimeIndex,
) -> float:
    verdict = run_overfitting_gate(
        dataset,
        cpcv_config=cpcv_config,
        model_params=params,
        num_boost_round=num_boost_round,
        n_trials=N_TRIALS,
        var_trial_sharpes=var_trial,
        seed=SEED,
        universe=settings_universe,
        start=start,
        end=end,
        frequency=frequency,
        bar_timestamps=bar_timestamps,
    )
    return float(verdict.dsr)


def main() -> None:
    """Yahoo once → real DSR → M label permutations → null report."""
    if len(RESEARCH_TRIAL_ICS) != N_TRIALS:
        msg = (
            f"RESEARCH_TRIAL_ICS length {len(RESEARCH_TRIAL_ICS)} "
            f"!= N_TRIALS={N_TRIALS}"
        )
        raise RuntimeError(msg)
    if N_PERMUTATIONS < 100:
        msg = f"N_PERMUTATIONS must be >= 100, got {N_PERMUTATIONS}"
        raise RuntimeError(msg)

    var_trial = research_var_trial_sharpes()
    settings = load_settings()
    research = settings.research.model_copy(
        update={
            "horizon": HORIZON,
            "embargo_days": HORIZON,
            "seed": SEED,
        }
    )
    settings = settings.model_copy(update={"research": research})

    request_start = datetime.combine(settings.start, time.min, tzinfo=UTC)
    request_end = datetime.combine(settings.end, time.min, tzinfo=UTC)
    clock = FrozenClock(request_end)
    bus = InProcessMessageBus()
    provider = YFinanceProvider(
        settings.symbol_aliases,
        bus,
        DataQualityGate(settings.quality, bus, clock=clock),
        clock=clock,
    )
    bars = provider.get_ohlcv(
        MarketDataRequest(
            symbols=settings.universe,
            start=request_start,
            end=request_end,
            frequency=settings.frequency,
        )
    )

    feature_matrix = FeaturePipeline(bus, clock=clock).compute(bars)
    labels, _meta = make_forward_return_labels(bars, horizon=HORIZON)
    dataset = assemble_baseline_dataset(
        feature_matrix,
        labels,
        horizon=HORIZON,
        feature_columns=FEATURE_COLUMNS,
    )
    bar_timestamps = pd.DatetimeIndex(
        sorted({ts for frame in bars.bars.values() for ts in frame.index})
    )

    params = build_lgbm_params(
        seed=SEED,
        learning_rate=research.learning_rate,
        num_leaves=research.num_leaves,
        min_data_in_leaf=research.min_data_in_leaf,
        feature_fraction=research.feature_fraction,
        bagging_fraction=research.bagging_fraction,
        bagging_freq=research.bagging_freq,
    )
    cpcv_config = CPCVConfig(
        n_blocks=N_BLOCKS,
        n_test_blocks=N_TEST_BLOCKS,
        embargo_days=HORIZON,
        horizon=HORIZON,
    )
    gate_kwargs = dict(
        cpcv_config=cpcv_config,
        params=params,
        num_boost_round=research.num_boost_round,
        var_trial=var_trial,
        settings_universe=tuple(settings.universe),
        start=settings.start,
        end=settings.end,
        frequency=settings.frequency,
        bar_timestamps=bar_timestamps,
    )

    print(
        f"loading done: n_symbols={len(bars.bars)} "
        f"rows={len(dataset.features)} var_trial_sharpes={var_trial:.6g}"
    )
    print("running real candidate gate…")
    real_dsr = _run_gate(dataset, **gate_kwargs)
    print(f"real_dsr: {real_dsr}")

    rng = np.random.default_rng(PERM_SEED)
    null_dsrs: list[float] = []
    for i in range(N_PERMUTATIONS):
        perm_label = _permute_labels_cross_section(dataset.label, rng=rng)
        perm_dataset = BaselineDataset(
            features=dataset.features,
            label=perm_label,
            horizon=dataset.horizon,
            feature_columns=dataset.feature_columns,
        )
        dsr_i = _run_gate(perm_dataset, **gate_kwargs)
        null_dsrs.append(dsr_i)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"perm {i + 1}/{N_PERMUTATIONS}: dsr={dsr_i:.6f}")

    null_arr = np.asarray(null_dsrs, dtype=np.float64)
    q50 = float(np.quantile(null_arr, 0.50))
    q95 = float(np.quantile(null_arr, 0.95))
    # Empirical percentile of real DSR within the null (fraction of null ≤ real).
    percentile = float(np.mean(null_arr <= real_dsr) * 100.0)
    beats_null = real_dsr > q95
    null_broken = q95 >= 0.90

    print("--- permutation null report ---")
    print(f"horizon: {HORIZON}")
    print(f"n_permutations: {N_PERMUTATIONS}")
    print(f"perm_seed: {PERM_SEED}")
    print("aggregation: merge_oos_path_returns_mean_per_timestamp")
    print(f"var_trial_sharpes: {var_trial}")
    print("var_trial_source: research_trial_ics_grinold_cs_proxy")
    print(f"real_dsr: {real_dsr}")
    print(f"null_dsr_q50: {q50}")
    print(f"null_dsr_q95: {q95}")
    print(f"real_percentile_vs_null: {percentile}")
    print(f"beats_null_95: {beats_null}")
    if null_broken:
        print(
            "GATE_BROKEN: null DSR 95% quantile is >= 0.90 — "
            "gate still passes noise; do NOT release Slice 2"
        )
    # Slice-2 handoff: must clear Bailey confidence (dsr >= 0.95) AND beat the
    # permutation null. Beating a near-zero null with dsr≪0.95 is not a pass.
    gate_confident = real_dsr >= 0.95
    print(f"gate_dsr_ge_0.95: {gate_confident}")
    verdict = "JA" if (beats_null and gate_confident and not null_broken) else "NEIN"
    print(f"corrected_verdict: {verdict}")
    print(f"slice2_handoff: {'RELEASE' if verdict == 'JA' else 'BLOCKED'}")
    if beats_null and not gate_confident:
        print(
            "handoff_note: real DSR beats null 95% numerically but absolute "
            f"DSR={real_dsr:.3g} << 0.95 — Bailey gate NEIN; Slice 2 blocked"
        )


if __name__ == "__main__":
    main()
