"""Permutation null for the Phase-3 CPCV/DSR gate (manual, Yahoo, not CI).

Destroys cross-sectional signal by shuffling forward-return labels within each
date while keeping the panel structure, then re-runs the full overfitting gate.
Reports the null DSR distribution and where the real candidate sits.

Acceptance (live handoff)
-------------------------
JA only if real DSR > 95th percentile of the permutation null.
If the null itself clusters near ~0.9+, the gate is still broken — report loudly.

Setup matches the live candidate: h=2, 50 names, feature set #17, seed=42.
"""

from __future__ import annotations

from datetime import UTC, datetime, time

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
from aihedgefund.research.research_trials import (
    N_RESEARCH_TRIALS,
    research_trial_sharpe_variance,
)

HORIZON = 2
SEED = 42
N_TRIALS = N_RESEARCH_TRIALS
N_BLOCKS = 6
N_TEST_BLOCKS = 2
N_PERMUTATIONS = 100
PERM_SEED = 20260717


def _permute_labels_within_dates(label: pd.Series, *, rng: np.random.Generator) -> pd.Series:
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


def _run_gate(
    dataset: BaselineDataset,
    *,
    cpcv_config: CPCVConfig,
    params: dict[str, object],
    num_boost_round: int,
    var_trial: float,
    seed: int,
    universe: tuple[str, ...],
    start: object,
    end: object,
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
        seed=seed,
        universe=universe,
        start=start,  # type: ignore[arg-type]
        end=end,  # type: ignore[arg-type]
        frequency=frequency,
        bar_timestamps=bar_timestamps,
    )
    return float(verdict.dsr)


def main() -> None:
    """Yahoo once → real gate DSR → M label permutations → null report."""
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
    var_trial = research_trial_sharpe_variance()
    gate_kwargs = dict(
        cpcv_config=cpcv_config,
        params=params,
        num_boost_round=research.num_boost_round,
        var_trial=var_trial,
        seed=SEED,
        universe=tuple(settings.universe),
        start=settings.start,
        end=settings.end,
        frequency=settings.frequency,
        bar_timestamps=bar_timestamps,
    )

    print(
        f"real candidate: h={HORIZON} seed={SEED} n_symbols={len(bars.bars)} "
        f"features={len(FEATURE_COLUMNS)} n_perm={N_PERMUTATIONS}"
    )
    print(f"var_trial_sharpes: {var_trial} (RESEARCH_TRIAL_SHARPES)")
    real_verdict = run_overfitting_gate(
        dataset,
        cpcv_config=cpcv_config,
        model_params=params,
        num_boost_round=research.num_boost_round,
        n_trials=N_TRIALS,
        var_trial_sharpes=var_trial,
        seed=SEED,
        universe=settings.universe,
        start=settings.start,
        end=settings.end,
        frequency=settings.frequency,
        bar_timestamps=bar_timestamps,
    )
    real_dsr = float(real_verdict.dsr)
    print(f"real_dsr: {real_dsr}")
    print(f"real_observed_sharpe: {real_verdict.deflated.observed_sharpe}")
    print(f"real_sr0: {real_verdict.deflated.sr0}")
    print(f"real_n_obs_T: {real_verdict.deflated.n_obs}")
    print(f"real_gate_verdict_dsr_gt_0: {real_verdict.verdict}")

    rng = np.random.default_rng(PERM_SEED)
    null_dsrs: list[float] = []
    for i in range(N_PERMUTATIONS):
        perm_label = _permute_labels_within_dates(dataset.label, rng=rng)
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
    # Empirical percentile of the real DSR within the null (higher = stronger).
    percentile = float(100.0 * np.mean(null_arr < real_dsr))
    beats_null = real_dsr > q95
    null_broken = q95 >= 0.9
    gate_ja = real_dsr >= 0.95

    print("--- permutation null ---")
    print(f"null_dsr_q50: {q50}")
    print(f"null_dsr_q95: {q95}")
    print(f"null_dsr_min: {float(null_arr.min())}")
    print(f"null_dsr_max: {float(null_arr.max())}")
    print(f"real_dsr: {real_dsr}")
    print(f"real_percentile_vs_null: {percentile:.2f}")
    print(f"beats_null_q95: {beats_null}")
    print(f"gate_dsr_ge_0_95: {gate_ja}")
    if null_broken:
        print(
            "GATE_BROKEN: null DSR 95% quantile is already >= 0.9 — "
            "selection-bias correction still inflated"
        )
    # JA only if absolute Bailey confidence clears 0.95 AND beats the null.
    corrected_verdict = (
        "JA" if gate_ja and beats_null and not null_broken else "NEIN"
    )
    print(f"corrected_verdict: {corrected_verdict}")
    print(
        "handoff_slice2_execution: "
        f"{'YES' if corrected_verdict == 'JA' else 'NO (blocked)'}"
    )


if __name__ == "__main__":
    main()
