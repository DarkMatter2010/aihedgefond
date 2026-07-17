"""Manual live CPCV + DSR gate for the best Phase-2 candidate.

Excluded from pytest/CI (Yahoo network). Reproduces horizon=2 on the
50-name universe with the main (#17) feature set and seed=42, then runs the
Phase-3 overfitting gate.

n_trials / var_trial_sharpes convention
---------------------------------------
``n_trials`` = count of independent research configurations explored before
selecting this candidate (honest selection-bias input; not the CPCV fold
count):

1. Phase-2 baseline h=5 (original 9-feature set)
2. 50-symbol IC validation re-run
3. test_start bar-gap fix re-run
4. Corporate-actions fix re-run (same h=5)
5. Feature-set expansion (#17) at h=5
6–10. Multi-horizon IC sweep h ∈ {1, 2, 5, 10, 20}
11–12. Momentum-breadth probe h ∈ {63, 126}

→ n_trials = 12

``var_trial_sharpes`` comes from ``research.trial_meta.research_var_trial_sharpes``
(variance of non-annualized trial Sharpes derived from those documented ICs
via the Grinold CS proxy SR ≈ IC·√N). It is **never** the CPCV path-Sharpe
variance.

Aggregation: overlapping CPCV OOS path returns are merged to one series
(mean per timestamp); DSR uses T = number of return observations in that
series.
"""

from __future__ import annotations

from datetime import UTC, datetime, time

import pandas as pd

from aihedgefund.core.bus import InProcessMessageBus
from aihedgefund.core.config import load_settings
from aihedgefund.core.runtime import FrozenClock
from aihedgefund.core.schemas import CPCVConfig, MarketDataRequest
from aihedgefund.data.adapters import YFinanceProvider
from aihedgefund.data.quality import DataQualityGate
from aihedgefund.features.pipeline import FEATURE_COLUMNS, FeaturePipeline
from aihedgefund.research.baseline import build_lgbm_params
from aihedgefund.research.dataset import assemble_baseline_dataset
from aihedgefund.research.forward_labels import make_forward_return_labels
from aihedgefund.research.gate import run_overfitting_gate
from aihedgefund.research.trial_meta import (
    RESEARCH_TRIAL_ICS,
    research_trial_sharpes,
    research_var_trial_sharpes,
)

# Best candidate from the multi-horizon sweep (PR #18 live result).
HORIZON = 2
SEED = 42
N_TRIALS = 12
N_BLOCKS = 6
N_TEST_BLOCKS = 2


def main() -> None:
    """Fetch Yahoo once, run CPCV/DSR gate, print the live verdict."""
    if len(RESEARCH_TRIAL_ICS) != N_TRIALS:
        msg = (
            f"RESEARCH_TRIAL_ICS length {len(RESEARCH_TRIAL_ICS)} "
            f"!= N_TRIALS={N_TRIALS}"
        )
        raise RuntimeError(msg)

    var_trial = research_var_trial_sharpes()
    trial_sharpes = research_trial_sharpes()

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
    # Full trading calendar before the final-horizon label drop — required so
    # CPCV resolves t1 on real bars instead of median-gap extrapolation.
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
    verdict = run_overfitting_gate(
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

    print(f"candidate: horizon={HORIZON} seed={SEED} n_symbols={len(bars.bars)}")
    print(f"feature_columns: {len(FEATURE_COLUMNS)}")
    print(f"cpcv: N={N_BLOCKS} k={N_TEST_BLOCKS} n_folds={verdict.cpcv.n_folds}")
    print("aggregation: merge_oos_path_returns_mean_per_timestamp")
    print(f"T_n_obs: {verdict.deflated.n_obs}")
    print(f"path_sharpe_mean: {verdict.path_sharpe_mean}")
    print(f"path_sharpe_std: {verdict.path_sharpe_std}")
    print(f"observed_sharpe: {verdict.deflated.observed_sharpe}")
    print(f"sr0: {verdict.deflated.sr0}")
    print(f"var_trial_sharpes: {verdict.deflated.var_trial_sharpes}")
    print(f"var_trial_source: research_trial_ics_grinold_cs_proxy n={len(trial_sharpes)}")
    print(f"n_trials: {verdict.n_trials}")
    print(f"dsr: {verdict.dsr}")
    print(f"verdict: {verdict.verdict}")
    print(f"execution_stack_release: {'YES' if verdict.verdict == 'JA' else 'NO'}")
    print(
        "note: Slice-2 handoff also requires DSR > permutation-null 95% "
        "(see scripts/run_gate_permutation_null.py)"
    )


if __name__ == "__main__":
    main()
