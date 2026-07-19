"""Documented research-trial Sharpes for DSR selection-bias correction.

``var_trial_sharpes`` for the Phase-3 gate MUST come from the dispersion of
independent research configurations actually tested — never from CPCV path
Sharpes (those share overlapping train/test windows and are not i.i.d. trials).

Source of ``RESEARCH_TRIAL_SHARPES`` (variance table)
----------------------------------------------------
Twelve non-annualized daily IR proxies on the same scale as the gate's
observed Sharpe (CS long/short portfolio return mean/std). Values are
Grinold-style ``IC * sqrt(breadth)`` with breadth=50 (Phase-2 universe),
using the live IC / rank-IC figures recorded for each configuration:

1. Phase-2 baseline h=5 (pre-CA diagnostics)     IC≈0.0106 → 0.075
2. 50-symbol IC validation                       IC≈0.010  → 0.071
3. test_start bar-gap fix                        IC≈0.010  → 0.071
4. Corporate-actions fix h=5                     IC≈0.0075 → 0.053
5. Feature-set expansion (#17) h=5               IC≈0.0049 → 0.035
6. Multi-horizon sweep h=1                       IC≈0.010  → 0.071
7. Multi-horizon sweep h=2                       IC≈0.0146 → 0.103
8. Multi-horizon sweep h=5                       IC≈0.0049 → 0.035
9. Multi-horizon sweep h=10                      IC≈−0.005 → −0.035
10. Multi-horizon sweep h=20                     IC≈−0.010 → −0.071
11. Momentum-breadth probe h=63                  rank_IC≈−0.009 → −0.064
12. Momentum-breadth probe h=126                 rank_IC≈−0.042 → −0.297

These are selection-bias inputs, not CPCV path diagnostics.

``N_RESEARCH_TRIALS`` (conservative count for DSR)
-------------------------------------------------
Bailey DSR uses ``n_trials`` for the expected-max-SR null under selection bias.
Counting only the 12 *logged* IC rows above understates the true multiple-testing
burden (intermediate feature / horizon / universe probes that were tried but not
written into the IC table).

Documented configuration families (≥10 distinct trials):
  Baseline, CA-Fix, Feature-Set #17, Sweep h=1/2/5/10/20, Mom-Breadth h=63/126
plus the earlier diagnostics rows in the variance table (50-symbol validation,
test_start bar-gap, pre-CA baseline) → 12 logged ICs.

Round defensively **up** to ``N_RESEARCH_TRIALS = 15`` so DSR is not falsely
optimistic. Variance still comes from the 12 logged Sharpes (dispersion of
observed research outcomes); ``n_trials`` is the selection-bias headcount.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Logged IC→SR proxies for ``var_trial_sharpes`` (dispersion of known outcomes).
RESEARCH_TRIAL_SHARPES: tuple[float, ...] = (
    0.075,
    0.071,
    0.071,
    0.053,
    0.035,
    0.071,
    0.103,
    0.035,
    -0.035,
    -0.071,
    -0.064,
    -0.297,
)

# Conservative selection-bias headcount for live gate scripts (see module doc).
# Must be >= len(RESEARCH_TRIAL_SHARPES); must not track len() alone.
N_RESEARCH_TRIALS: int = 15


def variance_of_trial_sharpes(sharpes: Sequence[float]) -> float:
    """Sample variance (ddof=1) of non-annualized research-trial Sharpes.

    Hard-fails when fewer than two finite values are supplied.
    """
    arr = np.asarray(list(sharpes), dtype=np.float64).reshape(-1)
    if arr.size < 2:
        msg = "variance_of_trial_sharpes requires at least 2 trial Sharpes"
        raise ValueError(msg)
    if not np.isfinite(arr).all():
        msg = "trial Sharpes must be finite"
        raise ValueError(msg)
    return float(np.var(arr, ddof=1))


def research_trial_sharpe_variance() -> float:
    """Variance of the documented Phase-2/3 research-trial Sharpe table."""
    if len(RESEARCH_TRIAL_SHARPES) < 2:
        msg = "RESEARCH_TRIAL_SHARPES must contain at least 2 values"
        raise RuntimeError(msg)
    if N_RESEARCH_TRIALS < len(RESEARCH_TRIAL_SHARPES):
        msg = (
            "N_RESEARCH_TRIALS must be >= len(RESEARCH_TRIAL_SHARPES); "
            "n_trials may round up for unlogged probes but must not undercount "
            "the variance table"
        )
        raise RuntimeError(msg)
    return variance_of_trial_sharpes(RESEARCH_TRIAL_SHARPES)
