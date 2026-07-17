"""Documented research-trial Sharpes for DSR selection-bias correction.

``var_trial_sharpes`` for the Phase-3 gate MUST come from the dispersion of
independent research configurations actually tested — never from CPCV path
Sharpes (those share overlapping train/test windows and are not i.i.d. trials).

Source of ``RESEARCH_TRIAL_SHARPES``
-----------------------------------
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
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Exactly the 12 configurations counted by ``n_trials`` in the live gate scripts.
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

N_RESEARCH_TRIALS: int = len(RESEARCH_TRIAL_SHARPES)


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
    if len(RESEARCH_TRIAL_SHARPES) != N_RESEARCH_TRIALS:
        msg = "RESEARCH_TRIAL_SHARPES length must equal N_RESEARCH_TRIALS"
        raise RuntimeError(msg)
    return variance_of_trial_sharpes(RESEARCH_TRIAL_SHARPES)
