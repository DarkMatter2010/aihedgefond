"""Documented research-trial Sharpes for DSR selection-bias correction.

``var_trial_sharpes`` for the Phase-3 gate MUST come from the dispersion of
independent research configurations actually tested — never from CPCV path
Sharpes (those share overlapping train/test windows and are not i.i.d. trials).

Source of ``RESEARCH_TRIAL_SHARPES`` (variance table)
----------------------------------------------------
Non-annualized daily IR proxies on the same scale as the gate's observed Sharpe
(CS long/short portfolio return mean/std). Values are Grinold-style
``Rank-IC * sqrt(median_cs_breadth)`` (rows 1–12 historically used breadth=50 /
Pearson IC; rows 13+ use the live run's Rank-IC and median breadth).

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
13. Universe-breadth diagnostic h=2 (broad)      rank_IC≈0.014808, N≈494 → 0.329
14. Feature-class triage reversal h=2            → 0.239
15. Feature-class triage reversal h=21           → 0.199
16. Feature-class triage low_vol h=2             → 0.375
17. Feature-class triage low_vol h=21            → 1.078
18. Feature-class triage range_vol h=2           → 0.075
19. Feature-class triage range_vol h=21          → 0.746
20. Feature-class triage all_new h=2             → 0.492
21. Feature-class triage all_new h=21            → 0.775
22. Feature-class triage new_plus_old h=2        → 0.335
23. Feature-class triage new_plus_old h=21       → 0.508
24. Meta-labeling triage (SMA-10 + TB + LGBM binary, broad,
    filtered OOS bet Sharpe, live 2026-07-20)     → 0.049881
25. Small-cap universe diagnostic ALL_NEW (live 2026-07-21,
    seed=42, kept 196/220, best h=2 Rank-IC≈-0.005316, N=196)
    → -0.074422

These are selection-bias inputs, not CPCV path diagnostics.

``N_RESEARCH_TRIALS`` (conservative count for DSR)
-------------------------------------------------
Bailey DSR uses ``n_trials`` for the expected-max-SR null under selection bias.
All 25 rows above are logged outcomes from distinct configurations (12 legacy +
1 breadth diagnostic + 10 feature-class triage + 1 meta-labeling triage +
1 small-cap universe diagnostic).

``N_RESEARCH_TRIALS = 25`` equals ``len(RESEARCH_TRIAL_SHARPES)``.
Variance still comes from the full logged Sharpe tuple.
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
    # 13. Universe-breadth diagnostic (broad, h=2): 0.014808 * sqrt(494)
    0.329,
    # 14–23. Feature-class IC triage (live 2026-07-19, seed=42, N≈494)
    0.239,
    0.199,
    0.375,
    1.078,
    0.075,
    0.746,
    0.492,
    0.775,
    0.335,
    0.508,
    # 24. Meta-labeling triage (live 2026-07-20): filtered OOS bet Sharpe
    0.049881,
    # 25. Small-cap ALL_NEW diagnostic (live 2026-07-21): best Grinold h=2
    -0.074422,
)

# Selection-bias headcount for live gate scripts (see module doc).
# Must be >= len(RESEARCH_TRIAL_SHARPES); must not track len() alone.
N_RESEARCH_TRIALS: int = 25


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
