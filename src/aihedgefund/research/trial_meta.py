"""Documented research-trial Sharpe inputs for DSR selection-bias deflation.

``var_trial_sharpes`` for SR0 must come from the variance of the *independent
research configurations actually tested* — never from CPCV path Sharpes
(those paths share overlapping training windows and are not an iid sample).

Trial Sharpes here are **non-annualized daily** estimates so they share the
same scale as ``sharpe_ratio`` / ``deflated_sharpe`` outputs.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt

import numpy as np

# Universe size used in the Phase-2/3 research runs that produced these trials.
RESEARCH_TRIAL_N_SYMBOLS = 50

# Documented IC inputs for the 12 independent configurations counted as
# ``n_trials`` in the live gate scripts (same order as scripts/run_cpcv_dsr_gate.py).
#
# Metric: Pearson ``ic_mean`` from live Yahoo reports where available; for the
# momentum-breadth probe (trials 11–12) the decision metric was ``rank_ic_mean``
# (materiality threshold), so those two use rank-IC.
#
# Sources (Slack / PR live notes, 2026-07-16–17):
#  1  Phase-2 baseline h=5 (9-feat, post-CA era)     PR#16 ic_mean
#  2  50-symbol IC validation re-run                 PR#15 ic_mean (same era log)
#  3  test_start bar-gap fix re-run                  PR#15 ic_mean
#  4  Corporate-actions fix re-run h=5               PR#16 ic_mean
#  5  Feature-set #17 at h=5                         PR#17 ic_mean
#  6  Multi-horizon sweep h=1                        PR#18 ic_mean
#  7  Multi-horizon sweep h=2 (gate candidate)       PR#18 ic_mean
#  8  Multi-horizon sweep h=5                        PR#18 ic_mean
#  9  Multi-horizon sweep h=10                       PR#18 ic_mean
# 10  Multi-horizon sweep h=20                       PR#18 ic_mean
# 11  Momentum-breadth probe h=63                    PR#19 rank_ic_mean
# 12  Momentum-breadth probe h=126                   PR#19 rank_ic_mean
RESEARCH_TRIAL_ICS: tuple[float, ...] = (
    0.00748130792855417,
    0.010580912018430077,
    0.010580912018430077,
    0.00748130792855417,
    -0.0019334085480568778,
    -0.001361,
    0.014570,
    0.004937,
    -0.016262,
    -0.012031,
    -0.008891,
    -0.042408,
)


def ic_to_nonann_sharpe(ic: float, *, n_symbols: int = RESEARCH_TRIAL_N_SYMBOLS) -> float:
    """Grinold CS proxy: non-annualized daily SR ≈ IC · √N.

    Used only to convert documented research ICs into Sharpe-scale trial
    values for ``var_trial_sharpes``. Hard-fails on invalid breadth.
    """
    if n_symbols < 2:
        msg = "n_symbols must be >= 2"
        raise ValueError(msg)
    if not np.isfinite(ic):
        msg = "ic must be finite"
        raise ValueError(msg)
    return float(ic) * sqrt(float(n_symbols))


def research_trial_sharpes(
    trial_ics: Sequence[float] = RESEARCH_TRIAL_ICS,
    *,
    n_symbols: int = RESEARCH_TRIAL_N_SYMBOLS,
) -> tuple[float, ...]:
    """Non-annualized trial Sharpes from documented research ICs."""
    if len(trial_ics) < 2:
        msg = "need at least 2 research trial ICs"
        raise ValueError(msg)
    return tuple(ic_to_nonann_sharpe(ic, n_symbols=n_symbols) for ic in trial_ics)


def variance_of_trial_sharpes(trial_sharpes: Sequence[float]) -> float:
    """Sample variance (ddof=1) of independent research-trial Sharpes.

    This is the only legitimate source of ``var_trial_sharpes`` for SR0 in the
    overfitting gate. Hard-fails when fewer than 2 finite values are supplied.
    """
    arr = np.asarray(list(trial_sharpes), dtype=np.float64).reshape(-1)
    if arr.size < 2:
        msg = "variance_of_trial_sharpes requires at least 2 trial Sharpes"
        raise ValueError(msg)
    if not np.isfinite(arr).all():
        msg = "trial Sharpes must be finite"
        raise ValueError(msg)
    return float(np.var(arr, ddof=1))


def research_var_trial_sharpes(
    trial_ics: Sequence[float] = RESEARCH_TRIAL_ICS,
    *,
    n_symbols: int = RESEARCH_TRIAL_N_SYMBOLS,
) -> float:
    """``var_trial_sharpes`` from the documented Phase-2/3 research trials."""
    return variance_of_trial_sharpes(
        research_trial_sharpes(trial_ics, n_symbols=n_symbols)
    )
