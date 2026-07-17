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

# Published / noted IC means (or rank-IC when Pearson was unavailable) for the
# 12 independent configurations counted as n_trials in the live gate scripts.
# Sources: PR #16–#19 live notes and the multi-horizon sweep report.
#
# Index mapping (must stay length-12 and ordered):
#  1 baseline h=5 (9-feat, post-CA)          ic_mean ≈ 0.00748
#  2 50-symbol IC validation                 same order (not separately logged)
#  3 test_start bar-gap fix re-run           same order
#  4 CA fix re-run h=5                       ic_mean ≈ 0.00748
#  5 Feature-set #17 at h=5                  ic_mean ≈ 0.0049
#  6 sweep h=1                               between 0 and h=2 (approx 0.010)
#  7 sweep h=2 (best candidate)              ic_mean ≈ 0.0146
#  8 sweep h=5                               ic_mean ≈ 0.0049
#  9 sweep h=10                              negative ic_mean (approx −0.005)
# 10 sweep h=20                              negative ic_mean (approx −0.008)
# 11 momentum-breadth h=63                   rank_ic ≈ −0.009
# 12 momentum-breadth h=126                  rank_ic ≈ −0.042
RESEARCH_TRIAL_ICS: tuple[float, ...] = (
    0.00748,
    0.00748,
    0.00748,
    0.00748,
    0.0049,
    0.0100,
    0.0146,
    0.0049,
    -0.0050,
    -0.0080,
    -0.0090,
    -0.0420,
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
