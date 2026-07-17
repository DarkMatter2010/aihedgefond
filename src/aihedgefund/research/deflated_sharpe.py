"""Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

All Sharpe inputs/outputs in this module are **non-annualized** unless a helper
explicitly documents an annualization convention. ``n_trials`` is always an
explicit caller-supplied parameter — never inferred — so selection-bias
deflation stays honest.

``T`` / ``n_obs``
    Always the number of **return observations** in the evaluated return
    series (``len(returns)``). It is **not** the number of CPCV folds/paths.
    Callers that merge overlapping CPCV OOS paths into one series must pass
    that merged series so ``T`` matches the unique OOS calendar length.

``var_trial_sharpes``
    Variance of the **independent research-trial** non-annualized Sharpes
    (the configurations actually tested). Must **not** be estimated from
    CPCV path Sharpes — those paths share overlapping train windows and are
    not an iid sample for selection-bias deflation. Same non-annualized
    scale as ``observed_sharpe`` / ``sr0``.
"""

from __future__ import annotations

from math import e as EULER_E

import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import norm, skew

from aihedgefund.core.schemas import DeflatedSharpeReport, SharpeReport

# Euler–Mascheroni constant used in the expected-max-SR approximation.
EULER_GAMMA = 0.5772156649015328606


def sharpe_ratio(returns: np.ndarray | list[float]) -> SharpeReport:
    """Non-annualized Sharpe of ``returns`` with skew and Pearson kurtosis.

    Hard-fails on empty input, non-finite values, or zero variance.
    Kurtosis is Pearson (normal == 3), matching Bailey & López de Prado (2014).
    """
    arr = _as_finite_1d(returns, name="returns")
    if arr.size < 2:
        msg = "sharpe_ratio requires at least 2 observations"
        raise ValueError(msg)
    std = float(np.std(arr, ddof=1))
    # Identical (or numerically identical) returns → undefined Sharpe.
    if (not np.isfinite(std)) or std <= 0.0 or np.unique(arr).size < 2:
        msg = "sharpe_ratio requires non-zero return variance"
        raise ValueError(msg)
    mean = float(np.mean(arr))
    sr = mean / std
    # bias=True matches the population-style moments used in the DSR paper examples.
    g3 = float(skew(arr, bias=True))
    g4 = float(scipy_kurtosis(arr, fisher=False, bias=True))
    if not np.isfinite(g3) or not np.isfinite(g4):
        msg = "sharpe_ratio requires finite skewness and kurtosis"
        raise ValueError(msg)
    return SharpeReport(sharpe=sr, n_obs=int(arr.size), skewness=g3, kurtosis=g4)


def expected_max_sharpe(n_trials: int, var_trial_sharpes: float) -> float:
    """Expected maximum Sharpe under the null given ``n_trials`` (SR0).

    ``var_trial_sharpes`` is the variance of **non-annualized research-trial**
    Sharpes (independent configurations), not CPCV path variance.
    Hard-fails when ``n_trials < 2`` or variance is negative.
    """
    if n_trials < 2:
        msg = "n_trials must be >= 2"
        raise ValueError(msg)
    if var_trial_sharpes < 0.0 or not np.isfinite(var_trial_sharpes):
        msg = "var_trial_sharpes must be a finite non-negative float"
        raise ValueError(msg)
    if var_trial_sharpes == 0.0:
        return 0.0
    z1 = float(norm.ppf(1.0 - 1.0 / n_trials))
    z2 = float(norm.ppf(1.0 - 1.0 / (n_trials * EULER_E)))
    return float(np.sqrt(var_trial_sharpes) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe(
    returns: np.ndarray | list[float],
    *,
    n_trials: int,
    var_trial_sharpes: float,
) -> DeflatedSharpeReport:
    """Probability that the true SR exceeds 0 after selection-bias deflation.

    Implements Bailey & López de Prado (2014):

    ``DSR = Φ( (SR* - SR0) * sqrt(T-1) / sqrt(1 - γ3·SR* + ((γ4-1)/4)·SR*²) )``

    ``T`` is ``len(returns)`` (return observations of the evaluated series).
    ``n_trials`` / ``var_trial_sharpes`` describe the independent research
    configurations actually tested (not CPCV fold counts / path variance).
    ``SR*`` and ``SR0`` share the non-annualized scale of ``returns``.
    """
    report = sharpe_ratio(returns)
    # report.n_obs == T == number of return observations (not CPCV paths).
    sr0 = expected_max_sharpe(n_trials, var_trial_sharpes)
    dsr = _dsr_from_moments(
        observed_sharpe=report.sharpe,
        sr0=sr0,
        n_obs=report.n_obs,
        skewness=report.skewness,
        kurtosis=report.kurtosis,
    )
    return DeflatedSharpeReport(
        observed_sharpe=report.sharpe,
        sr0=sr0,
        dsr=dsr,
        n_trials=n_trials,
        var_trial_sharpes=float(var_trial_sharpes),
        n_obs=report.n_obs,
        skewness=report.skewness,
        kurtosis=report.kurtosis,
    )


def deflated_sharpe_from_moments(
    *,
    observed_sharpe: float,
    n_obs: int,
    skewness: float,
    kurtosis: float,
    n_trials: int,
    var_trial_sharpes: float,
) -> DeflatedSharpeReport:
    """DSR from pre-computed moments (for published numerical examples)."""
    if n_obs < 2:
        msg = "n_obs must be >= 2"
        raise ValueError(msg)
    for name, value in (
        ("observed_sharpe", observed_sharpe),
        ("skewness", skewness),
        ("kurtosis", kurtosis),
    ):
        if not np.isfinite(value):
            msg = f"{name} must be finite"
            raise ValueError(msg)
    sr0 = expected_max_sharpe(n_trials, var_trial_sharpes)
    dsr = _dsr_from_moments(
        observed_sharpe=observed_sharpe,
        sr0=sr0,
        n_obs=n_obs,
        skewness=skewness,
        kurtosis=kurtosis,
    )
    return DeflatedSharpeReport(
        observed_sharpe=float(observed_sharpe),
        sr0=sr0,
        dsr=dsr,
        n_trials=n_trials,
        var_trial_sharpes=float(var_trial_sharpes),
        n_obs=n_obs,
        skewness=float(skewness),
        kurtosis=float(kurtosis),
    )


def _dsr_from_moments(
    *,
    observed_sharpe: float,
    sr0: float,
    n_obs: int,
    skewness: float,
    kurtosis: float,
) -> float:
    """Core DSR transform; moments use the observed SR* (paper Eq. 1)."""
    moment_term = (
        1.0
        - skewness * observed_sharpe
        + ((kurtosis - 1.0) / 4.0) * observed_sharpe * observed_sharpe
    )
    if moment_term <= 0.0:
        msg = f"non-positive DSR moment term: {moment_term}"
        raise ValueError(msg)
    z = (observed_sharpe - sr0) * np.sqrt(n_obs - 1) / np.sqrt(moment_term)
    return float(norm.cdf(z))


def _as_finite_1d(values: np.ndarray | list[float], *, name: str) -> np.ndarray:
    """Coerce to a finite 1-d float64 array or hard-fail."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        msg = f"{name} must be non-empty"
        raise ValueError(msg)
    if not np.isfinite(arr).all():
        msg = f"{name} must contain only finite values"
        raise ValueError(msg)
    return arr
