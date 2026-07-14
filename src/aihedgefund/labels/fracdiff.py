"""Causal fixed-width fractional differentiation and ADF-based d selection."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from aihedgefund.core.config import FracDiffSettings


def configured_frac_diff(series: pd.Series, settings: FracDiffSettings) -> pd.Series:
    """Apply the validated Phase 1 fractional-differentiation configuration."""
    return frac_diff_ffd(series, settings.d, settings.thresh)


def frac_diff_ffd(series: pd.Series, d: float, thresh: float) -> pd.Series:
    """Apply constant fractional weights over one fixed-width trailing window."""
    if not 0.0 <= d <= 1.0:
        msg = "d must be between zero and one"
        raise ValueError(msg)
    if not 0.0 < thresh < 1.0:
        msg = "thresh must be between zero and one"
        raise ValueError(msg)
    if series.isna().any():
        msg = "series must not contain NaNs"
        raise ValueError(msg)

    weights = _fixed_width_weights(d, thresh, len(series))
    width = len(weights)
    values = series.astype(float)
    output = pd.Series(np.nan, index=series.index, dtype=float, name=series.name)
    for end in range(width - 1, len(values)):
        window = values.iloc[end - width + 1 : end + 1].to_numpy(dtype=float)
        output.iloc[end] = float(np.dot(weights, window))
    return output


def min_ffd_d(
    series: pd.Series,
    *,
    thresh: float = 1e-4,
    significance: float = 0.05,
    step: float = 0.1,
) -> float:
    """Return the smallest grid d whose fixed-width output passes the ADF test."""
    if not 0.0 < significance < 1.0 or not 0.0 < step <= 1.0:
        msg = "significance and step must be in (0, 1]"
        raise ValueError(msg)
    candidates = np.arange(0.0, 1.0 + step / 2.0, step)
    for candidate in candidates:
        differentiated = frac_diff_ffd(series, float(min(candidate, 1.0)), thresh).dropna()
        if len(differentiated) < 20:
            continue
        adf_result = cast(
            tuple[float, float, int, int, dict[str, float], float],
            adfuller(
                differentiated.to_numpy(dtype=float),
                maxlag=1,
                regression="c",
                autolag=None,
            ),
        )
        if adf_result[1] < significance:
            return float(round(min(candidate, 1.0), 10))
    msg = "no d in [0, 1] passed the ADF stationarity threshold"
    raise ValueError(msg)


def _fixed_width_weights(d: float, thresh: float, max_size: int) -> np.ndarray:
    weights = [1.0]
    for lag in range(1, max_size):
        next_weight = -weights[-1] * (d - lag + 1.0) / lag
        if abs(next_weight) < thresh:
            break
        weights.append(next_weight)
    return np.asarray(weights[::-1], dtype=float)
