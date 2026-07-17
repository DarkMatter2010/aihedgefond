"""Hand-rolled causal technical indicators using only current and prior bars."""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_return(close: pd.Series) -> pd.Series:
    """One-bar close-to-close log return."""
    return np.log(close.astype(float) / close.astype(float).shift(1)).rename("log_return")


def realized_volatility(close: pd.Series, span: int) -> pd.Series:
    """Causal exponentially weighted standard deviation of log returns."""
    if span < 2:
        msg = "span must be at least two"
        raise ValueError(msg)
    return (
        log_return(close)
        .ewm(span=span, adjust=False, min_periods=span)
        .std()
        .rename(f"realized_vol_{span}")
    )


def momentum(close: pd.Series, periods: int) -> pd.Series:
    """Close relative to the close exactly ``periods`` completed bars earlier."""
    if periods < 1:
        msg = "periods must be positive"
        raise ValueError(msg)
    return (close.astype(float) / close.astype(float).shift(periods) - 1.0).rename(
        f"momentum_{periods}"
    )


def moving_average_ratio(close: pd.Series, window: int) -> pd.Series:
    """Current close divided by its trailing simple moving average."""
    if window < 2:
        msg = "window must be at least two"
        raise ValueError(msg)
    trailing_mean = close.astype(float).rolling(window, min_periods=window).mean()
    return (close.astype(float) / trailing_mean).rename(f"ma_ratio_{window}")


def rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder-style relative-strength index from completed close changes."""
    if period < 2:
        msg = "period must be at least two"
        raise ValueError(msg)
    delta = close.astype(float).diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    average_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    average_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    relative_strength = average_gain / average_loss
    values = 100.0 - (100.0 / (1.0 + relative_strength))
    values = values.mask((average_gain == 0.0) & (average_loss == 0.0))
    return values.rename(f"rsi_{period}")


def macd(
    close: pd.Series,
    fast: int,
    slow: int,
    signal: int,
) -> pd.DataFrame:
    """Moving-average convergence/divergence line and trailing signal line."""
    if min(fast, slow, signal) < 1 or fast >= slow:
        msg = "MACD windows must be positive and fast must be less than slow"
        raise ValueError(msg)
    values = close.astype(float)
    fast_ema = values.ewm(span=fast, adjust=False, min_periods=fast).mean()
    slow_ema = values.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = (fast_ema - slow_ema).rename("macd")
    signal_line = (
        macd_line.ewm(span=signal, adjust=False, min_periods=signal)
        .mean()
        .rename("macd_signal")
    )
    return pd.concat((macd_line, signal_line), axis=1)


def atr(frame: pd.DataFrame, period: int) -> pd.Series:
    """Wilder average true range using current OHLC and the prior close."""
    if period < 2:
        msg = "period must be at least two"
        raise ValueError(msg)
    prior_close = frame["close"].astype(float).shift(1)
    true_ranges = pd.concat(
        (
            frame["high"].astype(float) - frame["low"].astype(float),
            (frame["high"].astype(float) - prior_close).abs(),
            (frame["low"].astype(float) - prior_close).abs(),
        ),
        axis=1,
    )
    true_range = true_ranges.max(axis=1)
    return (
        true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=period)
        .mean()
        .rename(f"atr_{period}")
    )


def rolling_zscore(close: pd.Series, window: int) -> pd.Series:
    """Current close z-score against a trailing fixed-length window."""
    if window < 2:
        msg = "window must be at least two"
        raise ValueError(msg)
    values = close.astype(float)
    mean = values.rolling(window, min_periods=window).mean()
    std = values.rolling(window, min_periods=window).std(ddof=0)
    return ((values - mean) / std.replace(0.0, np.nan)).rename(f"close_zscore_{window}")


def rolling_return_std(close: pd.Series, window: int) -> pd.Series:
    """Causal rolling standard deviation of completed log returns."""
    if window < 2:
        msg = "window must be at least two"
        raise ValueError(msg)
    return (
        log_return(close)
        .rolling(window, min_periods=window)
        .std(ddof=0)
        .rename(f"ret_std_{window}")
    )


def mean_reversion(close: pd.Series, window: int) -> pd.Series:
    """Close deviation from its trailing SMA, scaled by the SMA."""
    if window < 2:
        msg = "window must be at least two"
        raise ValueError(msg)
    values = close.astype(float)
    trailing_mean = values.rolling(window, min_periods=window).mean()
    return ((values - trailing_mean) / trailing_mean).rename(f"mean_reversion_{window}")


def gain_loss_ratio(close: pd.Series, period: int) -> pd.Series:
    """Wilder-smoothed average-gain / average-loss ratio (RSI numerator path)."""
    if period < 2:
        msg = "period must be at least two"
        raise ValueError(msg)
    delta = close.astype(float).diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    average_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    average_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    ratio = average_gain / average_loss.replace(0.0, np.nan)
    ratio = ratio.mask((average_gain == 0.0) & (average_loss == 0.0))
    return ratio.rename(f"gain_loss_ratio_{period}")


def volume_ratio(volume: pd.Series, window: int) -> pd.Series:
    """Current volume divided by its trailing simple moving average."""
    if window < 2:
        msg = "window must be at least two"
        raise ValueError(msg)
    values = volume.astype(float)
    trailing_mean = values.rolling(window, min_periods=window).mean()
    return (values / trailing_mean.replace(0.0, np.nan)).rename(f"volume_ratio_{window}")
