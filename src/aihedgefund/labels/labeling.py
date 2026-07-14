"""Self-built event sampling, triple-barrier labels, and overlap weights."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def daily_volatility(close: pd.Series, span: int) -> pd.Series:
    """Causal EWM standard deviation of close-to-close log returns."""
    if span < 2:
        msg = "span must be at least two"
        raise ValueError(msg)
    returns = np.log(close.astype(float) / close.astype(float).shift(1))
    return returns.ewm(span=span, adjust=False, min_periods=2).std().rename("daily_volatility")


def cusum_filter(close: pd.Series, threshold: float) -> pd.DatetimeIndex:
    """Return timestamps where a symmetric CUSUM exceeds its fixed threshold."""
    if threshold <= 0:
        msg = "threshold must be positive"
        raise ValueError(msg)
    if not isinstance(close.index, pd.DatetimeIndex):
        msg = "close must use a DatetimeIndex"
        raise ValueError(msg)
    log_changes = np.log(close.astype(float)).diff().dropna()
    positive_sum = 0.0
    negative_sum = 0.0
    events: list[pd.Timestamp] = []
    for timestamp, change in log_changes.items():
        value = float(change)
        positive_sum = max(0.0, positive_sum + value)
        negative_sum = min(0.0, negative_sum + value)
        if positive_sum > threshold:
            positive_sum = 0.0
            events.append(pd.Timestamp(timestamp))
        elif negative_sum < -threshold:
            negative_sum = 0.0
            events.append(pd.Timestamp(timestamp))
    return pd.DatetimeIndex(events, name=close.index.name)


def triple_barrier(
    close: pd.Series,
    events: Iterable[pd.Timestamp],
    pt: float,
    sl: float,
    vertical_bars: int,
    side: pd.Series | None = None,
    vol: pd.Series | None = None,
) -> pd.DataFrame:
    """Label first horizontal touches or zero at the vertical holding barrier."""
    if pt <= 0 or sl <= 0 or vertical_bars < 1:
        msg = "pt, sl, and vertical_bars must be positive"
        raise ValueError(msg)
    if not isinstance(close.index, pd.DatetimeIndex) or not close.index.is_monotonic_increasing:
        msg = "close must have a sorted DatetimeIndex"
        raise ValueError(msg)
    if close.index.has_duplicates or close.isna().any() or (close <= 0).any():
        msg = "close must contain unique, positive, non-null observations"
        raise ValueError(msg)

    volatility = vol if vol is not None else daily_volatility(close, span=20)
    rows: list[dict[str, object]] = []
    for event in events:
        t0 = pd.Timestamp(event)
        if t0 not in close.index:
            raise ValueError(f"event {t0} is not present in close")
        position = int(close.index.get_loc(t0))
        target = float(volatility.loc[t0])
        if not np.isfinite(target) or target <= 0:
            raise ValueError(f"event {t0} has no positive volatility target")

        event_side = 1.0
        if side is not None:
            side_value = side.reindex(close.index).loc[t0]
            if pd.isna(side_value) or float(side_value) not in (-1.0, 1.0):
                raise ValueError(f"event {t0} has no valid side")
            event_side = float(side_value)

        vertical_position = min(position + vertical_bars, len(close) - 1)
        path = close.iloc[position + 1 : vertical_position + 1]
        signed_path_returns = (path / float(close.iloc[position]) - 1.0) * event_side
        upper_touches = signed_path_returns[signed_path_returns >= pt * target]
        lower_touches = signed_path_returns[signed_path_returns <= -sl * target]
        upper_time = upper_touches.index[0] if not upper_touches.empty else None
        lower_time = lower_touches.index[0] if not lower_touches.empty else None

        if upper_time is not None and (lower_time is None or upper_time <= lower_time):
            t1 = pd.Timestamp(upper_time)
            directional_label = 1
        elif lower_time is not None:
            t1 = pd.Timestamp(lower_time)
            directional_label = -1
        else:
            t1 = pd.Timestamp(close.index[vertical_position])
            directional_label = 0

        raw_return = float(close.loc[t1] / close.loc[t0] - 1.0)
        realized_return = raw_return * event_side
        label = directional_label if side is None else int(realized_return > 0.0)
        rows.append(
            {
                "label": label,
                "t0": t0,
                "t1": t1,
                "ret": realized_return,
            }
        )

    if not rows:
        return pd.DataFrame(columns=("label", "t0", "t1", "ret")).set_index("t0", drop=False)
    result = pd.DataFrame.from_records(rows)
    result.index = pd.DatetimeIndex(result["t0"], name="event")
    return result


def sample_weights(
    events_t1: pd.Series,
    observation_index: pd.DatetimeIndex | None = None,
) -> pd.Series:
    """Average inverse concurrency for each inclusive event interval."""
    if not isinstance(events_t1.index, pd.DatetimeIndex):
        msg = "events_t1 must use event starts as a DatetimeIndex"
        raise ValueError(msg)
    if events_t1.empty:
        return pd.Series(dtype=float, index=events_t1.index, name="weight")
    starts = pd.DatetimeIndex(events_t1.index)
    ends = pd.DatetimeIndex(pd.to_datetime(events_t1, utc=starts.tz is not None))
    if (ends < starts).any():
        msg = "event end times must not precede starts"
        raise ValueError(msg)

    if observation_index is None:
        observation_times = pd.date_range(
            starts.min().normalize(),
            ends.max().normalize(),
            freq="D",
            tz=starts.tz,
        )
    else:
        observation_times = observation_index[
            (observation_index >= starts.min()) & (observation_index <= ends.max())
        ].sort_values()
    if observation_times.empty:
        msg = "observation_index does not cover any event interval"
        raise ValueError(msg)
    concurrency = pd.Series(0.0, index=observation_times)
    for start, end in zip(starts, ends, strict=True):
        concurrency.loc[(concurrency.index >= start) & (concurrency.index <= end)] += 1.0

    weights = []
    for start, end in zip(starts, ends, strict=True):
        interval_concurrency = concurrency.loc[
            (concurrency.index >= start) & (concurrency.index <= end)
        ]
        weights.append(float((1.0 / interval_concurrency).mean()))
    return pd.Series(weights, index=events_t1.index, dtype=float, name="weight")
