"""Point-in-time joins and reusable look-ahead assertions."""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from aihedgefund.core.schemas import PointInTimeFrame


def assert_no_lookahead(df: pd.DataFrame, anchor_col: str) -> None:
    """Raise when any timestamp-bearing column is later than its row anchor."""
    if anchor_col not in df:
        msg = f"anchor column {anchor_col!r} is missing"
        raise ValueError(msg)
    anchors = pd.to_datetime(df[anchor_col], utc=True)
    for column in df.columns:
        if column == anchor_col or not is_datetime64_any_dtype(df[column].dtype):
            continue
        source_times = pd.to_datetime(df[column], utc=True)
        leaked = source_times.notna() & anchors.notna() & source_times.gt(anchors)
        if leaked.any():
            msg = f"look-ahead detected in {column!r}"
            raise ValueError(msg)


def pit_join(
    features: PointInTimeFrame,
    targets: PointInTimeFrame,
) -> PointInTimeFrame:
    """Backward-asof join targets onto feature anchors without future matches."""
    left = _timestamp_column(features.frame, "timestamp")
    right = _timestamp_column(targets.frame, "timestamp").rename(
        columns={"timestamp": "target_timestamp"}
    )
    left_has_symbol = "symbol" in left
    right_has_symbol = "symbol" in right
    if left_has_symbol != right_has_symbol:
        msg = "symbol must be present on both PIT join inputs or neither"
        raise ValueError(msg)
    by = "symbol" if left_has_symbol else None

    left_keys = ["timestamp", "symbol"] if by else ["timestamp"]
    right_keys = ["target_timestamp", "symbol"] if by else ["target_timestamp"]
    if left.duplicated(left_keys).any() or right.duplicated(right_keys).any():
        msg = "PIT join keys must be unique"
        raise ValueError(msg)

    left_sort = ["timestamp", "symbol"] if by else ["timestamp"]
    right_sort = ["target_timestamp", "symbol"] if by else ["target_timestamp"]
    joined = pd.merge_asof(
        left.sort_values(left_sort),
        right.sort_values(right_sort),
        left_on="timestamp",
        right_on="target_timestamp",
        by=by,
        direction="backward",
        allow_exact_matches=True,
        suffixes=("", "_target"),
    )
    assert_no_lookahead(joined, "timestamp")
    return PointInTimeFrame(frame=joined)


def _timestamp_column(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    result = frame.copy()
    if name not in result:
        if isinstance(result.index, pd.DatetimeIndex):
            result = result.rename_axis(name).reset_index()
        elif name in result.index.names:
            result = result.reset_index()
        else:
            msg = f"frame must have a {name!r} column or index level"
            raise ValueError(msg)
    result[name] = pd.to_datetime(result[name], utc=True)
    return result
