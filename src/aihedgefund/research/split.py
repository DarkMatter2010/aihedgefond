"""Fixed calendar train/test split with embargo and label-window guards."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from aihedgefund.core.schemas import BaselineDataset, SplitDefinition


def time_embargo_split(
    dataset: BaselineDataset,
    *,
    train_end: date,
    test_start: date,
    embargo_days: int,
    horizon: int,
) -> tuple[BaselineDataset, BaselineDataset, SplitDefinition]:
    """Split by fixed calendar dates with an embargo ≥ horizon.

    Train rows use ``timestamp.date() <= train_end``.
    Test rows use ``timestamp.date() >= test_start``.
    Label windows ``(t, t+h]`` must not overlap the complementary split.
    """
    if embargo_days < horizon:
        msg = "embargo_days must be >= horizon"
        raise ValueError(msg)
    if test_start <= train_end:
        msg = "test_start must be later than train_end"
        raise ValueError(msg)
    calendar_gap = (test_start - train_end).days
    if calendar_gap < embargo_days:
        msg = (
            "calendar gap between train_end and test_start must be "
            f">= embargo_days ({embargo_days}); got {calendar_gap}"
        )
        raise ValueError(msg)

    timestamps = dataset.features.index.get_level_values("timestamp")
    dates = np.array([ts.date() for ts in timestamps])
    train_mask = dates <= train_end
    test_mask = dates >= test_start
    if not train_mask.any():
        msg = "train split is empty"
        raise ValueError(msg)
    if not test_mask.any():
        msg = "test split is empty"
        raise ValueError(msg)

    train = _subset(dataset, train_mask)
    test = _subset(dataset, test_mask)
    _assert_no_label_window_overlap(
        train,
        test,
        horizon=horizon,
        full_index=dataset.features.index,
    )

    train_max = train.features.index.get_level_values("timestamp").max()
    test_min = test.features.index.get_level_values("timestamp").min()
    definition = SplitDefinition(
        train_end=train_end,
        test_start=test_start,
        embargo_days=embargo_days,
        horizon=horizon,
        train_rows=len(train.features),
        test_rows=len(test.features),
        train_max_timestamp=pd.Timestamp(train_max).to_pydatetime(),
        test_min_timestamp=pd.Timestamp(test_min).to_pydatetime(),
    )
    return train, test, definition


def _subset(dataset: BaselineDataset, mask: np.ndarray) -> BaselineDataset:
    """Return a BaselineDataset restricted to the boolean mask."""
    features = dataset.features.iloc[mask]
    label = dataset.label.iloc[mask]
    return BaselineDataset(
        features=features,
        label=label,
        horizon=dataset.horizon,
        feature_columns=dataset.feature_columns,
    )


def _assert_no_label_window_overlap(
    train: BaselineDataset,
    test: BaselineDataset,
    *,
    horizon: int,
    full_index: pd.MultiIndex,
) -> None:
    """Fail if any train label window ``(t, t+h]`` reaches a test feature time.

    The full pre-split index (including the embargo zone) is required so the
    trading-bar gap is not undercounted when embargo rows are excluded from
    both train and test.
    """
    for symbol in sorted(set(train.features.index.get_level_values("symbol"))):
        if symbol not in test.features.index.get_level_values("symbol"):
            continue
        full_symbol_ts = full_index.get_level_values("timestamp")[
            full_index.get_level_values("symbol") == symbol
        ].unique()
        full_symbol_ts = pd.DatetimeIndex(full_symbol_ts).sort_values()
        max_tr = train.features.xs(symbol, level="symbol").index.max()
        min_te = test.features.xs(symbol, level="symbol").index.min()
        train_pos = full_symbol_ts.get_loc(max_tr)
        test_pos = full_symbol_ts.get_loc(min_te)
        if not isinstance(train_pos, int) or not isinstance(test_pos, int):
            msg = f"{symbol}: ambiguous timestamp positions in split calendar"
            raise ValueError(msg)
        # shift(-h) at train_pos reads close at train_pos + h; require strict
        # separation so that bar does not land on (or past) the first test bar.
        bar_gap = test_pos - train_pos
        if bar_gap <= horizon:
            msg = (
                f"{symbol}: bar gap {bar_gap} between train/test is <= horizon "
                f"{horizon}; need bar_gap > horizon or label windows overlap"
            )
            raise ValueError(msg)
