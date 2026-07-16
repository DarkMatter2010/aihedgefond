"""Assemble Phase-1 features with Phase-2 forward-return labels."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from aihedgefund.core.schemas import BaselineDataset
from aihedgefund.features.pipeline import FEATURE_COLUMNS


def assemble_baseline_dataset(
    feature_matrix: pd.DataFrame,
    labels: pd.Series,
    *,
    horizon: int,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> BaselineDataset:
    """Inner-join causal features and forward returns; drop incomplete rows.

    Hard-fails if the feature matrix schema does not match the Phase-1 contract
    or if any surviving row still contains NaNs.
    """
    columns = tuple(feature_columns)
    if list(feature_matrix.index.names) != ["timestamp", "symbol"]:
        msg = "feature matrix index names must be ('timestamp', 'symbol')"
        raise ValueError(msg)
    if tuple(feature_matrix.columns) != columns:
        msg = f"feature matrix columns must be {columns}"
        raise ValueError(msg)

    aligned_features = feature_matrix.astype(
        {column: "float64" for column in columns}
    )
    aligned_labels = labels.astype("float64")
    common = aligned_features.index.intersection(aligned_labels.index)
    if common.empty:
        msg = "features and labels have no overlapping index rows"
        raise ValueError(msg)

    features = aligned_features.loc[common].sort_index()
    label = aligned_labels.loc[common].sort_index().rename(f"forward_return_{horizon}")
    complete = features.notna().all(axis=1) & label.notna()
    features = features.loc[complete]
    label = label.loc[complete]
    if features.empty:
        msg = "baseline dataset is empty after dropping NaN feature/label rows"
        raise ValueError(msg)

    return BaselineDataset(
        features=features,
        label=label,
        horizon=horizon,
        feature_columns=columns,
    )
