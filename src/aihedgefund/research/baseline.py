"""Deterministic LightGBM regressor baseline for Phase 2."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import lightgbm as lgb
import pandas as pd

from aihedgefund.core.schemas import BaselineDataset, PredictionOutput


def build_lgbm_params(
    *,
    seed: int,
    learning_rate: float,
    num_leaves: int,
    min_data_in_leaf: int,
    feature_fraction: float,
    bagging_fraction: float,
    bagging_freq: int,
) -> dict[str, Any]:
    """Return fully deterministic LightGBM regressor parameters."""
    return {
        "objective": "regression",
        "boosting_type": "gbdt",
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "min_data_in_leaf": min_data_in_leaf,
        "feature_fraction": feature_fraction,
        "bagging_fraction": bagging_fraction,
        "bagging_freq": bagging_freq,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": 1,
        "seed": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        "data_random_seed": seed,
    }


def train_baseline(
    train: BaselineDataset,
    *,
    params: Mapping[str, Any],
    num_boost_round: int,
) -> lgb.Booster:
    """Fit a deterministic LightGBM regressor on the training split."""
    if num_boost_round < 1:
        msg = "num_boost_round must be >= 1"
        raise ValueError(msg)
    if params.get("objective") != "regression":
        msg = "Phase-2 baseline requires objective='regression'"
        raise ValueError(msg)
    if params.get("deterministic") is not True:
        msg = "Phase-2 baseline requires deterministic=True"
        raise ValueError(msg)

    feature_names = list(train.feature_columns)
    dataset = lgb.Dataset(
        train.features.loc[:, feature_names],
        label=train.label.to_numpy(dtype=float),
        feature_name=feature_names,
        free_raw_data=False,
    )
    return lgb.train(
        dict(params),
        dataset,
        num_boost_round=num_boost_round,
    )


def predict_scores(
    model: lgb.Booster,
    features: pd.DataFrame,
    *,
    model_hash: str,
) -> PredictionOutput:
    """Score each ``(timestamp, symbol)`` row with the fitted regressor."""
    feature_names = list(model.feature_name())
    missing = [name for name in feature_names if name not in features.columns]
    if missing:
        msg = f"prediction features missing columns: {missing}"
        raise ValueError(msg)
    matrix = features.loc[:, feature_names]
    if matrix.isna().any().any():
        msg = "prediction features must not contain NaNs"
        raise ValueError(msg)
    raw_scores = model.predict(matrix)
    scores = pd.Series(raw_scores, index=features.index, name="score", dtype="float64")
    return PredictionOutput(scores=scores, model_hash=model_hash)
