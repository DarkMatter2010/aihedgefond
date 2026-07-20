"""Hardened CPCV/DSR gate for the meta-labeling (SMA-10 + Triple-Barrier) candidate.

Validates the already-logged triage trial (row 24). Does **not** increment
``N_RESEARCH_TRIALS``.

Uses LightGBM binary accept/reject (not the regression baseline). Path returns
are equal-weight means of accepted bet realized returns per timestamp, then
merged across CPCV paths for one DSR (same merge rule as ``gate.py``).

CPCV knobs match the prior hardened gate (N=6, k=2). Horizon / embargo =
``vertical_bars`` from Triple-Barrier (10).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any, Final

import lightgbm as lgb
import numpy as np
import pandas as pd

from aihedgefund.core.config import LabelSettings, Settings
from aihedgefund.core.schemas import (
    BarFrame,
    BaselineDataset,
    CPCVConfig,
    DeflatedSharpeReport,
    GatePathResult,
    GateVerdict,
)
from aihedgefund.features.feature_classes import ALL_NEW_FEATURE_CLASS_COLUMNS
from aihedgefund.research.all_new_gate import interpret_corrected_verdict
from aihedgefund.research.cpcv import combinatorial_purged_splits, subset_by_positions
from aihedgefund.research.deflated_sharpe import deflated_sharpe, sharpe_ratio
from aihedgefund.research.gate import merge_cpcv_path_returns
from aihedgefund.research.meta_labeling import (
    ACCEPT_PROBABILITY_THRESHOLD,
    META_FEATURE_COLUMNS,
    PRIMARY_MA_WINDOW,
    assemble_meta_dataset,
    build_lgbm_binary_params,
    predict_accept_proba,
    train_meta_classifier,
)
from aihedgefund.research.model_hash import compute_model_hash
from aihedgefund.research.research_trials import (
    N_RESEARCH_TRIALS,
    research_trial_sharpe_variance,
)

SEED: Final[int] = 42
N_TRIALS: Final[int] = N_RESEARCH_TRIALS
N_BLOCKS: Final[int] = 6
N_TEST_BLOCKS: Final[int] = 2
# Planned full permutation-null size (offline tests assert this). Live scripts
# default to a smaller M via env ``AIHF_GATE_N_PERMUTATIONS`` (see runner).
N_PERMUTATIONS: Final[int] = 100
PERM_SEED: Final[int] = 20260721
DSR_THRESHOLD: Final[float] = 0.95
VERTICAL_BARS: Final[int] = 10  # limits.yaml labels.vertical_bars

COUNTS_AS_NEW_RESEARCH_TRIAL: Final[bool] = False

CPCV_PARAM_NOTE: Final[str] = (
    "CPCV N=6 k=2 unchanged from prior hardened gate; horizon=embargo=vertical_bars=10"
)


def accepted_bets_to_daily_returns(
    accept_proba: pd.Series,
    bet_returns: pd.Series,
    *,
    threshold: float = ACCEPT_PROBABILITY_THRESHOLD,
) -> pd.Series:
    """Equal-weight mean of accepted bet ``ret`` per timestamp (0.0 if none)."""
    if not accept_proba.index.equals(bet_returns.index):
        msg = "accept_proba and bet_returns must share an identical index"
        raise ValueError(msg)
    if list(accept_proba.index.names) != ["timestamp", "symbol"]:
        msg = "index names must be ('timestamp', 'symbol')"
        raise ValueError(msg)

    frame = pd.DataFrame(
        {
            "proba": accept_proba,
            "ret": bet_returns,
        }
    ).sort_index()
    daily: list[tuple[pd.Timestamp, float]] = []
    for timestamp, group in frame.groupby(level="timestamp", sort=True):
        taken = group.loc[group["proba"] >= threshold, "ret"]
        if taken.empty:
            daily.append((pd.Timestamp(timestamp), 0.0))
        else:
            daily.append((pd.Timestamp(timestamp), float(taken.mean())))

    if not daily:
        msg = "no timestamps available for meta-labeling strategy returns"
        raise ValueError(msg)
    index = pd.DatetimeIndex([ts for ts, _ in daily], name="timestamp")
    return pd.Series(
        [ret for _, ret in daily],
        index=index,
        dtype="float64",
        name="strategy_return",
    )


def cpcv_config_for_meta(vertical_bars: int = VERTICAL_BARS) -> CPCVConfig:
    """Fixed hardened CPCV knobs; embargo matches barrier holding period."""
    if vertical_bars < 1:
        msg = "vertical_bars must be >= 1"
        raise ValueError(msg)
    return CPCVConfig(
        n_blocks=N_BLOCKS,
        n_test_blocks=N_TEST_BLOCKS,
        embargo_days=vertical_bars,
        horizon=vertical_bars,
    )


def bar_timestamps_from_bars(bars: BarFrame) -> pd.DatetimeIndex:
    """Union trading calendar for CPCV label-end resolution."""
    return pd.DatetimeIndex(sorted({ts for frame in bars.bars.values() for ts in frame.index}))


def permute_labels_within_dates(
    label: pd.Series,
    *,
    rng: np.random.Generator,
) -> pd.Series:
    """Shuffle meta-labels inside each timestamp's cross-section."""
    if list(label.index.names) != ["timestamp", "symbol"]:
        msg = "label index names must be ('timestamp', 'symbol')"
        raise ValueError(msg)
    parts: list[pd.Series] = []
    for _ts, group in label.groupby(level="timestamp", sort=True):
        values = group.to_numpy(dtype=np.float64).copy()
        rng.shuffle(values)
        parts.append(pd.Series(values, index=group.index, dtype="float64"))
    out = pd.concat(parts).sort_index()
    out.name = label.name
    return out


def prepare_meta_gate_inputs(
    bars: BarFrame,
    labels: LabelSettings,
) -> tuple[BaselineDataset, pd.Series, pd.DatetimeIndex]:
    """Assemble meta dataset + aligned bet returns + bar calendar."""
    dataset, bet_returns, _sides, _t1 = assemble_meta_dataset(
        bars,
        labels,
        ma_window=PRIMARY_MA_WINDOW,
        feature_columns=META_FEATURE_COLUMNS,
    )
    if dataset.feature_columns != ALL_NEW_FEATURE_CLASS_COLUMNS:
        msg = "meta gate requires ALL_NEW_FEATURE_CLASS_COLUMNS"
        raise ValueError(msg)
    if dataset.horizon != labels.vertical_bars:
        msg = "dataset.horizon must equal labels.vertical_bars"
        raise ValueError(msg)
    return dataset, bet_returns, bar_timestamps_from_bars(bars)


def run_meta_labeling_overfitting_gate(
    dataset: BaselineDataset,
    bet_returns: pd.Series,
    *,
    cpcv_config: CPCVConfig,
    model_params: Mapping[str, Any],
    num_boost_round: int,
    n_trials: int,
    var_trial_sharpes: float,
    seed: int,
    universe: Sequence[str],
    start: date,
    end: date,
    frequency: str,
    bar_timestamps: pd.DatetimeIndex | None = None,
) -> GateVerdict:
    """CPCV + binary meta-classifier + DSR on merged accepted-bet daily returns."""
    if n_trials < 2:
        msg = "n_trials must be >= 2"
        raise ValueError(msg)
    if not np.isfinite(var_trial_sharpes) or var_trial_sharpes < 0.0:
        msg = "var_trial_sharpes must be a finite non-negative float"
        raise ValueError(msg)
    if cpcv_config.horizon != dataset.horizon:
        msg = "cpcv_config.horizon must equal dataset.horizon"
        raise ValueError(msg)
    if model_params.get("objective") != "binary":
        msg = "meta-labeling gate requires objective='binary'"
        raise ValueError(msg)

    features = dataset.features.sort_index()
    label = dataset.label.loc[features.index]
    if not bet_returns.index.equals(features.index):
        aligned_rets = bet_returns.reindex(features.index)
        if aligned_rets.isna().any():
            msg = "bet_returns missing rows for dataset features index"
            raise ValueError(msg)
    else:
        aligned_rets = bet_returns.loc[features.index]
    dataset = BaselineDataset(
        features=features,
        label=label,
        horizon=dataset.horizon,
        feature_columns=dataset.feature_columns,
    )

    split = combinatorial_purged_splits(dataset, cpcv_config, bar_timestamps=bar_timestamps)
    path_results: list[GatePathResult] = []
    path_return_series: list[pd.Series] = []

    hyperparams = dict(model_params)
    hyperparams["num_boost_round"] = num_boost_round
    compute_model_hash(
        features=list(dataset.feature_columns),
        hyperparameters=hyperparams,
        universe=universe,
        start=start,
        end=end,
        frequency=frequency,
        seed=seed,
    )
    train_params = {k: v for k, v in hyperparams.items() if k != "num_boost_round"}

    for fold in split.folds:
        train = subset_by_positions(dataset, fold.train_positions)
        test = subset_by_positions(dataset, fold.test_positions)
        test_rets = aligned_rets.loc[test.features.index]
        model: lgb.Booster = train_meta_classifier(
            train,
            params=train_params,
            num_boost_round=num_boost_round,
        )
        proba = predict_accept_proba(model, test.features)
        returns = accepted_bets_to_daily_returns(proba, test_rets)
        path_return_series.append(returns)
        ret_arr = returns.to_numpy(dtype=np.float64)
        if len(ret_arr) < 2:
            msg = (
                f"fold {fold.fold_id}: path Sharpe undefined — "
                f"need >= 2 return observations, got {len(ret_arr)}"
            )
            raise ValueError(msg)
        std = float(np.std(ret_arr, ddof=1))
        if (not np.isfinite(std)) or std <= 0.0:
            msg = (
                f"fold {fold.fold_id}: path Sharpe undefined — "
                f"zero return variance (n={len(ret_arr)})"
            )
            raise ValueError(msg)
        path_sharpe = sharpe_ratio(ret_arr).sharpe
        path_results.append(
            GatePathResult(
                fold_id=fold.fold_id,
                sharpe=float(path_sharpe),
                n_return_obs=int(len(returns)),
                mean_return=float(returns.mean()) if len(returns) else 0.0,
            )
        )

    sharpes = np.asarray([p.sharpe for p in path_results], dtype=np.float64)
    path_mean = float(np.mean(sharpes))
    path_std = float(np.std(sharpes, ddof=1)) if len(sharpes) >= 2 else 0.0

    merged = merge_cpcv_path_returns(path_return_series)
    deflated: DeflatedSharpeReport = deflated_sharpe(
        merged.to_numpy(dtype=np.float64),
        n_trials=n_trials,
        var_trial_sharpes=float(var_trial_sharpes),
    )
    if deflated.n_obs != len(merged):
        msg = f"DSR n_obs ({deflated.n_obs}) must equal merged return length ({len(merged)})"
        raise RuntimeError(msg)

    return GateVerdict(
        verdict="JA" if deflated.dsr >= DSR_THRESHOLD else "NEIN",
        dsr=deflated.dsr,
        n_trials=n_trials,
        path_sharpe_mean=path_mean,
        path_sharpe_std=path_std,
        path_results=tuple(path_results),
        deflated=deflated,
        cpcv=split,
        horizon=dataset.horizon,
        seed=seed,
    )


def run_meta_labeling_gate(
    dataset: BaselineDataset,
    bet_returns: pd.Series,
    *,
    model_params: Mapping[str, Any],
    num_boost_round: int,
    seed: int,
    universe: tuple[str, ...],
    start: date,
    end: date,
    frequency: str,
    bar_timestamps: pd.DatetimeIndex,
) -> GateVerdict:
    """Gate wrapper with research-trial variance (does not bump n_trials)."""
    return run_meta_labeling_overfitting_gate(
        dataset,
        bet_returns,
        cpcv_config=cpcv_config_for_meta(dataset.horizon),
        model_params=model_params,
        num_boost_round=num_boost_round,
        n_trials=N_TRIALS,
        var_trial_sharpes=research_trial_sharpe_variance(),
        seed=seed,
        universe=universe,
        start=start,
        end=end,
        frequency=frequency,
        bar_timestamps=bar_timestamps,
    )


def binary_params_from_settings(settings: Settings) -> dict[str, Any]:
    """Deterministic binary LGBM params from research settings + fixed seed."""
    research = settings.research
    return build_lgbm_binary_params(
        seed=SEED,
        learning_rate=research.learning_rate,
        num_leaves=research.num_leaves,
        min_data_in_leaf=research.min_data_in_leaf,
        feature_fraction=research.feature_fraction,
        bagging_fraction=research.bagging_fraction,
        bagging_freq=research.bagging_freq,
    )


# Re-export for scripts / tests.
__all__ = [
    "ACCEPT_PROBABILITY_THRESHOLD",
    "COUNTS_AS_NEW_RESEARCH_TRIAL",
    "CPCV_PARAM_NOTE",
    "DSR_THRESHOLD",
    "N_BLOCKS",
    "N_PERMUTATIONS",
    "N_TEST_BLOCKS",
    "N_TRIALS",
    "PERM_SEED",
    "SEED",
    "VERTICAL_BARS",
    "accepted_bets_to_daily_returns",
    "bar_timestamps_from_bars",
    "binary_params_from_settings",
    "cpcv_config_for_meta",
    "interpret_corrected_verdict",
    "permute_labels_within_dates",
    "prepare_meta_gate_inputs",
    "run_meta_labeling_gate",
    "run_meta_labeling_overfitting_gate",
]
