"""Phase-3 overfitting gate: CPCV OOS returns + Deflated Sharpe Ratio.

Aggregation approach (explicit)
-------------------------------
CPCV produces C(N, k) overlapping OOS paths. Path Sharpes are **diagnostic
only** — they are not treated as an i.i.d. sample for SR estimation error.

Selected approach: merge every path's daily strategy returns into **one**
series by averaging concurrent path returns on each timestamp
(``merge_cpcv_path_returns``). DSR is then computed from **one** Sharpe of
that merged series. ``T`` / ``n_obs`` is the length of that series (number of
return observations), never the fold count C(N, k).

``var_trial_sharpes`` / SR0
--------------------------
Must be supplied by the caller from the variance of **independent research
configurations** actually tested (see ``research_trials``). It must **not**
be estimated from CPCV path-Sharpe dispersion.

Verdict rule (hard): ``JA`` iff ``dsr > 0``, else ``NEIN``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from aihedgefund.core.schemas import (
    BaselineDataset,
    CPCVConfig,
    DeflatedSharpeReport,
    GatePathResult,
    GateVerdict,
)
from aihedgefund.research.baseline import predict_scores, train_baseline
from aihedgefund.research.cpcv import combinatorial_purged_splits, subset_by_positions
from aihedgefund.research.deflated_sharpe import deflated_sharpe, sharpe_ratio
from aihedgefund.research.model_hash import compute_model_hash


def scores_to_strategy_returns(
    scores: pd.Series,
    forward_returns: pd.Series,
) -> pd.Series:
    """Dollar-neutral CS portfolio return per timestamp from model scores.

    Weights are demeaned scores re-scaled so ``sum(|w|) == 1``. Dates with
    fewer than two symbols or zero weight dispersion yield 0.0 (explicit, not
    dropped) so path lengths stay aligned with the OOS calendar.
    """
    if not scores.index.equals(forward_returns.index):
        msg = "scores and forward_returns must share an identical index"
        raise ValueError(msg)
    if list(scores.index.names) != ["timestamp", "symbol"]:
        msg = "index names must be ('timestamp', 'symbol')"
        raise ValueError(msg)

    frame = pd.DataFrame({"score": scores, "fwd": forward_returns}).sort_index()
    daily: list[tuple[pd.Timestamp, float]] = []
    for timestamp, group in frame.groupby(level="timestamp", sort=True):
        if len(group) < 2:
            daily.append((pd.Timestamp(timestamp), 0.0))
            continue
        demeaned = group["score"] - group["score"].mean()
        abs_sum = float(demeaned.abs().sum())
        if abs_sum == 0.0:
            daily.append((pd.Timestamp(timestamp), 0.0))
            continue
        weights = demeaned / abs_sum
        port = float((weights * group["fwd"]).sum())
        daily.append((pd.Timestamp(timestamp), port))

    if not daily:
        msg = "no timestamps available for strategy returns"
        raise ValueError(msg)
    index = pd.DatetimeIndex([ts for ts, _ in daily], name="timestamp")
    return pd.Series(
        [ret for _, ret in daily],
        index=index,
        dtype="float64",
        name="strategy_return",
    )


def merge_cpcv_path_returns(path_return_series: Sequence[pd.Series]) -> pd.Series:
    """Merge overlapping CPCV OOS path returns into one daily series.

    Concurrent path returns on the same timestamp are averaged. The resulting
    length is ``T`` for DSR (return observations of the evaluated series).
    Hard-fails on empty input.
    """
    if not path_return_series:
        msg = "path_return_series must be non-empty"
        raise ValueError(msg)
    concatenated = pd.concat(list(path_return_series)).sort_index()
    merged = concatenated.groupby(level=0).mean()
    merged.name = "strategy_return"
    if len(merged) < 2:
        msg = (
            f"merged OOS return series needs >= 2 observations for Sharpe/DSR; "
            f"got {len(merged)}"
        )
        raise ValueError(msg)
    return merged.astype("float64")


def run_overfitting_gate(
    dataset: BaselineDataset,
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
    """Train per CPCV fold, merge OOS returns, and emit a DSR gate verdict.

    ``n_trials``
        Number of independent research configurations explored (not CPCV folds).
    ``var_trial_sharpes``
        Sample variance of those research-trial Sharpes (same non-annualized
        scale as the observed Sharpe). Must not be derived from CPCV path
        Sharpes — see ``research.research_trials``.
    ``bar_timestamps``
        Forwarded to CPCV for label-end resolution (full trading calendar).
    """
    if n_trials < 2:
        msg = "n_trials must be >= 2"
        raise ValueError(msg)
    if not np.isfinite(var_trial_sharpes) or var_trial_sharpes < 0.0:
        msg = "var_trial_sharpes must be a finite non-negative float"
        raise ValueError(msg)
    if cpcv_config.horizon != dataset.horizon:
        msg = "cpcv_config.horizon must equal dataset.horizon"
        raise ValueError(msg)

    features = dataset.features.sort_index()
    dataset = BaselineDataset(
        features=features,
        label=dataset.label.loc[features.index],
        horizon=dataset.horizon,
        feature_columns=dataset.feature_columns,
    )

    split = combinatorial_purged_splits(
        dataset, cpcv_config, bar_timestamps=bar_timestamps
    )
    path_results: list[GatePathResult] = []
    path_return_series: list[pd.Series] = []

    hyperparams = dict(model_params)
    hyperparams["num_boost_round"] = num_boost_round
    model_hash = compute_model_hash(
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
        model = train_baseline(train, params=train_params, num_boost_round=num_boost_round)
        predictions = predict_scores(model, test.features, model_hash=model_hash)
        returns = scores_to_strategy_returns(predictions.scores, test.label)
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

    # One Sharpe on the merged OOS series; T = len(merged), not C(N, k).
    merged = merge_cpcv_path_returns(path_return_series)
    deflated: DeflatedSharpeReport = deflated_sharpe(
        merged.to_numpy(dtype=np.float64),
        n_trials=n_trials,
        var_trial_sharpes=float(var_trial_sharpes),
    )
    if deflated.n_obs != len(merged):
        msg = (
            f"DSR n_obs ({deflated.n_obs}) must equal merged return length "
            f"({len(merged)}); T is the evaluated return series, not fold count"
        )
        raise RuntimeError(msg)

    return GateVerdict(
        verdict="JA" if deflated.dsr > 0.0 else "NEIN",
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
