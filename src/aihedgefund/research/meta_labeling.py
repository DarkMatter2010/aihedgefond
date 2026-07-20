"""Phase-2 meta-labeling triage: SMA primary + Triple-Barrier + LGBM accept/reject.

Last free methodologically distinct probe after 23 return-regression trials.
Primary direction is a fixed SMA-10 sign rule; the meta-model only decides
whether to take the bet. Evaluation uses Precision/Recall/F1 vs base rate and
per-bet OOS Sharpe (filtered vs unfiltered) — not Rank-IC.

Does **not** run CPCV/DSR. Counts as one new research trial (row 24).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any, Final, Literal

import lightgbm as lgb
import numpy as np
import pandas as pd
from pydantic import Field

from aihedgefund.core.config import LabelSettings, Settings
from aihedgefund.core.schemas import (
    BarFrame,
    BaselineDataset,
    BoundaryDTO,
    FiniteFloat,
    NonEmptyText,
)
from aihedgefund.features.feature_classes import (
    ALL_NEW_FEATURE_CLASS_COLUMNS,
    build_triage_feature_matrix,
)
from aihedgefund.features.pipeline import _adjusted_feature_frame
from aihedgefund.labels.labeling import daily_volatility
from aihedgefund.research.deflated_sharpe import sharpe_ratio
from aihedgefund.research.feature_class_triage import (
    _test_start_for_dataset,
    test_start_for_horizon,
)
from aihedgefund.research.split import time_embargo_split
from aihedgefund.research.universes import SURVIVORSHIP_BIAS_NOTE

PRIMARY_MA_WINDOW: Final[int] = 10
ACCEPT_PROBABILITY_THRESHOLD: Final[float] = 0.5
META_FEATURE_COLUMNS: Final[tuple[str, ...]] = ALL_NEW_FEATURE_CLASS_COLUMNS

# Precision must clear base rate by this absolute margin to count as lift.
PRECISION_LIFT_MARGIN: Final[float] = 0.02

Interpretation = Literal["candidate_for_gate", "search_stop"]


class ClassificationVsBaserate(BoundaryDTO):
    """OOS classification metrics versus the primary win base rate."""

    precision: FiniteFloat
    recall: FiniteFloat
    f1: FiniteFloat
    base_rate: FiniteFloat
    lift: FiniteFloat
    n_true_positive: int = Field(ge=0)
    n_false_positive: int = Field(ge=0)
    n_false_negative: int = Field(ge=0)
    n_true_negative: int = Field(ge=0)


class MetaLabelingReport(BoundaryDTO):
    """Single-run meta-labeling triage result (no Gate)."""

    n_symbols_requested: int = Field(ge=1)
    n_symbols: int = Field(ge=1)
    seed: int = Field(ge=0)
    train_end: date
    configured_test_start: date
    test_start: date
    primary_ma_window: int = Field(ge=2)
    vertical_bars: int = Field(ge=1)
    feature_column_count: int = Field(ge=1)
    n_train: int = Field(ge=0)
    n_test: int = Field(ge=0)
    n_accepted: int = Field(ge=0)
    accept_rate: FiniteFloat
    classification: ClassificationVsBaserate
    primary_sharpe_oos: FiniteFloat
    filtered_sharpe_oos: FiniteFloat | None
    precision_above_baserate: bool
    filtered_sharpe_beats_primary: bool
    interpretation: Interpretation
    interpretation_note: NonEmptyText
    survivorship_bias_note: NonEmptyText
    counts_as_research_trial: bool = True
    trial_sharpe_logged: FiniteFloat | None


def primary_trend_side(close: pd.Series, window: int = PRIMARY_MA_WINDOW) -> pd.Series:
    """Causal SMA sign: +1 above SMA, -1 below, NaN during warmup or equality."""
    if window < 2:
        msg = "window must be at least two"
        raise ValueError(msg)
    values = close.astype(float)
    sma = values.rolling(window, min_periods=window).mean()
    side = pd.Series(np.nan, index=close.index, dtype="float64", name="primary_side")
    side = side.mask(values > sma, 1.0)
    side = side.mask(values < sma, -1.0)
    return side


def _dense_meta_barrier_numpy(
    close: np.ndarray,
    side: np.ndarray,
    vol: np.ndarray,
    *,
    pt: float,
    sl: float,
    vertical_bars: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """NumPy twin of ``triple_barrier(..., side=...)`` for dense primary events.

    Returns ``(positions, labels, t1_positions, realized_returns)`` for events
    with a full vertical window and positive vol. Path starts at ``t0+1``.
    Meta-label is ``int(realized_return > 0)`` at the first barrier touch.
    """
    n = int(close.shape[0])
    positions: list[int] = []
    labels: list[int] = []
    t1_positions: list[int] = []
    rets: list[float] = []
    for i in range(n - vertical_bars):
        event_side = float(side[i])
        if event_side not in (-1.0, 1.0):
            continue
        target = float(vol[i])
        if not np.isfinite(target) or target <= 0.0:
            continue
        entry = float(close[i])
        upper = pt * target
        lower = -sl * target
        hit_pos = i + vertical_bars
        for j in range(i + 1, i + vertical_bars + 1):
            signed_ret = (float(close[j]) / entry - 1.0) * event_side
            if signed_ret >= upper or signed_ret <= lower:
                hit_pos = j
                break
        realized_final = (float(close[hit_pos]) / entry - 1.0) * event_side
        positions.append(i)
        labels.append(int(realized_final > 0.0))
        t1_positions.append(hit_pos)
        rets.append(realized_final)
    if not positions:
        empty_i = np.asarray([], dtype=np.int64)
        empty_f = np.asarray([], dtype=np.float64)
        return empty_i, empty_i, empty_i, empty_f
    return (
        np.asarray(positions, dtype=np.int64),
        np.asarray(labels, dtype=np.int64),
        np.asarray(t1_positions, dtype=np.int64),
        np.asarray(rets, dtype=np.float64),
    )


def build_meta_label_events(
    bars: BarFrame,
    labels: LabelSettings,
    *,
    ma_window: int = PRIMARY_MA_WINDOW,
) -> pd.DataFrame:
    """Per-symbol SMA sides → Triple-Barrier meta-labels (bet wins = 1).

    Only events with a full vertical window (``vertical_bars`` forward bars)
    and positive volatility at ``t0`` are retained. Path inspection starts at
    ``t0+1`` (no look-ahead into the entry bar). Uses a NumPy dense path that
    matches ``triple_barrier(..., side=...)`` semantics (see unit tests).
    """
    timestamps_out: list[pd.Timestamp] = []
    symbols_out: list[str] = []
    sides_out: list[float] = []
    labels_out: list[int] = []
    t1_out: list[pd.Timestamp] = []
    rets_out: list[float] = []

    symbol_items = list(bars.bars.items())
    total = len(symbol_items)
    for idx, (symbol, frame) in enumerate(symbol_items, start=1):
        if idx == 1 or idx % 50 == 0 or idx == total:
            print(f"meta-label events: {idx}/{total} symbols", flush=True)
        adjusted = _adjusted_feature_frame(
            frame,
            bars.splits[symbol],
            bars.dividends[symbol],
        )
        close = adjusted["close"].astype(float)
        if not isinstance(close.index, pd.DatetimeIndex):
            msg = f"{symbol}: close must use a DatetimeIndex"
            raise ValueError(msg)
        side = primary_trend_side(close, ma_window)
        vol = daily_volatility(close, labels.vol_span)
        close_arr = close.to_numpy(dtype=np.float64, copy=False)
        side_arr = side.to_numpy(dtype=np.float64, copy=False)
        vol_arr = vol.to_numpy(dtype=np.float64, copy=False)
        positions, meta_labels, t1_pos, rets = _dense_meta_barrier_numpy(
            close_arr,
            side_arr,
            vol_arr,
            pt=labels.pt,
            sl=labels.sl,
            vertical_bars=labels.vertical_bars,
        )
        index = close.index
        for pos, lab, t1p, ret in zip(positions, meta_labels, t1_pos, rets, strict=True):
            timestamps_out.append(pd.Timestamp(index[int(pos)]))
            symbols_out.append(symbol)
            sides_out.append(float(side_arr[int(pos)]))
            labels_out.append(int(lab))
            t1_out.append(pd.Timestamp(index[int(t1p)]))
            rets_out.append(float(ret))

    if not timestamps_out:
        msg = "meta-labeling produced no events"
        raise ValueError(msg)
    events = pd.DataFrame(
        {
            "side": sides_out,
            "label": labels_out,
            "t1": t1_out,
            "ret": rets_out,
        },
        index=pd.MultiIndex.from_arrays(
            [timestamps_out, symbols_out],
            names=("timestamp", "symbol"),
        ),
    ).sort_index()
    if not events.index.is_unique:
        msg = "duplicate meta-label events"
        raise ValueError(msg)
    if not set(events["label"].unique()).issubset({0, 1}):
        msg = "meta labels must be binary {0, 1} when side is set"
        raise ValueError(msg)
    return events


def assemble_meta_dataset(
    bars: BarFrame,
    labels: LabelSettings,
    *,
    ma_window: int = PRIMARY_MA_WINDOW,
    feature_columns: tuple[str, ...] = META_FEATURE_COLUMNS,
) -> tuple[BaselineDataset, pd.Series, pd.Series, pd.Series]:
    """Join ALL_NEW features at ``t0`` with meta-labels; drop incomplete rows.

    Returns ``(dataset, bet_returns, sides, t1)`` all sharing the dataset index.
    ``dataset.label`` is the binary meta-label (bet wins). ``dataset.horizon``
    is ``vertical_bars`` for embargo / purge semantics.
    """
    events = build_meta_label_events(bars, labels, ma_window=ma_window)
    full_matrix = build_triage_feature_matrix(bars)
    missing = [c for c in feature_columns if c not in full_matrix.columns]
    if missing:
        msg = f"feature matrix missing columns: {missing}"
        raise ValueError(msg)
    features = full_matrix.loc[:, list(feature_columns)].astype(
        {column: "float64" for column in feature_columns}
    )
    common = features.index.intersection(events.index)
    if common.empty:
        msg = "features and meta-label events have no overlapping rows"
        raise ValueError(msg)
    features = features.loc[common].sort_index()
    events = events.loc[common].sort_index()
    complete = features.notna().all(axis=1) & events["label"].notna() & events["ret"].notna()
    features = features.loc[complete]
    events = events.loc[complete]
    if features.empty:
        msg = "meta dataset empty after dropping NaN feature/label rows"
        raise ValueError(msg)

    label = events["label"].astype("float64").rename("meta_label")
    bet_returns = events["ret"].astype("float64").rename("bet_return")
    sides = events["side"].astype("float64").rename("side")
    t1 = events["t1"].rename("t1")
    dataset = BaselineDataset(
        features=features,
        label=label,
        horizon=labels.vertical_bars,
        feature_columns=feature_columns,
    )
    return dataset, bet_returns, sides, t1


def build_lgbm_binary_params(
    *,
    seed: int,
    learning_rate: float,
    num_leaves: int,
    min_data_in_leaf: int,
    feature_fraction: float,
    bagging_fraction: float,
    bagging_freq: int,
) -> dict[str, Any]:
    """Deterministic LightGBM binary-classifier parameters for meta-labeling."""
    return {
        "objective": "binary",
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


def train_meta_classifier(
    train: BaselineDataset,
    *,
    params: Mapping[str, Any],
    num_boost_round: int,
) -> lgb.Booster:
    """Fit a deterministic LightGBM binary classifier on meta-labels."""
    if num_boost_round < 1:
        msg = "num_boost_round must be >= 1"
        raise ValueError(msg)
    if params.get("objective") != "binary":
        msg = "meta-labeling requires objective='binary'"
        raise ValueError(msg)
    if params.get("deterministic") is not True:
        msg = "meta-labeling requires deterministic=True"
        raise ValueError(msg)

    feature_names = list(train.feature_columns)
    dataset = lgb.Dataset(
        train.features.loc[:, feature_names],
        label=train.label.to_numpy(dtype=float),
        feature_name=feature_names,
        free_raw_data=False,
    )
    return lgb.train(dict(params), dataset, num_boost_round=num_boost_round)


def predict_accept_proba(model: lgb.Booster, features: pd.DataFrame) -> pd.Series:
    """Return P(bet wins) for each row; used with ``ACCEPT_PROBABILITY_THRESHOLD``."""
    feature_names = list(model.feature_name())
    missing = [name for name in feature_names if name not in features.columns]
    if missing:
        msg = f"prediction features missing columns: {missing}"
        raise ValueError(msg)
    matrix = features.loc[:, feature_names]
    if matrix.isna().any().any():
        msg = "prediction features must not contain NaNs"
        raise ValueError(msg)
    raw = model.predict(matrix)
    return pd.Series(raw, index=features.index, name="accept_proba", dtype="float64")


def classification_report_vs_baserate(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    *,
    base_rate: float | None = None,
) -> ClassificationVsBaserate:
    """Precision / Recall / F1 vs the empirical win base rate of ``y_true``."""
    true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if true.shape != pred.shape:
        msg = "y_true and y_pred must have the same shape"
        raise ValueError(msg)
    if true.size == 0:
        msg = "classification_report_vs_baserate requires at least one observation"
        raise ValueError(msg)
    if not set(np.unique(true)).issubset({0, 1}) or not set(np.unique(pred)).issubset({0, 1}):
        msg = "y_true and y_pred must be binary {0, 1}"
        raise ValueError(msg)

    tp = int(((pred == 1) & (true == 1)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    tn = int(((pred == 0) & (true == 0)).sum())
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (
        float(2.0 * precision * recall / (precision + recall))
        if (precision + recall) > 0.0
        else 0.0
    )
    rate = float(true.mean()) if base_rate is None else float(base_rate)
    if not np.isfinite(rate) or rate < 0.0 or rate > 1.0:
        msg = "base_rate must be a finite float in [0, 1]"
        raise ValueError(msg)
    lift = float(precision / rate) if rate > 0.0 else 0.0
    return ClassificationVsBaserate(
        precision=precision,
        recall=recall,
        f1=f1,
        base_rate=rate,
        lift=lift,
        n_true_positive=tp,
        n_false_positive=fp,
        n_false_negative=fn,
        n_true_negative=tn,
    )


def bet_strategy_sharpe(rets: pd.Series) -> float:
    """Non-annualized Sharpe of per-bet realized returns (``triple_barrier`` ret)."""
    arr = rets.to_numpy(dtype=np.float64).reshape(-1)
    return float(sharpe_ratio(arr).sharpe)


def _subset_dataset(dataset: BaselineDataset, mask: np.ndarray) -> BaselineDataset:
    """Boolean-mask a BaselineDataset (same contract as ``time_embargo_split``)."""
    features = dataset.features.iloc[mask]
    label = dataset.label.iloc[mask]
    return BaselineDataset(
        features=features,
        label=label,
        horizon=dataset.horizon,
        feature_columns=dataset.feature_columns,
    )


def _purge_train_by_barrier_end(
    train: BaselineDataset,
    t1: pd.Series,
    *,
    train_end: date,
) -> BaselineDataset:
    """Drop train events whose barrier path ends after ``train_end`` (no leak)."""
    train_t1 = t1.loc[train.features.index]
    keep = np.array(
        [pd.Timestamp(end).date() <= train_end for end in train_t1],
        dtype=bool,
    )
    if not keep.any():
        msg = "train split empty after barrier-end purge"
        raise ValueError(msg)
    return _subset_dataset(train, keep)


def interpret_meta_labeling(
    *,
    precision: float,
    base_rate: float,
    primary_sharpe: float,
    filtered_sharpe: float | None,
) -> tuple[Interpretation, str]:
    """Candidate only if Precision clears base rate and filtered Sharpe beats primary."""
    precision_ok = precision > base_rate + PRECISION_LIFT_MARGIN
    sharpe_ok = filtered_sharpe is not None and filtered_sharpe > primary_sharpe
    if precision_ok and sharpe_ok:
        return (
            "candidate_for_gate",
            "Precision above base rate and filtered OOS bet Sharpe beats "
            "unfiltered primary — next step is CPCV/DSR gate (separate prompt).",
        )
    return (
        "search_stop",
        "No material Precision lift and/or no filtered Sharpe gain vs primary. "
        "Recommend ending free-yfinance OHLCV signal search on this universe; "
        "treat Phase 0–3 as infrastructure/learning deliverable.",
    )


def run_meta_labeling_triage(bars: BarFrame, settings: Settings) -> MetaLabelingReport:
    """Train meta-classifier OOS; report classification + filtered bet Sharpe."""
    research = settings.research
    label_cfg = settings.labels
    horizon = label_cfg.vertical_bars
    embargo_days = max(research.embargo_days, horizon)
    configured_test_start = research.test_start
    train_end = research.train_end

    dataset, bet_returns, _sides, t1 = assemble_meta_dataset(
        bars,
        label_cfg,
        ma_window=PRIMARY_MA_WINDOW,
        feature_columns=META_FEATURE_COLUMNS,
    )
    test_start = _test_start_for_dataset(
        dataset,
        train_end=train_end,
        configured_test_start=configured_test_start,
        horizon=horizon,
    )
    # Calendar gap must also clear embargo_days (== max(research.embargo, vertical)).
    calendar_min = test_start_for_horizon(train_end, configured_test_start, embargo_days)
    if calendar_min > test_start:
        test_start = calendar_min

    train, test, _split = time_embargo_split(
        dataset,
        train_end=train_end,
        test_start=test_start,
        embargo_days=embargo_days,
        horizon=horizon,
    )
    train = _purge_train_by_barrier_end(train, t1, train_end=train_end)

    params = build_lgbm_binary_params(
        seed=research.seed,
        learning_rate=research.learning_rate,
        num_leaves=research.num_leaves,
        min_data_in_leaf=research.min_data_in_leaf,
        feature_fraction=research.feature_fraction,
        bagging_fraction=research.bagging_fraction,
        bagging_freq=research.bagging_freq,
    )
    model = train_meta_classifier(
        train,
        params=params,
        num_boost_round=research.num_boost_round,
    )
    proba = predict_accept_proba(model, test.features)
    accepted = proba >= ACCEPT_PROBABILITY_THRESHOLD
    y_pred = accepted.astype(int)
    y_true = test.label.astype(int)

    classification = classification_report_vs_baserate(y_true, y_pred)
    test_rets = bet_returns.loc[test.features.index]
    primary_sharpe = bet_strategy_sharpe(test_rets)
    filtered_rets = test_rets.loc[accepted.to_numpy()]
    n_accepted = int(accepted.sum())
    filtered_sharpe: float | None
    if n_accepted >= 2 and float(filtered_rets.std(ddof=1)) > 0.0:
        filtered_sharpe = bet_strategy_sharpe(filtered_rets)
    else:
        filtered_sharpe = None

    interpretation, note = interpret_meta_labeling(
        precision=classification.precision,
        base_rate=classification.base_rate,
        primary_sharpe=primary_sharpe,
        filtered_sharpe=filtered_sharpe,
    )
    precision_above = classification.precision > classification.base_rate + PRECISION_LIFT_MARGIN
    sharpe_beats = filtered_sharpe is not None and filtered_sharpe > primary_sharpe
    trial_sharpe = filtered_sharpe if filtered_sharpe is not None else primary_sharpe

    return MetaLabelingReport(
        n_symbols_requested=len(settings.universe),
        n_symbols=len(bars.bars),
        seed=research.seed,
        train_end=train_end,
        configured_test_start=configured_test_start,
        test_start=test_start,
        primary_ma_window=PRIMARY_MA_WINDOW,
        vertical_bars=horizon,
        feature_column_count=len(META_FEATURE_COLUMNS),
        n_train=len(train.features),
        n_test=len(test.features),
        n_accepted=n_accepted,
        accept_rate=float(n_accepted / len(test.features)) if len(test.features) else 0.0,
        classification=classification,
        primary_sharpe_oos=primary_sharpe,
        filtered_sharpe_oos=filtered_sharpe,
        precision_above_baserate=precision_above,
        filtered_sharpe_beats_primary=sharpe_beats,
        interpretation=interpretation,
        interpretation_note=note,
        survivorship_bias_note=SURVIVORSHIP_BIAS_NOTE,
        counts_as_research_trial=True,
        trial_sharpe_logged=trial_sharpe,
    )
