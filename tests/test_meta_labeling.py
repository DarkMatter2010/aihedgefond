"""Offline tests for meta-labeling triage (no yfinance)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from aihedgefund.core.config import load_settings
from aihedgefund.core.schemas import BarFrame
from aihedgefund.features.feature_classes import ALL_NEW_FEATURE_CLASS_COLUMNS
from aihedgefund.labels.labeling import triple_barrier
from aihedgefund.research.meta_labeling import (
    ACCEPT_PROBABILITY_THRESHOLD,
    META_FEATURE_COLUMNS,
    PRIMARY_MA_WINDOW,
    assemble_meta_dataset,
    bet_strategy_sharpe,
    build_meta_label_events,
    classification_report_vs_baserate,
    interpret_meta_labeling,
    primary_trend_side,
    run_meta_labeling_triage,
)
from aihedgefund.research.universes import BROAD_LIQUID_CANDIDATE_UNIVERSE

SEED = 20260720


def _synthetic_symbol_frame(symbol_seed: int, rows: int = 280) -> pd.DataFrame:
    rng = np.random.default_rng(symbol_seed)
    index = pd.date_range("2021-01-04", periods=rows, freq="B", tz="UTC")
    returns = rng.normal(0.0004, 0.012, rows)
    close = 100.0 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0] * 0.999, close[:-1]]
    high = np.maximum(open_, close) * (1.001 + rng.uniform(0.0, 0.002, rows))
    low = np.minimum(open_, close) * (0.999 - rng.uniform(0.0, 0.002, rows))
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": rng.integers(1_000_000, 5_000_000, rows).astype(float),
        },
        index=index,
    )


def _bars(n: int = 8, rows: int = 280) -> BarFrame:
    symbols = BROAD_LIQUID_CANDIDATE_UNIVERSE[:n]
    frames = {
        symbol: _synthetic_symbol_frame(SEED + idx, rows=rows)
        for idx, symbol in enumerate(symbols)
    }
    empty_ca = {
        symbol: pd.Series(0.0, index=frames[symbol].index, dtype="float64") for symbol in symbols
    }
    return BarFrame(bars=frames, dividends=empty_ca, splits=empty_ca)


def test_meta_constants() -> None:
    assert PRIMARY_MA_WINDOW == 10
    assert ACCEPT_PROBABILITY_THRESHOLD == 0.5
    assert META_FEATURE_COLUMNS == ALL_NEW_FEATURE_CLASS_COLUMNS
    assert len(META_FEATURE_COLUMNS) == 15


def test_primary_trend_side_causal() -> None:
    index = pd.date_range("2025-01-01", periods=15, freq="D", tz="UTC")
    # Flat then step up so SMA lags — side must stay NaN for first 9 bars.
    close = pd.Series([100.0] * 10 + [110.0] * 5, index=index)
    side = primary_trend_side(close, window=10)
    assert side.iloc[:9].isna().all()
    assert float(side.iloc[9]) == 0.0 or pd.isna(side.iloc[9]) or float(side.iloc[9]) in (
        -1.0,
        1.0,
    )
    # After the jump, close > SMA → long.
    assert float(side.iloc[-1]) == 1.0


def test_dense_numpy_matches_triple_barrier() -> None:
    """NumPy dense path must match Phase-1 ``triple_barrier`` meta-labels."""
    from aihedgefund.labels.labeling import daily_volatility
    from aihedgefund.research.meta_labeling import _dense_meta_barrier_numpy

    rng = np.random.default_rng(7)
    index = pd.date_range("2024-01-01", periods=80, freq="B", tz="UTC")
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 80))), index=index)
    side = primary_trend_side(close, window=10)
    vol = daily_volatility(close, span=20)
    positions, labels, t1_pos, rets = _dense_meta_barrier_numpy(
        close.to_numpy(dtype=np.float64),
        side.to_numpy(dtype=np.float64),
        vol.to_numpy(dtype=np.float64),
        pt=1.0,
        sl=1.0,
        vertical_bars=10,
    )
    assert len(positions) > 10
    events = [index[int(p)] for p in positions]
    ref = triple_barrier(
        close,
        events,
        pt=1.0,
        sl=1.0,
        vertical_bars=10,
        side=side,
        vol=vol,
    )
    for i, t0 in enumerate(events):
        row = ref.loc[t0]
        assert int(row["label"]) == int(labels[i])
        assert pd.Timestamp(row["t1"]) == index[int(t1_pos[i])]
        assert float(row["ret"]) == pytest.approx(float(rets[i]), rel=1e-12, abs=1e-12)


def test_triple_barrier_meta_label_no_lookahead() -> None:
    """Barrier touch uses only path after t0; known win/loss paths."""
    index = pd.date_range("2025-01-01", periods=6, freq="D", tz="UTC")
    # Long bet: hits +5% on day 1 (before later crash) → win label 1.
    close_win = pd.Series([100.0, 106.0, 90.0, 90.0, 90.0, 90.0], index=index)
    vol = pd.Series(0.05, index=index)
    side = pd.Series(1.0, index=index)
    win = triple_barrier(
        close_win,
        [index[0]],
        pt=1.0,
        sl=1.0,
        vertical_bars=4,
        side=side,
        vol=vol,
    )
    assert int(win.iloc[0]["label"]) == 1
    assert pd.Timestamp(win.iloc[0]["t1"]) == index[1]
    # Path must not use bars after t1 for the decision (t1 is first touch).
    assert pd.Timestamp(win.iloc[0]["t1"]) < index[2]

    # Long bet: drops first → loss label 0.
    close_loss = pd.Series([100.0, 94.0, 110.0, 110.0, 110.0, 110.0], index=index)
    loss = triple_barrier(
        close_loss,
        [index[0]],
        pt=1.0,
        sl=1.0,
        vertical_bars=4,
        side=side,
        vol=vol,
    )
    assert int(loss.iloc[0]["label"]) == 0
    assert pd.Timestamp(loss.iloc[0]["t1"]) == index[1]


def test_classification_report_hand_computed() -> None:
    y_true = pd.Series([1, 1, 0, 0, 1, 0])
    y_pred = pd.Series([1, 0, 0, 1, 1, 0])
    # TP=2 (idx 0,4), FP=1 (idx 3), FN=1 (idx 1), TN=2
    report = classification_report_vs_baserate(y_true, y_pred)
    assert report.n_true_positive == 2
    assert report.n_false_positive == 1
    assert report.n_false_negative == 1
    assert report.n_true_negative == 2
    assert report.precision == pytest.approx(2 / 3)
    assert report.recall == pytest.approx(2 / 3)
    assert report.f1 == pytest.approx(2 / 3)
    assert report.base_rate == pytest.approx(0.5)
    assert report.lift == pytest.approx((2 / 3) / 0.5)


def test_bet_strategy_sharpe_matches_helper() -> None:
    rets = pd.Series([0.01, -0.005, 0.02, 0.0, -0.01])
    assert bet_strategy_sharpe(rets) == pytest.approx(
        float(rets.mean() / rets.std(ddof=1))
    )


def test_interpret_rules() -> None:
    interp, _ = interpret_meta_labeling(
        precision=0.60,
        base_rate=0.50,
        primary_sharpe=0.1,
        filtered_sharpe=0.3,
    )
    assert interp == "candidate_for_gate"
    stop, _ = interpret_meta_labeling(
        precision=0.51,
        base_rate=0.50,
        primary_sharpe=0.2,
        filtered_sharpe=0.1,
    )
    assert stop == "search_stop"


def test_assemble_and_triage_deterministic_offline() -> None:
    bars = _bars(8, rows=320)
    settings = load_settings()
    settings = settings.model_copy(
        update={
            "universe": tuple(sorted(bars.bars)),
            "research": settings.research.model_copy(
                update={
                    "seed": 42,
                    "train_end": date(2021, 10, 29),
                    "test_start": date(2021, 11, 15),
                    "embargo_days": 10,
                    "num_boost_round": 20,
                    "num_leaves": 8,
                    "min_data_in_leaf": 5,
                }
            ),
        }
    )
    events = build_meta_label_events(bars, settings.labels)
    assert set(events["label"].unique()).issubset({0, 1})
    assert (events["t1"] > events.index.get_level_values("timestamp")).all() or (
        events["t1"] >= events.index.get_level_values("timestamp")
    ).all()

    dataset, rets, sides, t1 = assemble_meta_dataset(bars, settings.labels)
    assert dataset.feature_columns == ALL_NEW_FEATURE_CLASS_COLUMNS
    assert dataset.horizon == settings.labels.vertical_bars
    assert rets.index.equals(dataset.features.index)
    assert sides.index.equals(dataset.features.index)
    assert t1.index.equals(dataset.features.index)

    first = run_meta_labeling_triage(bars, settings)
    second = run_meta_labeling_triage(bars, settings)
    assert first.classification.precision == second.classification.precision
    assert first.classification.recall == second.classification.recall
    assert first.classification.f1 == second.classification.f1
    assert first.primary_sharpe_oos == second.primary_sharpe_oos
    assert first.filtered_sharpe_oos == second.filtered_sharpe_oos
    assert first.n_train == second.n_train
    assert first.n_test == second.n_test
    assert first.interpretation in ("candidate_for_gate", "search_stop")
    assert first.counts_as_research_trial is True
