"""Offline tests for meta-labeling CPCV/DSR gate (no yfinance)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aihedgefund.core.config import load_settings
from aihedgefund.core.schemas import BarFrame
from aihedgefund.features.feature_classes import ALL_NEW_FEATURE_CLASS_COLUMNS
from aihedgefund.research.meta_labeling_gate import (
    COUNTS_AS_NEW_RESEARCH_TRIAL,
    DSR_THRESHOLD,
    N_BLOCKS,
    N_PERMUTATIONS,
    N_TEST_BLOCKS,
    N_TRIALS,
    SEED,
    VERTICAL_BARS,
    accepted_bets_to_daily_returns,
    bar_timestamps_from_bars,
    binary_params_from_settings,
    cpcv_config_for_meta,
    interpret_corrected_verdict,
    permute_labels_within_dates,
    prepare_meta_gate_inputs,
    run_meta_labeling_gate,
)
from aihedgefund.research.research_trials import N_RESEARCH_TRIALS
from aihedgefund.research.universes import BROAD_LIQUID_CANDIDATE_UNIVERSE

FIXTURE_SEED = 20260721


def _synthetic_symbol_frame(symbol_seed: int, rows: int = 220) -> pd.DataFrame:
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


def _bars(n: int = 8, rows: int = 220) -> BarFrame:
    symbols = BROAD_LIQUID_CANDIDATE_UNIVERSE[:n]
    frames = {
        symbol: _synthetic_symbol_frame(FIXTURE_SEED + idx, rows=rows)
        for idx, symbol in enumerate(symbols)
    }
    empty_ca = {
        symbol: pd.Series(0.0, index=frames[symbol].index, dtype="float64") for symbol in symbols
    }
    return BarFrame(bars=frames, dividends=empty_ca, splits=empty_ca)


def test_meta_gate_constants() -> None:
    assert N_TRIALS == N_RESEARCH_TRIALS == 25
    assert N_BLOCKS == 6 and N_TEST_BLOCKS == 2
    assert N_PERMUTATIONS == 100
    assert DSR_THRESHOLD == 0.95
    assert SEED == 42
    assert VERTICAL_BARS == 10
    assert COUNTS_AS_NEW_RESEARCH_TRIAL is False
    cfg = cpcv_config_for_meta()
    assert cfg.horizon == cfg.embargo_days == 10


def test_accepted_bets_to_daily_returns() -> None:
    idx = pd.MultiIndex.from_product(
        [
            pd.date_range("2024-01-02", periods=2, freq="B", tz="UTC"),
            ["A", "B"],
        ],
        names=("timestamp", "symbol"),
    )
    proba = pd.Series([0.9, 0.1, 0.6, 0.7], index=idx, dtype="float64")
    rets = pd.Series([0.02, -0.01, 0.03, -0.04], index=idx, dtype="float64")
    daily = accepted_bets_to_daily_returns(proba, rets, threshold=0.5)
    # Day0: only A accepted → 0.02; Day1: both → mean(0.03, -0.04)
    assert daily.iloc[0] == pytest.approx(0.02)
    assert daily.iloc[1] == pytest.approx(-0.005)


def test_prepare_and_gate_deterministic_offline() -> None:
    bars = _bars(8, rows=260)
    settings = load_settings()
    dataset, bet_returns, calendar = prepare_meta_gate_inputs(bars, settings.labels)
    assert dataset.feature_columns == ALL_NEW_FEATURE_CLASS_COLUMNS
    assert dataset.horizon == settings.labels.vertical_bars
    assert bet_returns.index.equals(dataset.features.index)

    params = binary_params_from_settings(settings)
    params = {
        **params,
        "num_leaves": 8,
        "min_data_in_leaf": 5,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
    }
    kwargs = dict(
        model_params=params,
        num_boost_round=15,
        seed=SEED,
        universe=tuple(sorted(bars.bars)),
        start=date(2021, 1, 4),
        end=date(2021, 12, 31),
        frequency="1d",
        bar_timestamps=calendar,
    )
    first = run_meta_labeling_gate(dataset, bet_returns, **kwargs)
    second = run_meta_labeling_gate(dataset, bet_returns, **kwargs)
    assert first.n_trials == N_RESEARCH_TRIALS
    assert first.verdict in ("JA", "NEIN")
    assert first.horizon == settings.labels.vertical_bars
    assert first.dsr == pytest.approx(second.dsr, rel=0, abs=0)
    assert bar_timestamps_from_bars(bars).equals(calendar)


def test_permute_and_interpret() -> None:
    idx = pd.MultiIndex.from_product(
        [
            pd.date_range("2024-01-02", periods=2, freq="B", tz="UTC"),
            ["A", "B", "C"],
        ],
        names=("timestamp", "symbol"),
    )
    label = pd.Series([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], index=idx, dtype="float64")
    perm = permute_labels_within_dates(label, rng=np.random.default_rng(0))
    assert perm.index.equals(label.index)
    corrected, gate_ja, beats, broken = interpret_corrected_verdict(
        real_dsr=0.96,
        null_q95=0.1,
    )
    assert corrected == "JA" and gate_ja and beats and not broken


def test_meta_gate_module_no_yfinance() -> None:
    import aihedgefund.research.meta_labeling_gate as mod

    source = Path(mod.__file__).read_text(encoding="utf-8")
    assert "yfinance" not in source
