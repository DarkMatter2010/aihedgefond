"""Offline tests for all_new CPCV/DSR gate wiring (no yfinance)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aihedgefund.core.schemas import BarFrame
from aihedgefund.features.feature_classes import ALL_NEW_FEATURE_CLASS_COLUMNS
from aihedgefund.features.pipeline import FEATURE_COLUMNS
from aihedgefund.research.all_new_gate import (
    COUNTS_AS_NEW_RESEARCH_TRIAL,
    DSR_THRESHOLD,
    GATE_CANDIDATE_FEATURE_COLUMNS,
    N_BLOCKS,
    N_PERMUTATIONS,
    N_TEST_BLOCKS,
    N_TRIALS,
    PRIMARY_HORIZON,
    SECONDARY_HORIZON,
    SEED,
    assemble_all_new_dataset,
    assert_all_new_feature_set,
    bar_timestamps_from_bars,
    build_all_new_feature_matrix,
    cpcv_config_for_horizon,
    interpret_corrected_verdict,
    permute_labels_within_dates,
    run_all_new_gate,
)
from aihedgefund.research.baseline import build_lgbm_params
from aihedgefund.research.research_trials import N_RESEARCH_TRIALS
from aihedgefund.research.universes import BROAD_LIQUID_CANDIDATE_UNIVERSE

FIXTURE_SEED = 20260720


def _synthetic_symbol_frame(symbol_seed: int, rows: int = 180) -> pd.DataFrame:
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


def _bars(n: int = 8, rows: int = 180) -> BarFrame:
    symbols = BROAD_LIQUID_CANDIDATE_UNIVERSE[:n]
    frames = {
        symbol: _synthetic_symbol_frame(FIXTURE_SEED + idx, rows=rows)
        for idx, symbol in enumerate(symbols)
    }
    empty_ca = {
        symbol: pd.Series(0.0, index=frames[symbol].index, dtype="float64") for symbol in symbols
    }
    return BarFrame(bars=frames, dividends=empty_ca, splits=empty_ca)


def test_all_new_gate_constants() -> None:
    assert_all_new_feature_set()
    assert GATE_CANDIDATE_FEATURE_COLUMNS == ALL_NEW_FEATURE_CLASS_COLUMNS
    assert len(GATE_CANDIDATE_FEATURE_COLUMNS) == 15
    assert set(GATE_CANDIDATE_FEATURE_COLUMNS).isdisjoint(FEATURE_COLUMNS)
    assert PRIMARY_HORIZON == 21
    assert SECONDARY_HORIZON == 2
    assert SEED == 42
    assert N_TRIALS == N_RESEARCH_TRIALS == 24
    assert N_BLOCKS == 6 and N_TEST_BLOCKS == 2
    assert N_PERMUTATIONS == 100
    assert DSR_THRESHOLD == 0.95
    assert COUNTS_AS_NEW_RESEARCH_TRIAL is False


def test_build_all_new_matrix_schema() -> None:
    bars = _bars(6, rows=120)
    matrix = build_all_new_feature_matrix(bars)
    assert tuple(matrix.columns) == ALL_NEW_FEATURE_CLASS_COLUMNS
    assert list(matrix.index.names) == ["timestamp", "symbol"]


def test_assemble_and_gate_deterministic_offline() -> None:
    bars = _bars(8, rows=200)
    dataset = assemble_all_new_dataset(bars, horizon=2)
    assert dataset.feature_columns == ALL_NEW_FEATURE_CLASS_COLUMNS
    assert dataset.horizon == 2
    calendar = bar_timestamps_from_bars(bars)
    params = build_lgbm_params(
        seed=SEED,
        learning_rate=0.1,
        num_leaves=8,
        min_data_in_leaf=5,
        feature_fraction=1.0,
        bagging_fraction=1.0,
        bagging_freq=0,
    )
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
    first = run_all_new_gate(dataset, **kwargs)
    second = run_all_new_gate(dataset, **kwargs)
    assert first.n_trials == N_RESEARCH_TRIALS
    assert first.verdict in ("JA", "NEIN")
    assert first.dsr == pytest.approx(second.dsr, rel=0, abs=0)
    assert cpcv_config_for_horizon(2).horizon == 2
    assert cpcv_config_for_horizon(2).embargo_days == 2


def test_permute_labels_preserves_index_destroys_cs_order() -> None:
    idx = pd.MultiIndex.from_product(
        [
            pd.date_range("2024-01-02", periods=2, freq="B", tz="UTC"),
            ["A", "B", "C"],
        ],
        names=("timestamp", "symbol"),
    )
    label = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=idx, dtype="float64")
    rng = np.random.default_rng(0)
    perm = permute_labels_within_dates(label, rng=rng)
    assert perm.index.equals(label.index)
    # Same multiset per date after shuffle.
    for ts in label.index.get_level_values("timestamp").unique():
        assert sorted(perm.xs(ts, level="timestamp")) == sorted(label.xs(ts, level="timestamp"))


def test_interpret_corrected_verdict() -> None:
    corrected, gate_ja, beats, broken = interpret_corrected_verdict(
        real_dsr=0.96,
        null_q95=0.1,
    )
    assert corrected == "JA" and gate_ja and beats and not broken
    corrected, gate_ja, beats, broken = interpret_corrected_verdict(
        real_dsr=0.5,
        null_q95=0.1,
    )
    assert corrected == "NEIN" and not gate_ja
    corrected, _, _, broken = interpret_corrected_verdict(real_dsr=0.99, null_q95=0.95)
    assert corrected == "NEIN" and broken


def test_all_new_gate_module_no_yfinance() -> None:
    import aihedgefund.research.all_new_gate as mod

    source = Path(mod.__file__).read_text(encoding="utf-8")
    assert "yfinance" not in source
    assert "yfinance" not in mod.__dict__
