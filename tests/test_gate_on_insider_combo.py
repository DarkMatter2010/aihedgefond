"""Offline tests for insider_plus_all_new CPCV/DSR gate wiring (no network)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from aihedgefund.core.schemas import BarFrame, Form4Frame, Form4Record
from aihedgefund.features.feature_classes import ALL_NEW_FEATURE_CLASS_COLUMNS
from aihedgefund.features.insider import (
    INSIDER_FEATURE_COLUMNS,
    INSIDER_PLUS_ALL_NEW_FEATURE_COLUMNS,
)
from aihedgefund.features.pipeline import FEATURE_COLUMNS
from aihedgefund.research.baseline import build_lgbm_params
from aihedgefund.research.insider_combo_gate import (
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
    assemble_insider_combo_dataset,
    assert_insider_combo_feature_set,
    bar_timestamps_from_bars,
    build_insider_combo_feature_matrix,
    cpcv_config_for_horizon,
    interpret_corrected_verdict,
    run_insider_combo_gate,
)
from aihedgefund.research.research_trials import N_RESEARCH_TRIALS
from aihedgefund.research.universes import BROAD_LIQUID_CANDIDATE_UNIVERSE

FIXTURE_SEED = 20260722


def _synthetic_symbol_frame(symbol_seed: int, rows: int = 200) -> pd.DataFrame:
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


def _bars(n: int = 8, rows: int = 200) -> BarFrame:
    symbols = BROAD_LIQUID_CANDIDATE_UNIVERSE[:n]
    frames = {
        symbol: _synthetic_symbol_frame(FIXTURE_SEED + idx, rows=rows)
        for idx, symbol in enumerate(symbols)
    }
    empty_ca = {
        symbol: pd.Series(0.0, index=frames[symbol].index, dtype="float64")
        for symbol in symbols
    }
    return BarFrame(bars=frames, dividends=empty_ca, splits=empty_ca)


def _dense_form4(symbols: tuple[str, ...]) -> Form4Frame:
    records: list[Form4Record] = []
    for i, symbol in enumerate(symbols):
        for week in range(0, 120, 2):
            filed = datetime(2021, 1, 4, 16, 0, tzinfo=UTC) + timedelta(weeks=week)
            if filed.year > 2024:
                break
            records.append(
                Form4Record(
                    symbol=symbol,
                    cik=str(i),
                    accession=f"acc-{i}-{week}",
                    filed_at=filed,
                    transaction_date=filed.date(),
                    transaction_code="P" if week % 4 == 0 else "S",
                    shares=float(100 * (i + 1) + week),
                    price=10.0,
                    acquired_disposed="A" if week % 4 == 0 else "D",
                    reporting_owner=f"OWN-{i}",
                )
            )
    return Form4Frame(
        records=tuple(records),
        symbols_queried=symbols,
        symbols_without_filings=(),
    )


def test_insider_combo_gate_constants() -> None:
    assert_insider_combo_feature_set()
    assert GATE_CANDIDATE_FEATURE_COLUMNS == INSIDER_PLUS_ALL_NEW_FEATURE_COLUMNS
    assert len(GATE_CANDIDATE_FEATURE_COLUMNS) == 24
    assert GATE_CANDIDATE_FEATURE_COLUMNS[:9] == INSIDER_FEATURE_COLUMNS
    assert GATE_CANDIDATE_FEATURE_COLUMNS[9:] == ALL_NEW_FEATURE_CLASS_COLUMNS
    assert set(INSIDER_FEATURE_COLUMNS).isdisjoint(FEATURE_COLUMNS)
    assert PRIMARY_HORIZON == 21
    assert SECONDARY_HORIZON == 2
    assert SEED == 42
    assert N_TRIALS == N_RESEARCH_TRIALS == 29
    assert N_BLOCKS == 6 and N_TEST_BLOCKS == 2
    assert N_PERMUTATIONS == 100
    assert DSR_THRESHOLD == 0.95
    assert COUNTS_AS_NEW_RESEARCH_TRIAL is False


def test_build_insider_combo_matrix_schema() -> None:
    bars = _bars(6, rows=160)
    symbols = tuple(sorted(bars.bars))
    form4 = _dense_form4(symbols)
    matrix = build_insider_combo_feature_matrix(bars, form4)
    assert tuple(matrix.columns) == INSIDER_PLUS_ALL_NEW_FEATURE_COLUMNS
    assert list(matrix.index.names) == ["timestamp", "symbol"]


def test_assemble_and_gate_deterministic_offline() -> None:
    bars = _bars(8, rows=220)
    symbols = tuple(sorted(bars.bars))
    form4 = _dense_form4(symbols)
    dataset = assemble_insider_combo_dataset(bars, form4, horizon=2)
    assert dataset.feature_columns == INSIDER_PLUS_ALL_NEW_FEATURE_COLUMNS
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
        universe=symbols,
        start=date(2021, 1, 4),
        end=date(2021, 12, 31),
        frequency="1d",
        bar_timestamps=calendar,
    )
    first = run_insider_combo_gate(dataset, **kwargs)
    second = run_insider_combo_gate(dataset, **kwargs)
    assert first.n_trials == N_RESEARCH_TRIALS
    assert first.verdict in ("JA", "NEIN")
    assert first.dsr == pytest.approx(second.dsr, rel=0, abs=0)
    assert cpcv_config_for_horizon(21).horizon == 21
    assert cpcv_config_for_horizon(21).embargo_days == 21


def test_interpret_corrected_verdict_reuse() -> None:
    corrected, gate_ja, beats, broken = interpret_corrected_verdict(
        real_dsr=0.96,
        null_q95=0.5,
    )
    assert corrected == "JA"
    assert gate_ja and beats and not broken
    corrected2, _, _, _ = interpret_corrected_verdict(real_dsr=0.5, null_q95=0.6)
    assert corrected2 == "NEIN"
