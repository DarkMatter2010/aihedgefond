"""Offline tests for small-cap universe diagnostic (no yfinance)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aihedgefund.core.config import load_settings
from aihedgefund.core.schemas import BarFrame, ICMetricsReport
from aihedgefund.features.feature_classes import ALL_NEW_FEATURE_CLASS_COLUMNS
from aihedgefund.research.feature_class_triage import (
    RANK_IC_MATERIAL_THRESHOLD,
    grinold_trial_sharpe,
)
from aihedgefund.research.small_cap_universe_diagnostic import (
    DIAGNOSTIC_HORIZONS,
    SmallCapUniverseRow,
    interpret_small_cap_universe,
    run_small_cap_universe_diagnostic,
    settings_for_small_cap_universe_diagnostic,
)
from aihedgefund.research.universes import (
    BROAD_LIQUID_CANDIDATE_UNIVERSE,
    SMALL_CAP_CANDIDATE_UNIVERSE,
)

SEED = 20260721


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


def _bars_from_universe_subset(n: int = 8, rows: int = 320) -> tuple[BarFrame, tuple[str, ...]]:
    symbols = SMALL_CAP_CANDIDATE_UNIVERSE[:n]
    frames = {
        symbol: _synthetic_symbol_frame(SEED + idx, rows=rows) for idx, symbol in enumerate(symbols)
    }
    empty_ca = {
        symbol: pd.Series(0.0, index=frames[symbol].index, dtype="float64") for symbol in symbols
    }
    bars = BarFrame(bars=frames, dividends=empty_ca, splits=empty_ca)
    return bars, symbols


def test_small_cap_universe_size_unique_disjoint() -> None:
    assert 150 <= len(SMALL_CAP_CANDIDATE_UNIVERSE) <= 300
    assert len(set(SMALL_CAP_CANDIDATE_UNIVERSE)) == len(SMALL_CAP_CANDIDATE_UNIVERSE)
    overlap = set(SMALL_CAP_CANDIDATE_UNIVERSE) & set(BROAD_LIQUID_CANDIDATE_UNIVERSE)
    assert not overlap
    assert SMALL_CAP_CANDIDATE_UNIVERSE == tuple(sorted(SMALL_CAP_CANDIDATE_UNIVERSE))


def test_settings_for_small_cap_universe_diagnostic() -> None:
    base = load_settings()
    diag = settings_for_small_cap_universe_diagnostic(base, horizon=21)
    assert diag.universe == SMALL_CAP_CANDIDATE_UNIVERSE
    assert diag.research.horizon == 21
    assert diag.research.embargo_days == 21
    assert diag.research.seed == base.research.seed
    assert diag.research.train_end == base.research.train_end
    assert (diag.research.test_start - diag.research.train_end).days >= 21


def test_interpret_small_cap_rules() -> None:
    def _row(horizon: int, rank_ic: float) -> SmallCapUniverseRow:
        metrics = ICMetricsReport(
            ic_mean=rank_ic,
            rank_ic_mean=rank_ic,
            icir=None,
            rank_icir=None,
            median_cs_breadth=50.0,
            cs_breadth_warning=False,
            ic_materially_positive=rank_ic > 0.02,
            ic_positive_threshold=0.02,
            n_dates=10,
        )
        return SmallCapUniverseRow(
            horizon=horizon,
            feature_column_count=len(ALL_NEW_FEATURE_CLASS_COLUMNS),
            metrics=metrics,
            grinold_sr=grinold_trial_sharpe(rank_ic=rank_ic, median_cs_breadth=50.0),
            rank_ic_above_threshold=rank_ic >= RANK_IC_MATERIAL_THRESHOLD,
            test_start=date(2021, 8, 2),
            broad_all_new_rank_ic=0.02,
        )

    exhausted, note = interpret_small_cap_universe((_row(2, 0.01), _row(21, 0.015)))
    assert exhausted == "free_ohlcv_search_exhausted"
    assert "exhausted" in note.lower() or "exhausted" in note

    candidate, cnote = interpret_small_cap_universe((_row(2, 0.01), _row(21, 0.025)))
    assert candidate == "candidate_for_gate"
    assert "0.025" in cnote or "h=21" in cnote


def test_run_small_cap_universe_diagnostic_deterministic_offline(tmp_path: Path) -> None:
    bars, symbols = _bars_from_universe_subset(8, rows=320)
    base = load_settings()
    research = base.research.model_copy(
        update={
            "horizon": 2,
            "embargo_days": 2,
            "train_end": date(2021, 6, 30),
            "test_start": date(2021, 8, 2),
            "seed": SEED,
            "num_boost_round": 20,
        }
    )
    settings = base.model_copy(
        update={
            "universe": symbols,
            "research": research,
            "start": date(2021, 1, 4),
            "end": date(2021, 12, 31),
            "artifact_root": tmp_path / "artifacts",
        }
    )
    settings.artifact_root.mkdir(parents=True, exist_ok=True)

    first = run_small_cap_universe_diagnostic(bars, settings, n_symbols_dropped=3)
    second = run_small_cap_universe_diagnostic(bars, settings, n_symbols_dropped=3)
    assert first.horizons == DIAGNOSTIC_HORIZONS
    assert first.feature_columns == ALL_NEW_FEATURE_CLASS_COLUMNS
    assert first.counts_as_research_trial is True
    assert first.n_symbols_dropped == 3
    assert first.drop_rate == pytest.approx(3 / len(SMALL_CAP_CANDIDATE_UNIVERSE))
    assert first.n_symbols == len(symbols)
    assert len(first.rows) == len(DIAGNOSTIC_HORIZONS)
    assert first.survivorship_bias_note

    for a, b in zip(first.rows, second.rows, strict=True):
        assert a.horizon == b.horizon
        assert a.metrics.rank_ic_mean == pytest.approx(b.metrics.rank_ic_mean, rel=0, abs=0)
        assert a.metrics.ic_mean == pytest.approx(b.metrics.ic_mean, rel=0, abs=0)
        assert a.grinold_sr == pytest.approx(b.grinold_sr, rel=0, abs=0)


def test_small_cap_module_does_not_import_yfinance() -> None:
    import aihedgefund.research.small_cap_universe_diagnostic as mod

    assert "yfinance" not in mod.__dict__
    source = Path(mod.__file__).read_text(encoding="utf-8")
    assert "yfinance" not in source
