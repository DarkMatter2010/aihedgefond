"""Offline tests for the universe-breadth diagnostic (no yfinance)."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aihedgefund.core.bus import InProcessMessageBus
from aihedgefund.core.config import load_settings
from aihedgefund.core.schemas import BarFrame
from aihedgefund.features.pipeline import FEATURE_COLUMNS, FeaturePipeline
from aihedgefund.research.adapters.filesystem import FilesystemModelArtifactAdapter
from aihedgefund.research.universe_breadth_diagnostic import (
    DIAGNOSTIC_HORIZON,
    NARROW_H2_RANK_IC,
    interpret_breadth_vs_narrow,
    run_universe_breadth_diagnostic,
    settings_for_breadth_diagnostic,
)
from aihedgefund.research.universes import BROAD_LIQUID_CANDIDATE_UNIVERSE

SEED = 20260719
CREATED_AT = datetime(2026, 7, 19, 12, 0, tzinfo=UTC)


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


def _bars_from_universe_subset(n: int = 8) -> tuple[BarFrame, tuple[str, ...]]:
    symbols = BROAD_LIQUID_CANDIDATE_UNIVERSE[:n]
    frames = {
        symbol: _synthetic_symbol_frame(SEED + idx, rows=220)
        for idx, symbol in enumerate(symbols)
    }
    empty_ca = {
        symbol: pd.Series(0.0, index=frames[symbol].index, dtype="float64")
        for symbol in symbols
    }
    bars = BarFrame(bars=frames, dividends=empty_ca, splits=empty_ca)
    return bars, symbols


def test_broad_universe_constant_is_large_and_stable() -> None:
    assert len(BROAD_LIQUID_CANDIDATE_UNIVERSE) >= 400
    assert BROAD_LIQUID_CANDIDATE_UNIVERSE[0] == "A"
    assert "AAPL" in BROAD_LIQUID_CANDIDATE_UNIVERSE
    assert len(set(BROAD_LIQUID_CANDIDATE_UNIVERSE)) == len(BROAD_LIQUID_CANDIDATE_UNIVERSE)


def test_settings_for_breadth_diagnostic_changes_only_universe_and_horizon() -> None:
    base = load_settings()
    diag = settings_for_breadth_diagnostic(base)
    assert diag.universe == BROAD_LIQUID_CANDIDATE_UNIVERSE
    assert diag.research.horizon == DIAGNOSTIC_HORIZON
    assert diag.research.embargo_days == DIAGNOSTIC_HORIZON
    assert diag.research.seed == base.research.seed
    assert diag.research.train_end == base.research.train_end
    assert diag.research.test_start == base.research.test_start
    assert diag.research.num_boost_round == base.research.num_boost_round
    assert diag.research.learning_rate == base.research.learning_rate
    assert diag.start == base.start
    assert diag.end == base.end


def test_interpret_breadth_rules() -> None:
    label, _ = interpret_breadth_vs_narrow(rank_ic_broad=0.04)
    assert label == "breadth_noise"
    label, _ = interpret_breadth_vs_narrow(rank_ic_broad=0.001)
    assert label == "features_dead"
    # Above narrow reference but still below 0.02 material threshold.
    label, _ = interpret_breadth_vs_narrow(rank_ic_broad=0.019)
    assert label == "inconclusive"


def test_run_universe_breadth_diagnostic_deterministic_offline(tmp_path: Path) -> None:
    bars, symbols = _bars_from_universe_subset(8)
    base = load_settings()
    # Shrink calendar so synthetic bars cover train/test with h=2.
    research = base.research.model_copy(
        update={
            "horizon": DIAGNOSTIC_HORIZON,
            "embargo_days": DIAGNOSTIC_HORIZON,
            "train_end": date(2021, 6, 30),
            "test_start": date(2021, 7, 15),
            "seed": SEED,
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
    bus = InProcessMessageBus()
    pipeline = FeaturePipeline(bus)
    adapter = FilesystemModelArtifactAdapter(settings.artifact_root)

    first = run_universe_breadth_diagnostic(
        bars,
        settings,
        feature_pipeline=pipeline,
        artifact_adapter=adapter,
        created_at=CREATED_AT,
    )
    second_root = tmp_path / "artifacts2"
    second_root.mkdir(parents=True, exist_ok=True)
    second = run_universe_breadth_diagnostic(
        bars,
        settings,
        feature_pipeline=FeaturePipeline(InProcessMessageBus()),
        artifact_adapter=FilesystemModelArtifactAdapter(second_root),
        created_at=CREATED_AT,
    )

    assert first.feature_columns == FEATURE_COLUMNS
    assert second.feature_columns == FEATURE_COLUMNS
    assert first.horizon == DIAGNOSTIC_HORIZON
    assert first.n_symbols == len(symbols)
    assert first.metrics.rank_ic_mean == pytest.approx(second.metrics.rank_ic_mean)
    assert first.metrics.ic_mean == pytest.approx(second.metrics.ic_mean)
    assert first.metrics.median_cs_breadth == pytest.approx(second.metrics.median_cs_breadth)
    assert first.narrow_rank_ic_reference == pytest.approx(NARROW_H2_RANK_IC)
    assert first.counts_as_research_trial is True
    assert "survivorship" in first.survivorship_bias_note.lower()
    assert first.interpretation in {"breadth_noise", "features_dead", "inconclusive"}
