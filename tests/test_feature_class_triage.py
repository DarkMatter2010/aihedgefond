"""Offline tests for feature-class IC triage (no yfinance)."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aihedgefund.core.bus import InProcessMessageBus
from aihedgefund.core.config import load_settings
from aihedgefund.core.schemas import BarFrame, ICMetricsReport
from aihedgefund.features.feature_classes import (
    ALL_NEW_FEATURE_CLASS_COLUMNS,
    FEATURE_CLASS_CONFIGS,
    LOW_VOL_FEATURE_COLUMNS,
    LOW_VOL_RAW_FEATURE_COLUMNS,
    MAX_TRIAGE_FEATURE_WARMUP_BARS,
    NEW_PLUS_OLD_FEATURE_COLUMNS,
    RANGE_VOL_FEATURE_COLUMNS,
    RANGE_VOL_RAW_FEATURE_COLUMNS,
    REVERSAL_FEATURE_COLUMNS,
    REVERSAL_RAW_FEATURE_COLUMNS,
    REVERSAL_SIGN_NOTE,
    build_triage_feature_matrix,
    compute_symbol_feature_class_raws,
)
from aihedgefund.features.indicators import (
    inverse_realized_vol,
    parkinson_volatility,
    reversal,
)
from aihedgefund.features.pipeline import FEATURE_COLUMNS, FeaturePipeline
from aihedgefund.research.feature_class_triage import (
    RANK_IC_MATERIAL_THRESHOLD,
    TRIAGE_HORIZONS,
    FeatureClassTriageRow,
    grinold_trial_sharpe,
    interpret_feature_class_triage,
    run_feature_class_triage,
    settings_for_feature_class_triage,
)
from aihedgefund.research.universes import BROAD_LIQUID_CANDIDATE_UNIVERSE

SEED = 20260719
CREATED_AT = datetime(2026, 7, 19, 15, 0, tzinfo=UTC)


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


def _bars_from_universe_subset(n: int = 8, rows: int = 220) -> tuple[BarFrame, tuple[str, ...]]:
    symbols = BROAD_LIQUID_CANDIDATE_UNIVERSE[:n]
    frames = {
        symbol: _synthetic_symbol_frame(SEED + idx, rows=rows) for idx, symbol in enumerate(symbols)
    }
    empty_ca = {
        symbol: pd.Series(0.0, index=frames[symbol].index, dtype="float64") for symbol in symbols
    }
    bars = BarFrame(bars=frames, dividends=empty_ca, splits=empty_ca)
    return bars, symbols


def test_production_feature_columns_unchanged() -> None:
    assert len(FEATURE_COLUMNS) == 36
    assert set(ALL_NEW_FEATURE_CLASS_COLUMNS).isdisjoint(FEATURE_COLUMNS)
    assert NEW_PLUS_OLD_FEATURE_COLUMNS[:36] == FEATURE_COLUMNS
    assert NEW_PLUS_OLD_FEATURE_COLUMNS[36:] == ALL_NEW_FEATURE_CLASS_COLUMNS


def test_feature_class_registries_identifiable() -> None:
    assert REVERSAL_RAW_FEATURE_COLUMNS == ("reversal_5", "reversal_21")
    assert LOW_VOL_RAW_FEATURE_COLUMNS == ("inv_ret_std_21", "inv_ret_std_63")
    assert RANGE_VOL_RAW_FEATURE_COLUMNS == ("parkinson_vol_21",)
    assert MAX_TRIAGE_FEATURE_WARMUP_BARS == 63
    labels = [label for label, _ in FEATURE_CLASS_CONFIGS]
    assert labels == ["reversal", "low_vol", "range_vol", "all_new", "new_plus_old"]
    assert FEATURE_CLASS_CONFIGS[0][1] == REVERSAL_FEATURE_COLUMNS
    assert FEATURE_CLASS_CONFIGS[1][1] == LOW_VOL_FEATURE_COLUMNS
    assert FEATURE_CLASS_CONFIGS[2][1] == RANGE_VOL_FEATURE_COLUMNS
    assert "reversal_k = -1" in REVERSAL_SIGN_NOTE


def test_reversal_sign_negative_after_up_move() -> None:
    close = pd.Series(
        [100.0, 101.0, 102.0, 103.0, 104.0, 110.0],
        index=pd.date_range("2021-01-04", periods=6, freq="B", tz="UTC"),
    )
    rev5 = reversal(close, 5)
    # close[5]/close[0] - 1 = 0.10 → reversal = -0.10
    assert rev5.iloc[5] == pytest.approx(-0.10)
    assert rev5.iloc[5] < 0.0


def test_low_vol_inverse_ranks_lower_vol_higher() -> None:
    index = pd.date_range("2021-01-04", periods=40, freq="B", tz="UTC")
    rng = np.random.default_rng(7)
    calm = pd.Series(
        100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.001, 40))),
        index=index,
    )
    wild = pd.Series(
        100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.05, 40))),
        index=index,
    )
    inv_calm = inverse_realized_vol(calm, 21).iloc[-1]
    inv_wild = inverse_realized_vol(wild, 21).iloc[-1]
    assert np.isfinite(inv_calm) and np.isfinite(inv_wild)
    assert inv_calm > inv_wild


def test_parkinson_and_reversal_pit_no_lookahead() -> None:
    frame = _synthetic_symbol_frame(SEED, rows=80)
    anchor = frame.index[50]
    full = compute_symbol_feature_class_raws(frame)
    truncated = compute_symbol_feature_class_raws(frame.loc[:anchor])
    for column in (
        *REVERSAL_RAW_FEATURE_COLUMNS,
        *LOW_VOL_RAW_FEATURE_COLUMNS,
        *RANGE_VOL_RAW_FEATURE_COLUMNS,
    ):
        assert full.loc[anchor, column] == pytest.approx(
            truncated.loc[anchor, column],
            rel=1e-12,
            abs=1e-12,
            nan_ok=True,
        )


def test_parkinson_uses_only_high_low_through_t() -> None:
    frame = _synthetic_symbol_frame(SEED + 1, rows=60)
    anchor = frame.index[40]
    full = parkinson_volatility(frame, 21)
    # Inflate future highs — must not change value at anchor.
    mutated = frame.copy()
    mutated.loc[mutated.index > anchor, "high"] = mutated.loc[mutated.index > anchor, "high"] * 10.0
    truncated_equiv = parkinson_volatility(mutated, 21)
    assert full.loc[anchor] == pytest.approx(truncated_equiv.loc[anchor], rel=1e-12, abs=1e-12)


def test_feature_pipeline_still_returns_only_production_columns() -> None:
    bars, _symbols = _bars_from_universe_subset(4, rows=100)
    bus = InProcessMessageBus()
    matrix = FeaturePipeline(bus).compute(bars)
    assert tuple(matrix.columns) == FEATURE_COLUMNS
    assert len(matrix.columns) == 36


def test_build_triage_matrix_includes_new_and_old() -> None:
    bars, _symbols = _bars_from_universe_subset(4, rows=100)
    matrix = build_triage_feature_matrix(bars)
    assert tuple(matrix.columns) == NEW_PLUS_OLD_FEATURE_COLUMNS
    assert "reversal_5" in matrix.columns
    assert "reversal_5_cs_zscore" in matrix.columns
    assert "inv_ret_std_63" in matrix.columns
    assert "parkinson_vol_21_cs_rank" in matrix.columns
    assert "momentum_20" in matrix.columns


def test_settings_for_feature_class_triage() -> None:
    base = load_settings()
    diag = settings_for_feature_class_triage(base, horizon=21)
    assert diag.universe == BROAD_LIQUID_CANDIDATE_UNIVERSE
    assert diag.research.horizon == 21
    assert diag.research.embargo_days == 21
    assert diag.research.seed == base.research.seed
    assert diag.research.train_end == base.research.train_end
    # limits.yaml gap is 11 days; h=21 must push test_start forward.
    from aihedgefund.research.feature_class_triage import test_start_for_horizon

    expected = test_start_for_horizon(
        base.research.train_end,
        base.research.test_start,
        21,
    )
    assert diag.research.test_start == expected
    assert (diag.research.test_start - diag.research.train_end).days >= 21


def test_test_start_for_horizon_keeps_or_extends() -> None:
    from pandas.tseries.offsets import BDay

    from aihedgefund.research.feature_class_triage import test_start_for_horizon

    train_end = date(2022, 12, 30)
    configured = date(2023, 1, 10)
    # Without a trading calendar, holiday-buffered BDay(horizon+5) applies.
    assert test_start_for_horizon(train_end, configured, 2) == configured
    expected_h21 = (pd.Timestamp(train_end) + BDay(26)).date()
    assert test_start_for_horizon(train_end, configured, 21) == expected_h21

    # With an explicit calendar, require bar_gap > horizon.
    sessions = tuple(d.date() for d in pd.bdate_range("2022-12-01", "2023-03-01", freq="C"))
    cal_start = test_start_for_horizon(
        train_end,
        configured,
        21,
        trading_dates=sessions,
    )
    train_pos = max(i for i, d in enumerate(sessions) if d <= train_end)
    test_pos = sessions.index(cal_start)
    assert test_pos - train_pos > 21


def test_grinold_and_interpret_helpers() -> None:
    assert grinold_trial_sharpe(rank_ic=0.02, median_cs_breadth=100.0) == pytest.approx(0.2)

    def _row(label: str, horizon: int, rank_ic: float) -> FeatureClassTriageRow:
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
        return FeatureClassTriageRow(
            class_label=label,
            horizon=horizon,
            feature_column_count=6,
            metrics=metrics,
            grinold_sr=grinold_trial_sharpe(rank_ic=rank_ic, median_cs_breadth=50.0),
            rank_ic_above_threshold=rank_ic >= RANK_IC_MATERIAL_THRESHOLD,
            test_start=date(2021, 8, 2),
        )

    dead_rows = (
        _row("reversal", 2, 0.03),
        _row("reversal", 21, 0.01),
    )
    label, _ = interpret_feature_class_triage(dead_rows)
    assert label == "ohlcv_features_dead"

    live_rows = (
        _row("reversal", 2, 0.03),
        _row("reversal", 21, 0.025),
        _row("low_vol", 2, 0.0),
        _row("low_vol", 21, 0.0),
    )
    label, note = interpret_feature_class_triage(live_rows)
    assert label == "candidate_for_gate"
    assert "reversal" in note


def test_run_feature_class_triage_deterministic_offline(tmp_path: Path) -> None:
    # Need enough calendar length for h=21 embargo gap + OOS window.
    bars, symbols = _bars_from_universe_subset(8, rows=320)
    base = load_settings()
    research = base.research.model_copy(
        update={
            "horizon": 2,
            "embargo_days": 2,
            "train_end": date(2021, 6, 30),
            "test_start": date(2021, 8, 2),  # >= 21 calendar days after train_end
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

    first = run_feature_class_triage(bars, settings)
    second = run_feature_class_triage(bars, settings)
    assert first.n_configs_measured == len(FEATURE_CLASS_CONFIGS) * len(TRIAGE_HORIZONS)
    assert first.horizons == TRIAGE_HORIZONS
    assert first.counts_as_research_trial is True
    assert first.survivorship_bias_note
    assert "reversal" in first.reversal_sign_note.lower() or "-1" in first.reversal_sign_note

    assert len(first.rows) == len(second.rows)
    for a, b in zip(first.rows, second.rows, strict=True):
        assert a.class_label == b.class_label
        assert a.horizon == b.horizon
        assert a.metrics.rank_ic_mean == pytest.approx(b.metrics.rank_ic_mean, rel=0, abs=0)
        assert a.metrics.ic_mean == pytest.approx(b.metrics.ic_mean, rel=0, abs=0)
        assert a.grinold_sr == pytest.approx(b.grinold_sr, rel=0, abs=0)


def test_feature_class_triage_module_does_not_import_yfinance() -> None:
    import aihedgefund.features.feature_classes as feat
    import aihedgefund.research.feature_class_triage as mod

    assert "yfinance" not in mod.__dict__
    assert "yfinance" not in feat.__dict__
    source = Path(mod.__file__).read_text(encoding="utf-8")
    assert "yfinance" not in source
    feat_source = Path(feat.__file__).read_text(encoding="utf-8")
    assert "yfinance" not in feat_source
