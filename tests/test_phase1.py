"""Offline, deterministic definition-of-done tests for Phase 1."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel, ValidationError
from statsmodels.tsa.stattools import adfuller

import aihedgefund.data.adapters.yfinance as yfinance_adapter
from aihedgefund.core.bus import InProcessMessageBus
from aihedgefund.core.config import LabelSettings, QualitySettings, load_settings
from aihedgefund.core.runtime import FrozenClock, SeededIdProvider
from aihedgefund.core.schemas import (
    BarFrame,
    CorporateActionInput,
    DataIngested,
    FeaturesComputed,
    LabelsComputed,
    MarketDataRequest,
    OHLCVBar,
    OHLCVRequest,
    PointInTimeFrame,
    QualityGateFailed,
    QualityReportProduced,
)
from aihedgefund.data.adapters import Phase0DataVendorAdapter, YFinanceProvider
from aihedgefund.data.corporate_actions import adjust_corporate_actions
from aihedgefund.data.provider import (
    DataUnavailableError,
    MarketDataProvider,
    ProviderChain,
)
from aihedgefund.data.quality import DataQualityError, DataQualityGate
from aihedgefund.features.pipeline import (
    FEATURE_COLUMNS,
    FeaturePipeline,
    to_feature_vectors,
)
from aihedgefund.features.pit import assert_no_lookahead, pit_join
from aihedgefund.labels.fracdiff import configured_frac_diff, frac_diff_ffd, min_ffd_d
from aihedgefund.labels.labeling import (
    cusum_filter,
    sample_weights,
    triple_barrier,
)
from aihedgefund.labels.pipeline import LabelPipeline

SEED = 1729


def synthetic_ohlcv(rows: int = 180) -> pd.DataFrame:
    """Generate deterministic business-day GBM bars."""
    rng = np.random.default_rng(SEED)
    index = pd.date_range("2025-01-02", periods=rows, freq="B", tz="UTC")
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


def bar_frame(frame: pd.DataFrame | None = None) -> BarFrame:
    """Wrap one canonical frame with aligned zero corporate actions."""
    canonical = synthetic_ohlcv() if frame is None else frame
    zeros = pd.Series(0.0, index=canonical.index)
    return BarFrame(
        bars={"AAPL": canonical},
        dividends={"AAPL": zeros.copy()},
        splits={"AAPL": zeros.copy()},
    )


def market_data_request(
    frame: pd.DataFrame,
    symbols: tuple[str, ...] = ("AAPL",),
) -> MarketDataRequest:
    """Build a typed request spanning a synthetic frame."""
    return MarketDataRequest(
        symbols=symbols,
        start=frame.index[0].to_pydatetime(),
        end=(frame.index[-1] + pd.Timedelta(days=1)).to_pydatetime(),
        frequency="1d",
    )


def yfinance_fixture(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert canonical synthetic bars to an in-memory Yahoo response."""
    result = frame.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "adj_close": "Adj Close",
            "volume": "Volume",
        }
    )
    result["Dividends"] = 0.0
    result["Stock Splits"] = 0.0
    return result


def test_yfinance_fixture_is_canonical_utc_and_validated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = synthetic_ohlcv(8).rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "adj_close": "Adj Close",
            "volume": "Volume",
        }
    )
    source.index = source.index.tz_localize(None)
    source["Dividends"] = 0.0
    source["Stock Splits"] = 0.0

    def fake_download(**kwargs: object) -> pd.DataFrame:
        assert kwargs["auto_adjust"] is False
        assert kwargs["actions"] is True
        return source

    monkeypatch.setattr(yfinance_adapter.yf, "download", fake_download)
    bus = InProcessMessageBus()
    ingested: list[DataIngested] = []
    bus.subscribe_event(DataIngested, ingested.append)
    clock = FrozenClock(source.index[-1].tz_localize("UTC").to_pydatetime())
    provider = YFinanceProvider(
        {"APPLE": "AAPL"},
        bus,
        DataQualityGate(load_settings().quality, bus, clock=clock),
        clock=clock,
    )

    result = provider.get_ohlcv(
        MarketDataRequest(
            symbols=("APPLE",),
            start=datetime(2025, 1, 1, tzinfo=UTC),
            end=datetime(2025, 2, 1, tzinfo=UTC),
            frequency="1d",
        )
    )

    canonical = result.bars["APPLE"]
    assert isinstance(result, BaseModel)
    assert tuple(canonical.columns) == (
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    )
    assert str(canonical.index.tz) == "UTC"
    assert result.dividends["APPLE"].eq(0.0).all()
    assert ingested[0].rows == {"APPLE": 8}
    assert ingested[0].payload.data == result
    assert ingested[0].timestamp == clock.now()
    first = canonical.iloc[0]
    validated_bar = OHLCVBar(
        symbol="APPLE",
        timestamp=canonical.index[0].to_pydatetime(),
        open=Decimal(str(first["open"])),
        high=Decimal(str(first["high"])),
        low=Decimal(str(first["low"])),
        close=Decimal(str(first["close"])),
        volume=Decimal(str(first["volume"])),
    )
    assert validated_bar.symbol == "APPLE"


@pytest.mark.parametrize("corruption", ["stale", "nan_ratio", "outlier"])
def test_ingest_quality_gate_blocks_bad_frames_before_features(
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    canonical = synthetic_ohlcv(60)
    if corruption == "nan_ratio":
        canonical.iloc[20, canonical.columns.get_loc("close")] = np.nan
    elif corruption == "outlier":
        canonical.iloc[30, canonical.columns.get_loc("close")] *= 10.0
    source = yfinance_fixture(canonical)
    monkeypatch.setattr(yfinance_adapter.yf, "download", lambda **_: source)

    checked_at = canonical.index[-1]
    if corruption == "stale":
        checked_at += pd.Timedelta(days=8)
    clock = FrozenClock(checked_at.to_pydatetime())
    bus = InProcessMessageBus()
    ingested: list[DataIngested] = []
    features: list[FeaturesComputed] = []
    bus.subscribe_event(DataIngested, ingested.append)
    bus.subscribe_event(FeaturesComputed, features.append)
    provider = YFinanceProvider(
        {},
        bus,
        DataQualityGate(load_settings().quality, bus, clock=clock),
        clock=clock,
    )

    with pytest.raises(DataQualityError):
        bars = provider.get_ohlcv(market_data_request(canonical))
        FeaturePipeline(bus, clock=clock).compute(bars)

    assert ingested == []
    assert features == []


def test_ingest_rejects_missing_corporate_action_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = synthetic_ohlcv(20)
    source = yfinance_fixture(canonical).drop(columns="Stock Splits")
    monkeypatch.setattr(yfinance_adapter.yf, "download", lambda **_: source)
    clock = FrozenClock(canonical.index[-1].to_pydatetime())
    bus = InProcessMessageBus()
    provider = YFinanceProvider(
        {},
        bus,
        DataQualityGate(load_settings().quality, bus, clock=clock),
        clock=clock,
    )

    with pytest.raises(DataUnavailableError, match="required action column"):
        provider.get_ohlcv(market_data_request(canonical))


@pytest.mark.parametrize(
    ("column", "value"),
    [
        pytest.param("Stock Splits", np.nan, id="nan-split"),
        pytest.param("Stock Splits", np.inf, id="infinite-split"),
        pytest.param("Stock Splits", -1.0, id="non-positive-split-factor"),
        pytest.param("Dividends", -0.01, id="negative-dividend"),
    ],
)
def test_ingest_rejects_corrupt_corporate_actions_before_event(
    monkeypatch: pytest.MonkeyPatch,
    column: str,
    value: float,
) -> None:
    canonical = synthetic_ohlcv(20)
    source = yfinance_fixture(canonical)
    source.loc[source.index[5], column] = value
    monkeypatch.setattr(yfinance_adapter.yf, "download", lambda **_: source)
    clock = FrozenClock(canonical.index[-1].to_pydatetime())
    bus = InProcessMessageBus()
    ingested: list[DataIngested] = []
    failures: list[QualityGateFailed] = []
    bus.subscribe_event(DataIngested, ingested.append)
    bus.subscribe_event(QualityGateFailed, failures.append)
    provider = YFinanceProvider(
        {},
        bus,
        DataQualityGate(load_settings().quality, bus, clock=clock),
        clock=clock,
    )

    with pytest.raises(DataQualityError):
        provider.get_ohlcv(market_data_request(canonical))

    assert ingested == []
    assert [failure.symbol for failure in failures] == ["AAPL"]


@pytest.mark.parametrize(
    ("column", "value"),
    [
        pytest.param("open", np.nan, id="nan-open"),
        pytest.param("open", np.inf, id="infinite-open"),
        pytest.param("open", -1.0, id="negative-open"),
        pytest.param("high", np.nan, id="nan-high"),
        pytest.param("high", np.inf, id="infinite-high"),
        pytest.param("high", -1.0, id="negative-high"),
        pytest.param("low", np.nan, id="nan-low"),
        pytest.param("low", np.inf, id="infinite-low"),
        pytest.param("low", -1.0, id="negative-low"),
        pytest.param("close", np.nan, id="nan-close"),
        pytest.param("close", np.inf, id="infinite-close"),
        pytest.param("close", -1.0, id="negative-close"),
        pytest.param("volume", np.nan, id="nan-volume"),
        pytest.param("volume", np.inf, id="infinite-volume"),
        pytest.param("volume", -1.0, id="negative-volume"),
    ],
)
def test_ingest_rejects_invalid_ohlcv_values_before_event(
    monkeypatch: pytest.MonkeyPatch,
    column: str,
    value: float,
) -> None:
    canonical = synthetic_ohlcv(20)
    canonical.loc[canonical.index[5], column] = value
    source = yfinance_fixture(canonical)
    monkeypatch.setattr(yfinance_adapter.yf, "download", lambda **_: source)
    clock = FrozenClock(canonical.index[-1].to_pydatetime())
    bus = InProcessMessageBus()
    ingested: list[DataIngested] = []
    failures: list[QualityGateFailed] = []
    bus.subscribe_event(DataIngested, ingested.append)
    bus.subscribe_event(QualityGateFailed, failures.append)
    provider = YFinanceProvider(
        {},
        bus,
        DataQualityGate(load_settings().quality, bus, clock=clock),
        clock=clock,
    )

    with pytest.raises(DataQualityError):
        provider.get_ohlcv(market_data_request(canonical))

    assert ingested == []
    assert [failure.symbol for failure in failures] == ["AAPL"]


def test_ingest_rejects_high_below_low_before_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = synthetic_ohlcv(20)
    canonical.loc[canonical.index[5], "high"] = canonical.loc[canonical.index[5], "low"] - 1.0
    source = yfinance_fixture(canonical)
    monkeypatch.setattr(yfinance_adapter.yf, "download", lambda **_: source)
    clock = FrozenClock(canonical.index[-1].to_pydatetime())
    bus = InProcessMessageBus()
    ingested: list[DataIngested] = []
    failures: list[QualityGateFailed] = []
    bus.subscribe_event(DataIngested, ingested.append)
    bus.subscribe_event(QualityGateFailed, failures.append)
    provider = YFinanceProvider(
        {},
        bus,
        DataQualityGate(load_settings().quality, bus, clock=clock),
        clock=clock,
    )

    with pytest.raises(DataQualityError, match="price bounds"):
        provider.get_ohlcv(market_data_request(canonical))

    assert ingested == []
    assert [failure.symbol for failure in failures] == ["AAPL"]


def test_ingest_publishes_event_for_valid_ohlcv_and_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = synthetic_ohlcv(20)
    source = yfinance_fixture(canonical)
    source.loc[source.index[5], "Stock Splits"] = 2.0
    source.loc[source.index[10], "Dividends"] = 0.25
    monkeypatch.setattr(yfinance_adapter.yf, "download", lambda **_: source)
    clock = FrozenClock(canonical.index[-1].to_pydatetime())
    bus = InProcessMessageBus()
    ingested: list[DataIngested] = []
    bus.subscribe_event(DataIngested, ingested.append)
    provider = YFinanceProvider(
        {},
        bus,
        DataQualityGate(load_settings().quality, bus, clock=clock),
        clock=clock,
    )

    result = provider.get_ohlcv(market_data_request(canonical))

    assert result.splits["AAPL"].loc[source.index[5]] == 2.0
    assert result.dividends["AAPL"].loc[source.index[10]] == 0.25
    assert len(ingested) == 1
    assert ingested[0].payload.data == result


def test_quality_gate_passes_clean_data_and_emits_report() -> None:
    frame = synthetic_ohlcv()
    bus = InProcessMessageBus()
    reports: list[QualityReportProduced] = []
    bus.subscribe_event(QualityReportProduced, reports.append)
    gate = DataQualityGate(load_settings().quality, bus)

    report = gate.validate(
        frame,
        "AAPL",
        now=frame.index[-1].to_pydatetime(),
    )

    assert report.passed is True
    assert report.rows == len(frame)
    assert reports[0].report == report


@pytest.mark.parametrize("corruption", ["nan", "flat", "outlier", "non_monotonic"])
def test_quality_gate_hard_fails_corrupted_data(corruption: str) -> None:
    frame = synthetic_ohlcv()
    if corruption == "nan":
        frame.iloc[20, frame.columns.get_loc("close")] = np.nan
    elif corruption == "flat":
        frame.iloc[30:35, frame.columns.get_loc("close")] = frame.iloc[29]["close"]
    elif corruption == "outlier":
        frame.iloc[60, frame.columns.get_loc("close")] *= 10.0
    else:
        order = [1, 0, *range(2, len(frame))]
        frame = frame.iloc[order]

    bus = InProcessMessageBus()
    failures: list[QualityGateFailed] = []
    bus.subscribe_event(QualityGateFailed, failures.append)
    gate = DataQualityGate(load_settings().quality, bus)

    with pytest.raises(DataQualityError):
        gate.validate(frame, "AAPL", now=frame.index.max().to_pydatetime())
    assert len(failures) == 1


def test_quality_gate_enforces_zscore_and_last_bar_age() -> None:
    frame = synthetic_ohlcv()
    frame.iloc[80, frame.columns.get_loc("close")] *= 1.2
    frame.iloc[80, frame.columns.get_loc("high")] = (
        frame.iloc[80]["close"] * 1.001
    )
    strict_zscore = QualitySettings(
        max_nan_ratio=0.0,
        max_abs_logret=1.0,
        stale_bars=5,
        zscore_cap=3.0,
        max_last_bar_age_days=7,
    )
    with pytest.raises(DataQualityError, match="z-score"):
        DataQualityGate(strict_zscore, InProcessMessageBus()).validate(
            frame,
            "AAPL",
            now=frame.index[-1].to_pydatetime(),
        )
    with pytest.raises(DataQualityError, match="stale"):
        DataQualityGate(load_settings().quality, InProcessMessageBus()).validate(
            synthetic_ohlcv(),
            "AAPL",
            now=(synthetic_ohlcv().index[-1] + pd.Timedelta(days=8)).to_pydatetime(),
        )


def test_corporate_action_adjustment_handles_split_and_dividend_exactly() -> None:
    index = pd.date_range("2025-01-01", periods=4, freq="D", tz="UTC")
    raw = pd.Series([100.0, 102.0, 51.0, 52.0], index=index)
    splits = pd.Series([0.0, 0.0, 2.0, 0.0], index=index)
    dividends = pd.Series(0.0, index=index)

    adjusted = adjust_corporate_actions(
        CorporateActionInput(raw_close=raw, splits=splits, dividends=dividends)
    )

    assert adjusted.split_adjusted.loc[index[1]] == pytest.approx(102.0)
    assert adjusted.split_adjusted.loc[index[2]] == pytest.approx(102.0)
    assert adjusted.raw_close.loc[index[0]] == adjusted.split_adjusted.loc[index[0]]

    dividend_raw = pd.Series([100.0, 99.0], index=index[:2])
    dividend = pd.Series([0.0, 1.0], index=index[:2])
    dividend_adjusted = adjust_corporate_actions(
        CorporateActionInput(
            raw_close=dividend_raw,
            splits=pd.Series(0.0, index=index[:2]),
            dividends=dividend,
        )
    )
    assert dividend_adjusted.as_of_adjusted.iloc[0] == pytest.approx(100.0)
    assert dividend_adjusted.as_of_adjusted.iloc[1] == pytest.approx(100.0)


def test_pit_join_never_selects_t_plus_one_sentinel() -> None:
    t0 = pd.Timestamp("2025-01-02", tz="UTC")
    features = pd.DataFrame({"timestamp": [t0], "symbol": ["AAPL"], "feature": [1.0]})
    targets = pd.DataFrame(
        {
            "timestamp": [t0 - pd.Timedelta(days=1), t0 + pd.Timedelta(days=1)],
            "symbol": ["AAPL", "AAPL"],
            "value": [7.0, 999_999.0],
        }
    )

    joined = pit_join(
        PointInTimeFrame(frame=features),
        PointInTimeFrame(frame=targets),
    ).frame

    assert joined.loc[0, "value"] == 7.0
    assert joined.loc[0, "target_timestamp"] <= joined.loc[0, "timestamp"]
    assert_no_lookahead(joined, "timestamp")
    leaked = joined.assign(source_timestamp=t0 + pd.Timedelta(days=1))
    with pytest.raises(ValueError, match="look-ahead"):
        assert_no_lookahead(leaked, "timestamp")


def test_pit_join_rejects_symbol_mismatch_and_duplicate_keys() -> None:
    timestamp = pd.Timestamp("2025-01-02", tz="UTC")
    symbol_features = pd.DataFrame(
        {"timestamp": [timestamp], "symbol": ["AAPL"], "feature": [1.0]}
    )
    symbol_free_targets = pd.DataFrame({"timestamp": [timestamp], "value": [2.0]})
    with pytest.raises(ValueError, match="symbol must be present"):
        pit_join(
            PointInTimeFrame(frame=symbol_features),
            PointInTimeFrame(frame=symbol_free_targets),
        )

    duplicate_targets = pd.concat((symbol_free_targets, symbol_free_targets))
    with pytest.raises(ValueError, match="keys must be unique"):
        pit_join(
            PointInTimeFrame(frame=symbol_features.drop(columns="symbol")),
            PointInTimeFrame(frame=duplicate_targets),
        )


def test_feature_pipeline_is_causal_and_produces_phase0_dtos() -> None:
    source = synthetic_ohlcv(80)
    bus = InProcessMessageBus()
    events: list[FeaturesComputed] = []
    bus.subscribe_event(FeaturesComputed, events.append)
    clock = FrozenClock(source.index[-1].to_pydatetime())
    pipeline = FeaturePipeline(bus, clock=clock)

    full = pipeline.compute(bar_frame(source))
    anchor = source.index[50]
    truncated = pipeline.compute(bar_frame(source.loc[:anchor]))

    pd.testing.assert_series_equal(
        full.loc[(anchor, "AAPL")],
        truncated.loc[(anchor, "AAPL")],
    )
    vectors = to_feature_vectors(full, feature_set_version="phase1")
    assert vectors
    assert vectors[-1].symbol == "AAPL"
    assert events[0].rows == len(source)
    assert tuple(full.columns) == FEATURE_COLUMNS
    assert tuple(str(dtype) for dtype in full.dtypes) == ("float64",) * len(FEATURE_COLUMNS)
    assert full.index.names == ["timestamp", "symbol"]
    assert full.index.is_monotonic_increasing


def test_feature_pipeline_explicitly_adjusts_a_split() -> None:
    index = pd.date_range("2025-01-01", periods=50, freq="D", tz="UTC")
    close = np.r_[np.full(30, 100.0), np.full(20, 50.0)]
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adj_close": close,
            "volume": np.full(len(index), 1_000.0),
        },
        index=index,
    )
    splits = pd.Series(0.0, index=index)
    splits.iloc[30] = 2.0
    bars = BarFrame(
        bars={"AAPL": frame},
        dividends={"AAPL": pd.Series(0.0, index=index)},
        splits={"AAPL": splits},
    )
    matrix = FeaturePipeline(InProcessMessageBus()).compute(bars)

    assert matrix.loc[(index[30], "AAPL"), "log_return"] == pytest.approx(0.0)
    assert matrix.loc[(index[30], "AAPL"), "momentum_20"] == pytest.approx(0.0)


def test_future_corporate_action_cannot_change_past_features_or_labels() -> None:
    source = synthetic_ohlcv()
    baseline = bar_frame(source)
    split_time = source.index[120]
    future_splits = pd.Series(0.0, index=source.index)
    future_splits.loc[split_time] = 2.0
    with_future_action = BarFrame(
        bars={"AAPL": source},
        dividends={"AAPL": pd.Series(0.0, index=source.index)},
        splits={"AAPL": future_splits},
    )
    clock = FrozenClock(source.index[-1].to_pydatetime())

    baseline_features = FeaturePipeline(
        InProcessMessageBus(),
        clock=clock,
    ).compute(baseline)
    future_features = FeaturePipeline(
        InProcessMessageBus(),
        clock=clock,
    ).compute(with_future_action)
    past_rows = baseline_features.index.get_level_values("timestamp") < split_time
    pd.testing.assert_frame_equal(
        baseline_features.loc[past_rows],
        future_features.loc[past_rows],
        check_exact=True,
    )

    settings = LabelSettings(
        vol_span=20,
        cusum_threshold=0.02,
        pt=1.0,
        sl=1.0,
        vertical_bars=5,
    )
    label_events = tuple(source.index[position] for position in (40, 70, 100, 130))
    baseline_labels = LabelPipeline(
        settings,
        InProcessMessageBus(),
        clock=clock,
    ).compute_from_bars(baseline, "AAPL", label_events)
    future_labels = LabelPipeline(
        settings,
        InProcessMessageBus(),
        clock=clock,
    ).compute_from_bars(with_future_action, "AAPL", label_events)

    assert tuple(sample for sample in baseline_labels if sample.t1 < split_time) == tuple(
        sample for sample in future_labels if sample.t1 < split_time
    )


def test_full_phase1_runs_are_deterministic_with_injected_runtime_providers() -> None:
    source = synthetic_ohlcv()
    bars = bar_frame(source)
    clock = FrozenClock(source.index[-1].to_pydatetime())
    settings = LabelSettings(
        vol_span=20,
        cusum_threshold=0.02,
        pt=1.0,
        sl=1.0,
        vertical_bars=5,
    )
    label_events = tuple(source.index[position] for position in (40, 70, 100))

    def run_once() -> tuple[
        pd.DataFrame,
        tuple[object, ...],
        tuple[object, ...],
        tuple[FeaturesComputed, ...],
        tuple[LabelsComputed, ...],
    ]:
        bus = InProcessMessageBus()
        feature_events: list[FeaturesComputed] = []
        label_output_events: list[LabelsComputed] = []
        bus.subscribe_event(FeaturesComputed, feature_events.append)
        bus.subscribe_event(LabelsComputed, label_output_events.append)
        matrix = FeaturePipeline(bus, clock=clock).compute(bars)
        vectors = to_feature_vectors(
            matrix,
            feature_set_version="phase1",
            id_provider=SeededIdProvider(SEED),
        )
        labels = LabelPipeline(settings, bus, clock=clock).compute_from_bars(
            bars,
            "AAPL",
            label_events,
        )
        return (
            matrix,
            vectors,
            labels,
            tuple(feature_events),
            tuple(label_output_events),
        )

    first = run_once()
    second = run_once()

    pd.testing.assert_frame_equal(first[0], second[0], check_exact=True)
    assert first[1:] == second[1:]
    assert [vector.feature_vector_id for vector in first[1]] == [
        vector.feature_vector_id for vector in second[1]
    ]
    assert all(event.timestamp == clock.now() for event in (*first[3], *first[4]))


def _barrier_result(
    prices: list[float],
    *,
    side: float | None = None,
) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=len(prices), freq="D", tz="UTC")
    close = pd.Series(prices, index=index)
    volatility = pd.Series(0.05, index=index)
    sides = None if side is None else pd.Series(side, index=index)
    return triple_barrier(
        close,
        [index[0]],
        pt=1.0,
        sl=1.0,
        vertical_bars=len(prices) - 1,
        side=sides,
        vol=volatility,
    )


def test_triple_barrier_directional_and_meta_labels() -> None:
    upper = _barrier_result([100.0, 106.0, 90.0])
    lower = _barrier_result([100.0, 94.0, 110.0])
    vertical = _barrier_result([100.0, 102.0, 101.0])

    assert upper.iloc[0]["label"] == 1
    assert upper.iloc[0]["t1"] == upper.iloc[0]["t0"] + pd.Timedelta(days=1)
    assert lower.iloc[0]["label"] == -1
    assert vertical.iloc[0]["label"] == 0
    assert _barrier_result([100.0, 106.0], side=1.0).iloc[0]["label"] == 1
    assert _barrier_result([100.0, 94.0], side=1.0).iloc[0]["label"] == 0
    assert set(_barrier_result([100.0, 94.0], side=-1.0)["label"]) <= {0, 1}


def test_label_pipeline_returns_dtos_and_emits_event() -> None:
    index = pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC")
    close = pd.Series([100.0, 106.0, 106.0, 100.0, 94.0], index=index)
    vol = pd.Series(0.05, index=index)
    settings = LabelSettings(
        vol_span=2,
        cusum_threshold=0.02,
        pt=1.0,
        sl=1.0,
        vertical_bars=1,
    )
    bus = InProcessMessageBus()
    events: list[LabelsComputed] = []
    bus.subscribe_event(LabelsComputed, events.append)

    samples = LabelPipeline(settings, bus).compute(
        close,
        (index[0], index[3]),
        vol=vol,
    )

    assert [sample.label for sample in samples] == [1, -1]
    assert all(sample.t1 >= sample.t0 and 0.0 < sample.weight <= 1.0 for sample in samples)
    assert events[0].samples == 2


def test_label_pipeline_uses_configured_cusum_when_events_are_omitted() -> None:
    index = pd.date_range("2025-01-01", periods=6, freq="D", tz="UTC")
    close = pd.Series([100.0, 100.1, 100.2, 110.0, 110.1, 110.2], index=index)
    settings = LabelSettings(
        vol_span=2,
        cusum_threshold=0.05,
        pt=10.0,
        sl=10.0,
        vertical_bars=2,
    )

    samples = LabelPipeline(settings, InProcessMessageBus()).compute(close)

    assert len(samples) == 1
    assert samples[0].t0 == index[3].to_pydatetime()
    assert samples[0].t1 == index[5].to_pydatetime()


def test_cusum_detects_injected_step_and_ignores_flat_series() -> None:
    index = pd.date_range("2025-01-01", periods=8, freq="D", tz="UTC")
    stepped = pd.Series([100.0, 100.0, 100.0, 110.0, 110.0, 110.0, 110.0, 110.0], index=index)
    flat = pd.Series(100.0, index=index)

    events = cusum_filter(stepped, threshold=0.05)

    assert list(events) == [index[3]]
    assert cusum_filter(flat, threshold=0.05).empty


def test_fixed_width_fracdiff_stationarity_memory_and_integer_limits() -> None:
    rng = np.random.default_rng(SEED)
    index = pd.date_range("2010-01-01", periods=2_000, freq="D", tz="UTC")
    level = pd.Series(100.0 + np.cumsum(rng.normal(0.0, 1.0, len(index))), index=index)

    differentiated = frac_diff_ffd(level, d=0.6, thresh=1e-3).dropna()
    pvalue = adfuller(differentiated.to_numpy(), maxlag=1, autolag=None)[1]
    correlation = differentiated.corr(level.loc[differentiated.index])

    assert pvalue < 0.05
    assert abs(correlation) > 0.5
    pd.testing.assert_series_equal(frac_diff_ffd(level, d=0.0, thresh=1e-3), level)
    pd.testing.assert_series_equal(
        frac_diff_ffd(level, d=1.0, thresh=1e-3),
        level.diff(),
    )


def test_fracdiff_config_and_minimum_d_helper() -> None:
    rng = np.random.default_rng(SEED)
    stationary = pd.Series(rng.normal(size=500))
    settings = load_settings().fracdiff

    pd.testing.assert_series_equal(
        configured_frac_diff(stationary, settings),
        frac_diff_ffd(stationary, settings.d, settings.thresh),
    )
    assert min_ffd_d(stationary, thresh=1e-2, step=0.2) == 0.0


def test_sample_weights_penalize_overlap_only() -> None:
    index = pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC")
    overlapping = pd.Series([index[2], index[3]], index=index[:2])
    separate = pd.Series([index[1], index[4]], index=[index[0], index[3]])

    overlap_weights = sample_weights(overlapping)
    separate_weights = sample_weights(separate)

    assert (overlap_weights < 1.0).all()
    assert (separate_weights == 1.0).all()


def test_sample_weights_use_every_observation_in_long_intervals() -> None:
    index = pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC")
    events = pd.Series([index[4], index[4]], index=[index[0], index[2]])

    weights = sample_weights(events)

    assert weights.iloc[0] == pytest.approx(0.7)
    assert weights.iloc[1] == pytest.approx(0.5)


class _FakeProvider(MarketDataProvider):
    def __init__(self, result: BarFrame | Exception) -> None:
        self.result = result
        self.calls = 0

    def get_ohlcv(
        self,
        request: MarketDataRequest,
    ) -> BarFrame:
        del request
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_provider_fallback_and_exhaustion_are_explicit() -> None:
    primary = _FakeProvider(DataUnavailableError("primary unavailable"))
    secondary = _FakeProvider(bar_frame())
    chain = ProviderChain((primary, secondary), max_attempts=1)
    request = market_data_request(synthetic_ohlcv())

    result = chain.get_ohlcv(request)

    assert result is secondary.result
    assert primary.calls == secondary.calls == 1

    both_fail = ProviderChain(
        (
            _FakeProvider(DataUnavailableError("primary unavailable")),
            _FakeProvider(DataUnavailableError("secondary unavailable")),
        ),
        max_attempts=2,
    )
    with pytest.raises(DataUnavailableError, match="all market-data providers failed"):
        both_fail.get_ohlcv(request)


def test_provider_chain_rejects_partial_symbol_responses() -> None:
    request = market_data_request(synthetic_ohlcv(), ("AAPL", "MSFT"))
    chain = ProviderChain((_FakeProvider(bar_frame()),), max_attempts=1)

    with pytest.raises(DataUnavailableError, match="missing=\\['MSFT'\\]"):
        chain.get_ohlcv(request)


def test_ingest_and_corporate_action_boundaries_reject_wrong_types() -> None:
    index = pd.date_range("2025-01-01", periods=2, freq="D", tz="UTC")
    with pytest.raises(ValidationError):
        MarketDataRequest.model_validate(
            {
                "symbols": ["AAPL"],
                "start": index[0].to_pydatetime(),
                "end": index[1].to_pydatetime(),
                "frequency": "1d",
            }
        )
    with pytest.raises(ValidationError):
        CorporateActionInput.model_validate(
            {
                "raw_close": [100.0, 101.0],
                "splits": pd.Series(0.0, index=index),
                "dividends": pd.Series(0.0, index=index),
            }
        )
    with pytest.raises(ValidationError):
        PointInTimeFrame.model_validate({"frame": "not-a-dataframe"})


def test_phase0_data_vendor_port_is_preserved_by_compatibility_adapter() -> None:
    adapter = Phase0DataVendorAdapter(_FakeProvider(bar_frame()))
    request = OHLCVRequest(
        symbol="AAPL",
        start=datetime(2025, 1, 1, tzinfo=UTC),
        end=datetime(2025, 12, 31, tzinfo=UTC),
    )

    bars = adapter.get_ohlcv(request)

    assert bars
    assert all(isinstance(item, OHLCVBar) for item in bars)


def test_phase1_config_hard_fails_reversed_date_range(tmp_path: Path) -> None:
    config = Path("src/aihedgefund/config/limits.yaml").read_text(encoding="utf-8")
    invalid = tmp_path / "invalid_phase1.yaml"
    invalid.write_text(
        config.replace("end: 2026-01-01", "end: 2014-01-01"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="end must be later"):
        load_settings(invalid)
