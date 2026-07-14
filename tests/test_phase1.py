"""Offline, deterministic definition-of-done tests for Phase 1."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel
from statsmodels.tsa.stattools import adfuller

import aihedgefund.data.adapters.yfinance as yfinance_adapter
from aihedgefund.core.bus import InProcessMessageBus
from aihedgefund.core.config import LabelSettings, QualitySettings, load_settings
from aihedgefund.core.schemas import (
    BarFrame,
    DataIngested,
    FeaturesComputed,
    LabelsComputed,
    OHLCVBar,
    OHLCVRequest,
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
    FeatureParameters,
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
    provider = YFinanceProvider({"APPLE": "AAPL"}, bus)

    result = provider.get_ohlcv(
        ("APPLE",),
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2025, 2, 1, tzinfo=UTC),
        "1d",
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

    adjusted = adjust_corporate_actions(raw, splits, dividends)

    assert adjusted.loc[index[1], "split_adjusted"] == pytest.approx(51.0)
    assert adjusted.loc[index[2], "split_adjusted"] == pytest.approx(51.0)
    assert raw.iloc[0] / adjusted["split_adjusted"].iloc[0] == pytest.approx(2.0)

    dividend_raw = pd.Series([100.0, 99.0], index=index[:2])
    dividend = pd.Series([0.0, 1.0], index=index[:2])
    dividend_adjusted = adjust_corporate_actions(
        dividend_raw,
        pd.Series(0.0, index=index[:2]),
        dividend,
    )
    assert dividend_adjusted["total_return_adjusted"].iloc[0] == pytest.approx(99.0)
    assert dividend_adjusted["total_return_adjusted"].iloc[1] == pytest.approx(99.0)


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

    joined = pit_join(features, targets)

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
        pit_join(symbol_features, symbol_free_targets)

    duplicate_targets = pd.concat((symbol_free_targets, symbol_free_targets))
    with pytest.raises(ValueError, match="keys must be unique"):
        pit_join(symbol_features.drop(columns="symbol"), duplicate_targets)


def test_feature_pipeline_is_causal_and_produces_phase0_dtos() -> None:
    source = synthetic_ohlcv(80)
    parameters = FeatureParameters(
        volatility_span=5,
        momentum_periods=3,
        moving_average_window=5,
        rsi_period=5,
        macd_fast=3,
        macd_slow=6,
        macd_signal=3,
        atr_period=5,
        zscore_window=5,
    )
    bus = InProcessMessageBus()
    events: list[FeaturesComputed] = []
    bus.subscribe_event(FeaturesComputed, events.append)
    pipeline = FeaturePipeline(bus, parameters)

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


def test_feature_pipeline_explicitly_adjusts_a_split() -> None:
    index = pd.date_range("2025-01-01", periods=8, freq="D", tz="UTC")
    close = np.array([100.0, 100.0, 100.0, 100.0, 50.0, 50.0, 50.0, 50.0])
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
    splits = pd.Series([0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0], index=index)
    bars = BarFrame(
        bars={"AAPL": frame},
        dividends={"AAPL": pd.Series(0.0, index=index)},
        splits={"AAPL": splits},
    )
    parameters = FeatureParameters(
        volatility_span=2,
        momentum_periods=1,
        moving_average_window=2,
        rsi_period=2,
        macd_fast=1,
        macd_slow=2,
        macd_signal=1,
        atr_period=2,
        zscore_window=2,
    )

    matrix = FeaturePipeline(InProcessMessageBus(), parameters).compute(bars)

    assert matrix.loc[(index[4], "AAPL"), "log_return"] == pytest.approx(0.0)
    assert matrix.loc[(index[4], "AAPL"), "momentum_1"] == pytest.approx(0.0)


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
        symbols: tuple[str, ...],
        start: datetime,
        end: datetime,
        frequency: str,
    ) -> BarFrame:
        del symbols, start, end, frequency
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_provider_fallback_and_exhaustion_are_explicit() -> None:
    primary = _FakeProvider(DataUnavailableError("primary unavailable"))
    secondary = _FakeProvider(bar_frame())
    chain = ProviderChain((primary, secondary), max_attempts=1)

    result = chain.get_ohlcv(
        ("AAPL",),
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2025, 12, 31, tzinfo=UTC),
        "1d",
    )

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
        both_fail.get_ohlcv(
            ("AAPL",),
            datetime(2025, 1, 1, tzinfo=UTC),
            datetime(2025, 12, 31, tzinfo=UTC),
            "1d",
        )


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
