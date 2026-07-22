"""Offline SEC Form 4 adapter + insider feature PIT tests (no network)."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aihedgefund.core.bus import InProcessMessageBus
from aihedgefund.core.config import EdgarSettings, load_settings
from aihedgefund.core.runtime import FrozenClock
from aihedgefund.core.schemas import (
    BarFrame,
    Form4Frame,
    Form4Record,
    Form4Request,
)
from aihedgefund.data.adapters.sec_edgar import (
    SecEdgarForm4Provider,
    parse_form4_xml,
)
from aihedgefund.data.form4_quality import Form4QualityError, Form4QualityGate
from aihedgefund.data.provider import DataUnavailableError
from aihedgefund.features.insider import (
    INSIDER_FEATURE_CLASS_CONFIGS,
    INSIDER_FEATURE_COLUMNS,
    INSIDER_RAW_FEATURE_COLUMNS,
    build_insider_feature_matrix,
    compute_symbol_insider_raws,
)
from aihedgefund.features.pipeline import FEATURE_COLUMNS
from aihedgefund.research.insider_triage import (
    interpret_insider_triage,
    run_insider_form4_triage,
)
from aihedgefund.research.research_trials import N_RESEARCH_TRIALS, RESEARCH_TRIAL_SHARPES
from aihedgefund.research.universes import BROAD_LIQUID_CANDIDATE_UNIVERSE

CLOCK = datetime(2024, 6, 1, tzinfo=UTC)

FORM4_XML = b"""<?xml version="1.0"?>
<ownershipDocument>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerCik>0001234567</rptOwnerCik>
      <rptOwnerName>ALICE BUYER</rptOwnerName>
    </reportingOwnerId>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2023-03-01</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>1000</value></transactionShares>
        <transactionPricePerShare><value>10.0</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <transactionDate><value>2023-03-02</value></transactionDate>
      <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>400</value></transactionShares>
        <transactionPricePerShare><value>11.0</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>
"""


def _edgar_settings(tmp_path: Path) -> EdgarSettings:
    return EdgarSettings(
        user_agent="AIHedgeFond Test test@example.com",
        cache_dir=tmp_path / "sec_cache",
        max_rps=8.0,
    )


def test_parse_form4_xml_buy_sell() -> None:
    filed = datetime(2023, 3, 3, 16, 0, tzinfo=UTC)
    records = parse_form4_xml(
        FORM4_XML,
        symbol="AAA",
        cik="0000320193",
        accession="0000320193-23-000001",
        filed_at=filed,
    )
    assert len(records) == 2
    assert records[0].acquired_disposed == "A"
    assert records[0].shares == 1000.0
    assert records[0].transaction_code == "P"
    assert records[0].filed_at == filed
    assert records[0].transaction_date == date(2023, 3, 1)
    assert records[1].acquired_disposed == "D"
    assert records[1].shares == 400.0


def test_quality_missing_activity_ok_future_filed_fails() -> None:
    bus = InProcessMessageBus()
    gate = Form4QualityGate(bus, clock=FrozenClock(CLOCK))
    empty = Form4Frame(records=(), symbols_queried=("AAA",), symbols_without_filings=("AAA",))
    assert gate.validate(empty).records == ()

    bad = Form4Frame(
        records=(
            Form4Record(
                symbol="AAA",
                cik="0000320193",
                accession="0001",
                filed_at=datetime(2025, 1, 1, tzinfo=UTC),
                transaction_date=date(2024, 12, 30),
                transaction_code="P",
                shares=10.0,
                price=1.0,
                acquired_disposed="A",
            ),
        ),
        symbols_queried=("AAA",),
    )
    with pytest.raises(Form4QualityError, match="after quality clock"):
        gate.validate(bad)


def test_adapter_uses_fixture_http_and_cache(tmp_path: Path) -> None:
    settings = _edgar_settings(tmp_path)
    tickers = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple"}}
    submissions = {
        "filings": {
            "recent": {
                "form": ["4"],
                "accessionNumber": ["0000320193-23-000001"],
                "acceptanceDateTime": ["2023-03-03 16:00:00"],
                "primaryDocument": ["primary.xml"],
            }
        }
    }
    urls: list[str] = []

    def http_get(url: str) -> bytes:
        urls.append(url)
        if "company_tickers" in url:
            return json.dumps(tickers).encode()
        if "submissions" in url:
            return json.dumps(submissions).encode()
        if "Archives" in url:
            return FORM4_XML
        raise AssertionError(f"unexpected url {url}")

    bus = InProcessMessageBus()
    provider = SecEdgarForm4Provider(
        settings,
        bus,
        Form4QualityGate(bus, clock=FrozenClock(CLOCK)),
        clock=FrozenClock(CLOCK),
        http_get=http_get,
    )
    request = Form4Request(
        symbols=("AAPL",),
        start=datetime(2023, 1, 1, tzinfo=UTC),
        end=datetime(2023, 12, 31, tzinfo=UTC),
    )
    first = provider.get_form4(request)
    assert len(first.records) == 2
    n_first = len(urls)
    second = provider.get_form4(request)
    assert len(second.records) == 2
    assert len(urls) == n_first  # full cache hit, no extra HTTP


def test_adapter_source_down_hard_fails(tmp_path: Path) -> None:
    settings = _edgar_settings(tmp_path)

    def http_get(url: str) -> bytes:
        raise DataUnavailableError("SEC unreachable")

    bus = InProcessMessageBus()
    provider = SecEdgarForm4Provider(
        settings,
        bus,
        Form4QualityGate(bus, clock=FrozenClock(CLOCK)),
        clock=FrozenClock(CLOCK),
        http_get=http_get,
    )
    with pytest.raises(DataUnavailableError, match="unreachable"):
        provider.get_form4(
            Form4Request(
                symbols=("AAPL",),
                start=datetime(2023, 1, 1, tzinfo=UTC),
                end=datetime(2023, 12, 31, tzinfo=UTC),
            )
        )


def test_insider_feature_pit_no_lookahead() -> None:
    index = pd.date_range("2023-03-01", periods=40, freq="B", tz="UTC")
    early = Form4Record(
        symbol="AAA",
        cik="1",
        accession="a1",
        filed_at=datetime(2023, 3, 10, 15, 0, tzinfo=UTC),
        transaction_date=date(2023, 3, 8),
        transaction_code="P",
        shares=1000.0,
        price=10.0,
        acquired_disposed="A",
        reporting_owner="ALICE",
    )
    late = Form4Record(
        symbol="AAA",
        cik="1",
        accession="a2",
        filed_at=datetime(2023, 4, 20, 15, 0, tzinfo=UTC),
        transaction_date=date(2023, 4, 18),
        transaction_code="P",
        shares=5000.0,
        price=10.0,
        acquired_disposed="A",
        reporting_owner="BOB",
    )
    t = index[15]
    base = compute_symbol_insider_raws(
        index,
        pd.DataFrame(
            [
                {
                    "symbol": early.symbol,
                    "filed_at": early.filed_at,
                    "transaction_date": early.transaction_date,
                    "transaction_code": early.transaction_code,
                    "shares": early.shares,
                    "acquired_disposed": early.acquired_disposed,
                    "reporting_owner": early.reporting_owner,
                    "accession": early.accession,
                }
            ]
        ),
    )
    with_future = compute_symbol_insider_raws(
        index,
        pd.DataFrame(
            [
                {
                    "symbol": r.symbol,
                    "filed_at": r.filed_at,
                    "transaction_date": r.transaction_date,
                    "transaction_code": r.transaction_code,
                    "shares": r.shares,
                    "acquired_disposed": r.acquired_disposed,
                    "reporting_owner": r.reporting_owner,
                    "accession": r.accession,
                }
                for r in (early, late)
            ]
        ),
    )
    # At t before late.filed_at, features must match early-only.
    assert t < late.filed_at
    pd.testing.assert_series_equal(base.loc[t], with_future.loc[t], check_names=False)


def test_insider_truncation_pit_matrix() -> None:
    symbols = BROAD_LIQUID_CANDIDATE_UNIVERSE[:4]
    frames = {}
    for i, symbol in enumerate(symbols):
        rng = np.random.default_rng(100 + i)
        index = pd.date_range("2021-01-04", periods=120, freq="B", tz="UTC")
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, 120)))
        frames[symbol] = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "adj_close": close,
                "volume": rng.integers(1e6, 2e6, 120).astype(float),
            },
            index=index,
        )
    empty = {s: pd.Series(0.0, index=frames[s].index) for s in symbols}
    bars = BarFrame(bars=frames, dividends=empty, splits=empty)
    records = []
    for symbol in symbols:
        records.append(
            Form4Record(
                symbol=symbol,
                cik="1",
                accession=f"{symbol}-1",
                filed_at=datetime(2021, 3, 1, 16, 0, tzinfo=UTC),
                transaction_date=date(2021, 2, 28),
                transaction_code="P",
                shares=100.0,
                price=1.0,
                acquired_disposed="A",
                reporting_owner="X",
            )
        )
    form4 = Form4Frame(records=tuple(records), symbols_queried=symbols)
    full = build_insider_feature_matrix(bars, form4)
    cut = datetime(2021, 6, 1, tzinfo=UTC)
    truncated_bars = BarFrame(
        bars={s: f.loc[:cut] for s, f in frames.items()},
        dividends={s: empty[s].loc[:cut] for s in symbols},
        splits={s: empty[s].loc[:cut] for s in symbols},
    )
    trunc = build_insider_feature_matrix(truncated_bars, form4)
    shared = trunc.index.intersection(full.index)
    pd.testing.assert_frame_equal(trunc.loc[shared], full.loc[shared])


def test_insider_registry_and_trials_headcount() -> None:
    assert len(FEATURE_COLUMNS) == 36
    assert set(INSIDER_FEATURE_COLUMNS).isdisjoint(FEATURE_COLUMNS)
    assert INSIDER_RAW_FEATURE_COLUMNS == (
        "insider_net_buy_ratio_21",
        "insider_buyer_count_21",
        "insider_signed_volume_21",
    )
    labels = [label for label, _ in INSIDER_FEATURE_CLASS_CONFIGS]
    assert labels == ["insider", "insider_plus_all_new"]
    assert N_RESEARCH_TRIALS == 29
    assert len(RESEARCH_TRIAL_SHARPES) == 29


def test_interpret_insider_exhausted_vs_candidate() -> None:
    from aihedgefund.core.schemas import ICMetricsReport
    from aihedgefund.research.insider_triage import InsiderTriageRow

    def _row(label: str, h: int, ric: float) -> InsiderTriageRow:
        return InsiderTriageRow(
            class_label=label,
            horizon=h,
            feature_column_count=9,
            metrics=ICMetricsReport(
                ic_mean=ric,
                rank_ic_mean=ric,
                icir=None,
                rank_icir=None,
                median_cs_breadth=50.0,
                cs_breadth_warning=False,
                ic_materially_positive=ric >= 0.02,
                ic_positive_threshold=0.02,
                n_dates=10,
            ),
            grinold_sr=ric * (50.0**0.5),
            rank_ic_above_threshold=ric >= 0.02,
            test_start=date(2023, 1, 10),
        )

    interp, _ = interpret_insider_triage((_row("insider", 2, 0.01), _row("insider", 21, 0.005)))
    assert interp == "free_data_levers_exhausted"
    interp2, _ = interpret_insider_triage((_row("insider", 2, 0.03), _row("insider", 21, 0.01)))
    assert interp2 == "candidate_for_gate"


def test_run_insider_triage_offline_deterministic() -> None:
    symbols = BROAD_LIQUID_CANDIDATE_UNIVERSE[:8]
    frames = {}
    n_bars = 900
    for i, symbol in enumerate(symbols):
        rng = np.random.default_rng(42 + i)
        index = pd.date_range("2021-01-04", periods=n_bars, freq="B", tz="UTC")
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n_bars)))
        open_ = np.r_[close[0], close[:-1]]
        frames[symbol] = pd.DataFrame(
            {
                "open": open_,
                "high": np.maximum(open_, close) * 1.002,
                "low": np.minimum(open_, close) * 0.998,
                "close": close,
                "adj_close": close,
                "volume": rng.integers(1e6, 3e6, n_bars).astype(float),
            },
            index=index,
        )
    empty = {s: pd.Series(0.0, index=frames[s].index) for s in symbols}
    bars = BarFrame(bars=frames, dividends=empty, splits=empty)
    records = []
    # Dense Form-4 activity so lookback windows are non-NaN after assemble dropna.
    from datetime import timedelta

    for i, symbol in enumerate(symbols):
        for week in range(0, 220, 2):
            filed = datetime(2021, 1, 4, 16, 0, tzinfo=UTC) + timedelta(weeks=week)
            if filed.year > 2025:
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
    form4 = Form4Frame(
        records=tuple(records),
        symbols_queried=symbols,
        symbols_without_filings=(),
    )
    settings = load_settings().model_copy(update={"universe": symbols})
    first = run_insider_form4_triage(bars, form4, settings)
    second = run_insider_form4_triage(bars, form4, settings)
    assert first.n_configs_measured == 4
    assert len(first.rows) == 4
    assert first.model_dump() == second.model_dump()


def test_adapter_resolves_xsl_primary_to_form4_xml(tmp_path: Path) -> None:
    settings = _edgar_settings(tmp_path)
    tickers = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple"}}
    submissions = {
        "filings": {
            "recent": {
                "form": ["4"],
                "accessionNumber": ["0000320193-23-000001"],
                "acceptanceDateTime": ["2023-03-03 16:00:00"],
                "primaryDocument": ["xslF345X05/form4.xml"],
            }
        }
    }
    urls: list[str] = []

    def http_get(url: str) -> bytes:
        urls.append(url)
        if "company_tickers" in url:
            return json.dumps(tickers).encode()
        if "submissions" in url:
            return json.dumps(submissions).encode()
        if url.endswith("/form4.xml") and "xsl" not in url.split("/")[-2:-1]:
            return FORM4_XML
        if "xslF345" in url:
            return b"<!DOCTYPE html><html></html>"
        raise AssertionError(f"unexpected url {url}")

    bus = InProcessMessageBus()
    provider = SecEdgarForm4Provider(
        settings,
        bus,
        Form4QualityGate(bus, clock=FrozenClock(CLOCK)),
        clock=FrozenClock(CLOCK),
        http_get=http_get,
    )
    frame = provider.get_form4(
        Form4Request(
            symbols=("AAPL",),
            start=datetime(2023, 1, 1, tzinfo=UTC),
            end=datetime(2023, 12, 31, tzinfo=UTC),
        )
    )
    assert len(frame.records) == 2
    assert any(u.endswith("/form4.xml") and "xsl" not in u for u in urls)
