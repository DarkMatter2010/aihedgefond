"""Manual live insider Form 4 IC triage; excluded from pytest/CI (Yahoo + SEC).

Pre-registered single run (no post-hoc retune after seeing results).
Universum: BROAD_LIQUID_CANDIDATE_UNIVERSE. Seed/split/HPs from limits.yaml.
Horizons: h=2 and h=21. No CPCV / Gate — IC triage only.
SEC: official data.sec.gov + Archives XML, rate-limited, disk-cached.
"""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from datetime import time as dt_time
from typing import Final, Literal

import pandas as pd

from aihedgefund.core.bus import InProcessMessageBus
from aihedgefund.core.config import load_settings
from aihedgefund.core.runtime import FrozenClock
from aihedgefund.core.schemas import BarFrame, Form4Request, MarketDataRequest
from aihedgefund.data.adapters import SecEdgarForm4Provider, YFinanceProvider
from aihedgefund.data.form4_quality import Form4QualityGate
from aihedgefund.data.provider import DataUnavailableError
from aihedgefund.data.quality import DataQualityError, DataQualityGate
from aihedgefund.features.insider import INSIDER_FEATURE_CLASS_CONFIGS, INSIDER_PIT_NOTE
from aihedgefund.features.pipeline import FEATURE_COLUMNS
from aihedgefund.research.feature_class_triage import (
    RANK_IC_MATERIAL_THRESHOLD,
    TRIAGE_HORIZONS,
)
from aihedgefund.research.insider_triage import run_insider_form4_triage
from aihedgefund.research.universes import (
    BROAD_LIQUID_CANDIDATE_UNIVERSE,
    SURVIVORSHIP_BIAS_NOTE,
)

DOWNLOAD_BATCH_SIZE: Final[int] = 50
BATCH_PAUSE_SEC: Final[float] = 1.5


def _enable_insecure_yfinance_ssl_if_needed() -> None:
    """Opt-in workaround: curl_cffi ignores the Windows trust store (MITM/corp SSL)."""
    if os.environ.get("AIHF_ALLOW_INSECURE_YFINANCE_SSL", "1") != "1":
        return
    try:
        from curl_cffi.requests import Session
    except ImportError:
        return
    if getattr(Session.request, "_aihf_insecure", False):
        return
    original = Session.request

    def _request(self: object, *args: object, **kwargs: object) -> object:
        kwargs["verify"] = False
        return original(self, *args, **kwargs)

    _request._aihf_insecure = True  # type: ignore[attr-defined]
    Session.request = _request  # type: ignore[method-assign]
    print("note: AIHF_ALLOW_INSECURE_YFINANCE_SSL=1 (curl_cffi verify=False)")


def _download_broad_universe(
    symbols: tuple[str, ...],
    *,
    provider: YFinanceProvider,
    request_start: datetime,
    request_end: datetime,
    frequency: Literal["1d"],
) -> tuple[BarFrame, tuple[str, ...], tuple[str, ...]]:
    """Batch Yahoo fetch with per-symbol fallback; no continuity re-filter."""
    collected_bars: dict[str, pd.DataFrame] = {}
    collected_divs: dict[str, pd.Series] = {}
    collected_splits: dict[str, pd.Series] = {}
    dropped: list[str] = []

    for offset in range(0, len(symbols), DOWNLOAD_BATCH_SIZE):
        batch = symbols[offset : offset + DOWNLOAD_BATCH_SIZE]
        try:
            batch_bars = provider.get_ohlcv(
                MarketDataRequest(
                    symbols=batch,
                    start=request_start,
                    end=request_end,
                    frequency=frequency,
                )
            )
        except (DataUnavailableError, DataQualityError, ValueError) as exc:
            print(f"batch {batch[0]}.. failed ({exc}); retrying per symbol")
            for symbol in batch:
                try:
                    one = provider.get_ohlcv(
                        MarketDataRequest(
                            symbols=(symbol,),
                            start=request_start,
                            end=request_end,
                            frequency=frequency,
                        )
                    )
                    collected_bars[symbol] = one.bars[symbol]
                    collected_divs[symbol] = one.dividends[symbol]
                    collected_splits[symbol] = one.splits[symbol]
                except (DataUnavailableError, DataQualityError, ValueError) as symbol_exc:
                    dropped.append(symbol)
                    print(f"  drop {symbol}: {symbol_exc}")
                time.sleep(0.2)
            time.sleep(BATCH_PAUSE_SEC)
            continue

        for symbol in batch:
            collected_bars[symbol] = batch_bars.bars[symbol]
            collected_divs[symbol] = batch_bars.dividends[symbol]
            collected_splits[symbol] = batch_bars.splits[symbol]
        print(f"batch ok {batch[0]}..{batch[-1]} (kept so far {len(collected_bars)})")
        time.sleep(BATCH_PAUSE_SEC)

    if len(collected_bars) < 30:
        msg = f"too few symbols after download: {len(collected_bars)}"
        raise RuntimeError(msg)

    kept = tuple(sorted(collected_bars))
    bars = BarFrame(
        bars={s: collected_bars[s] for s in kept},
        dividends={s: collected_divs[s] for s in kept},
        splits={s: collected_splits[s] for s in kept},
    )
    return bars, kept, tuple(sorted(set(dropped)))


def main() -> None:
    """Fetch broad universe + Form 4, run insider IC triage, print report."""
    _enable_insecure_yfinance_ssl_if_needed()
    base = load_settings()
    settings = base.model_copy(update={"universe": BROAD_LIQUID_CANDIDATE_UNIVERSE})
    request_start = datetime.combine(settings.start, dt_time.min, tzinfo=UTC)
    request_end = datetime.combine(settings.end, dt_time.min, tzinfo=UTC)
    clock = FrozenClock(request_end)
    bus = InProcessMessageBus()
    provider = YFinanceProvider(
        settings.symbol_aliases,
        bus,
        DataQualityGate(settings.quality, bus, clock=clock),
        clock=clock,
    )
    # Live triage uses filings.recent only (fast). Full multi-file history is
    # available via include_historical_files=True on the adapter.
    form4_provider = SecEdgarForm4Provider(
        settings.edgar,
        bus,
        Form4QualityGate(bus, clock=clock),
        clock=clock,
        symbol_aliases=settings.symbol_aliases,
        include_historical_files=False,
        skip_uncached_filings=True,
    )

    print("=== Insider Form 4 IC triage ===")
    print(f"requested_n: {len(BROAD_LIQUID_CANDIDATE_UNIVERSE)}")
    print(f"horizons: {TRIAGE_HORIZONS}")
    print(f"seed: {settings.research.seed}")
    print(
        f"train_end / configured_test_start: "
        f"{settings.research.train_end} / {settings.research.test_start}"
    )
    print(f"production FEATURE_COLUMNS: {len(FEATURE_COLUMNS)} (unchanged)")
    print(f"configs: {[label for label, _ in INSIDER_FEATURE_CLASS_CONFIGS]}")
    print(f"threshold Rank-IC: {RANK_IC_MATERIAL_THRESHOLD}")
    print(f"survivorship: {SURVIVORSHIP_BIAS_NOTE}")
    print(f"pit: {INSIDER_PIT_NOTE}")
    print(f"edgar user-agent: {settings.edgar.user_agent}")
    print(f"edgar cache: {settings.edgar.cache_dir} max_rps={settings.edgar.max_rps}")

    bars, kept, dropped = _download_broad_universe(
        BROAD_LIQUID_CANDIDATE_UNIVERSE,
        provider=provider,
        request_start=request_start,
        request_end=request_end,
        frequency=settings.frequency,
    )
    print(f"kept_n: {len(kept)}  dropped_n: {len(dropped)}")

    print("fetching Form 4 filings (cached, rate-limited)...")
    form4 = form4_provider.get_form4(
        Form4Request(symbols=kept, start=request_start, end=request_end)
    )
    print(
        f"form4 records: {len(form4.records)}  "
        f"symbols_without_filings: {len(form4.symbols_without_filings)}"
    )

    run_settings = settings.model_copy(update={"universe": kept})
    report = run_insider_form4_triage(bars, form4, run_settings)

    print("--- OOS metrics (config x horizon) ---")
    print(
        "class | h | test_start | n_feat | N_dates | median_breadth | "
        "Rank-IC | IC | ICIR | GrinoldSR | vs 0.02"
    )
    for row in report.rows:
        m = row.metrics
        icir_s = "None" if m.icir is None else f"{m.icir:.6f}"
        flag = ">=" if row.rank_ic_above_threshold else "<"
        print(
            f"{row.class_label} | {row.horizon} | {row.test_start} | "
            f"{row.feature_column_count} | {m.n_dates} | {m.median_cs_breadth:.1f} | "
            f"{m.rank_ic_mean:.6f} | {m.ic_mean:.6f} | {icir_s} | "
            f"{row.grinold_sr:.6f} | {flag} threshold"
        )

    print(
        f"best: {report.best_class_label} h={report.best_horizon} "
        f"Rank-IC={report.best_rank_ic:.6f}"
    )
    print(
        f"prior best all_new Rank-IC: h2={report.prior_best_all_new_rank_ic_h2} "
        f"h21={report.prior_best_all_new_rank_ic_h21}"
    )
    print(f"interpretation: {report.interpretation}")
    print(f"note: {report.interpretation_note}")
    print(f"counts_as_research_trial: {report.counts_as_research_trial}")
    print(f"n_configs_measured: {report.n_configs_measured}")
    print("=== HANDOFF ===")
    print("modules: aihedgefund.features.insider, aihedgefund.research.insider_triage")
    print("script: scripts/run_insider_form4_triage.py")
    print("next: gate if candidate_for_gate else free_data_levers_exhausted / paid data")


if __name__ == "__main__":
    main()
