"""Manual live universe-breadth diagnostic; excluded from pytest/CI (Yahoo).

Single controlled change vs the 50-name Phase-2 baseline: replace the universe
with ``BROAD_LIQUID_CANDIDATE_UNIVERSE``. Horizon forced to h=2 to match the
documented Rank-IC 0.0176 reference; seed/split/HPs come from limits.yaml.
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
from aihedgefund.core.schemas import BarFrame, MarketDataRequest
from aihedgefund.data.adapters import YFinanceProvider
from aihedgefund.data.provider import DataUnavailableError
from aihedgefund.data.quality import DataQualityError, DataQualityGate
from aihedgefund.features.pipeline import FEATURE_COLUMNS, FeaturePipeline
from aihedgefund.research.adapters.filesystem import FilesystemModelArtifactAdapter
from aihedgefund.research.universe_breadth_diagnostic import (
    DIAGNOSTIC_HORIZON,
    NARROW_H2_RANK_IC,
    NARROW_UNIVERSE_N,
    RANK_IC_MATERIAL_THRESHOLD,
    run_universe_breadth_diagnostic,
    settings_for_breadth_diagnostic,
)
from aihedgefund.research.universes import (
    BROAD_LIQUID_CANDIDATE_UNIVERSE,
    SURVIVORSHIP_BIAS_NOTE,
)

DOWNLOAD_BATCH_SIZE: Final[int] = 50
BATCH_PAUSE_SEC: Final[float] = 1.5


def _enable_insecure_yfinance_ssl_if_needed() -> None:
    """Opt-in workaround: curl_cffi ignores the Windows trust store (MITM/corp SSL).

    Set ``AIHF_ALLOW_INSECURE_YFINANCE_SSL=0`` to disable. Default on for this
    manual script only — never used by CI/unit tests.
    """
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
    """Batch Yahoo fetch with per-symbol fallback; no continuity re-filter.

    Continuity / calendar surgery would be a second methodological change.
    Only symbols that pass the existing quality gate are kept.
    """
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
    """Fetch broad universe once, run Phase-2 baseline triage, print report."""
    _enable_insecure_yfinance_ssl_if_needed()
    base = load_settings()
    settings = settings_for_breadth_diagnostic(base)
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

    print("=== Universe-breadth diagnostic (single variable: universe) ===")
    print(f"requested_n: {len(BROAD_LIQUID_CANDIDATE_UNIVERSE)}")
    print(f"horizon: {DIAGNOSTIC_HORIZON} (forced; limits.yaml default overridden)")
    print(f"seed: {settings.research.seed}")
    print(f"train_end/test_start: {settings.research.train_end} / {settings.research.test_start}")
    print(f"feature_columns: {len(FEATURE_COLUMNS)} (production FEATURE_COLUMNS)")
    print(f"survivorship: {SURVIVORSHIP_BIAS_NOTE}")

    bars, kept, dropped = _download_broad_universe(
        BROAD_LIQUID_CANDIDATE_UNIVERSE,
        provider=provider,
        request_start=request_start,
        request_end=request_end,
        frequency=settings.frequency,
    )
    print(f"kept_n: {len(kept)}  dropped_n: {len(dropped)}")

    report = run_universe_breadth_diagnostic(
        bars,
        settings,
        feature_pipeline=FeaturePipeline(bus, clock=clock),
        artifact_adapter=FilesystemModelArtifactAdapter(settings.artifact_root),
        created_at=clock.now(),
    )
    m = report.metrics
    print("--- OOS metrics ---")
    print(f"universe | N | median_breadth | Rank-IC | IC | ICIR | vs {RANK_IC_MATERIAL_THRESHOLD}")
    icir_s = "None" if m.icir is None else f"{m.icir:.6f}"
    print(
        f"{report.universe_label} | {report.n_symbols} | {m.median_cs_breadth:.1f} | "
        f"{m.rank_ic_mean:.6f} | {m.ic_mean:.6f} | {icir_s} | "
        f"{'>=' if m.rank_ic_mean >= RANK_IC_MATERIAL_THRESHOLD else '<'} threshold"
    )
    print(
        f"reference_50_h2 | {NARROW_UNIVERSE_N} | n/a | {NARROW_H2_RANK_IC:.4f} | "
        f"(documented) | n/a | reference"
    )
    print(f"interpretation: {report.interpretation}")
    print(f"note: {report.interpretation_note}")
    print(f"counts_as_research_trial: {report.counts_as_research_trial}")
    print(f"ic_materially_positive (Pearson rule): {m.ic_materially_positive}")
    print(f"rank_icir: {m.rank_icir}")


if __name__ == "__main__":
    main()
