"""Manual live small-cap universe IC diagnostic; excluded from pytest/CI (Yahoo).

Pre-registered single run (no post-hoc retune after seeing results).
Universum: SMALL_CAP_CANDIDATE_UNIVERSE. Features: ALL_NEW_FEATURE_CLASS_COLUMNS.
Seed/split/HPs from limits.yaml. Horizons: h=2 and h=21. No CPCV / Gate.
Counts as one research trial.
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
from aihedgefund.features.feature_classes import ALL_NEW_FEATURE_CLASS_COLUMNS
from aihedgefund.features.pipeline import FEATURE_COLUMNS
from aihedgefund.research.small_cap_universe_diagnostic import (
    BROAD_ALL_NEW_H2_RANK_IC,
    BROAD_ALL_NEW_H21_RANK_IC,
    DIAGNOSTIC_HORIZONS,
    RANK_IC_MATERIAL_THRESHOLD,
    run_small_cap_universe_diagnostic,
)
from aihedgefund.research.universes import (
    SMALL_CAP_CANDIDATE_UNIVERSE,
    SMALL_CAP_SURVIVORSHIP_BIAS_NOTE,
)

DOWNLOAD_BATCH_SIZE: Final[int] = 50
BATCH_PAUSE_SEC: Final[float] = 1.5


def _enable_insecure_yfinance_ssl_if_needed() -> None:
    """Opt-in workaround: curl_cffi ignores the Windows trust store (MITM/corp SSL).

    Set `AIHF_ALLOW_INSECURE_YFINANCE_SSL=0` to disable. Default on for this
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


def _download_universe(
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
    """Fetch small-cap universe once, run ALL_NEW IC diagnostic, print report."""
    _enable_insecure_yfinance_ssl_if_needed()
    base = load_settings()
    settings = base.model_copy(update={"universe": SMALL_CAP_CANDIDATE_UNIVERSE})
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

    print("=== Small-cap universe IC diagnostic (ALL_NEW features) ===")
    print(f"requested_n: {len(SMALL_CAP_CANDIDATE_UNIVERSE)}")
    print(f"horizons: {DIAGNOSTIC_HORIZONS}")
    print(f"seed: {settings.research.seed}")
    print(
        f"train_end / configured_test_start: "
        f"{settings.research.train_end} / {settings.research.test_start}"
    )
    print("(per-horizon test_start may extend so embargo_days == horizon is feasible)")
    print(f"production FEATURE_COLUMNS (alt): {len(FEATURE_COLUMNS)} (unchanged)")
    print(f"features: ALL_NEW ({len(ALL_NEW_FEATURE_CLASS_COLUMNS)})")
    print(f"threshold Rank-IC: {RANK_IC_MATERIAL_THRESHOLD}")
    print(
        f"broad all_new refs: h=2 {BROAD_ALL_NEW_H2_RANK_IC:.4f} / "
        f"h=21 {BROAD_ALL_NEW_H21_RANK_IC:.4f}"
    )
    print(f"survivorship: {SMALL_CAP_SURVIVORSHIP_BIAS_NOTE}")

    bars, kept, dropped = _download_universe(
        SMALL_CAP_CANDIDATE_UNIVERSE,
        provider=provider,
        request_start=request_start,
        request_end=request_end,
        frequency=settings.frequency,
    )
    drop_rate = len(dropped) / len(SMALL_CAP_CANDIDATE_UNIVERSE)
    print(f"kept_n: {len(kept)}  dropped_n: {len(dropped)}  drop_rate: {drop_rate:.4f}")
    if dropped:
        print(f"dropped: {', '.join(dropped)}")

    run_settings = settings.model_copy(update={"universe": kept})
    report = run_small_cap_universe_diagnostic(
        bars,
        run_settings,
        n_symbols_dropped=len(dropped),
    )

    print("--- OOS metrics (Small-cap ALL_NEW x Horizont) ---")
    print(
        "h | test_start | n_feat | N_dates | median_breadth | "
        "Rank-IC | IC | ICIR | GrinoldSR | broad_all_new | vs 0.02"
    )
    for row in report.rows:
        m = row.metrics
        icir_s = "None" if m.icir is None else f"{m.icir:.6f}"
        flag = ">=" if row.rank_ic_above_threshold else "<"
        print(
            f"{row.horizon} | {row.test_start} | {row.feature_column_count} | "
            f"{m.n_dates} | {m.median_cs_breadth:.1f} | "
            f"{m.rank_ic_mean:.6f} | {m.ic_mean:.6f} | {icir_s} | "
            f"{row.grinold_sr:.6f} | {row.broad_all_new_rank_ic:.6f} | "
            f"{flag} threshold"
        )

    print(
        f"best: h={report.best_horizon} Rank-IC={report.best_rank_ic:.6f} "
        f"GrinoldSR={report.best_grinold_sr:.6f}"
    )
    print(
        f"drop_rate: {report.drop_rate:.4f} "
        f"({report.n_symbols_dropped}/{report.n_symbols_requested})"
    )
    print(f"interpretation: {report.interpretation}")
    print(f"note: {report.interpretation_note}")
    print(f"counts_as_research_trial: {report.counts_as_research_trial}")
    print("=== HANDOFF ===")
    print("modules: aihedgefund.research.small_cap_universe_diagnostic, universes")
    print("script: scripts/run_small_cap_universe_diagnostic.py")
    if report.interpretation == "candidate_for_gate":
        print("next: Gate-Prompt for small-cap ALL_NEW (this run is not a gate)")
    else:
        print("next: Free-OHLCV search stop — no further free-data OHLCV variants")


if __name__ == "__main__":
    main()
