"""Manual live meta-labeling triage; excluded from pytest/CI (Yahoo).

Pre-registered single run (no post-hoc retune after seeing results).
Universum: BROAD_LIQUID_CANDIDATE_UNIVERSE. Seed/split/HPs from limits.yaml.
Primary: SMA-10 sign. Meta: LGBM binary on ALL_NEW_FEATURE_CLASS_COLUMNS.
No CPCV / Gate — classification + bet-Sharpe triage only.
"""

from __future__ import annotations

import os
import pickle
import time
from datetime import UTC, datetime
from datetime import time as dt_time
from pathlib import Path
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
from aihedgefund.research.meta_labeling import (
    ACCEPT_PROBABILITY_THRESHOLD,
    PRIMARY_MA_WINDOW,
    run_meta_labeling_triage,
)
from aihedgefund.research.universes import (
    BROAD_LIQUID_CANDIDATE_UNIVERSE,
    SURVIVORSHIP_BIAS_NOTE,
)

DOWNLOAD_BATCH_SIZE: Final[int] = 50
BATCH_PAUSE_SEC: Final[float] = 1.5
BARS_CACHE_PATH: Final[Path] = Path("artifacts/meta_labeling_bars.pkl")


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
    """Fetch broad universe once, run meta-labeling triage, print report."""
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

    print("=== Meta-labeling triage (SMA-10 primary + Triple-Barrier + LGBM) ===")
    print(f"requested_n: {len(BROAD_LIQUID_CANDIDATE_UNIVERSE)}")
    print(f"seed: {settings.research.seed}")
    print(
        f"train_end / configured_test_start: "
        f"{settings.research.train_end} / {settings.research.test_start}"
    )
    print(f"primary: SMA-{PRIMARY_MA_WINDOW} sign; accept_p >= {ACCEPT_PROBABILITY_THRESHOLD}")
    print(f"barriers: pt={settings.labels.pt} sl={settings.labels.sl} "
          f"vertical_bars={settings.labels.vertical_bars} vol_span={settings.labels.vol_span}")
    print(f"features: ALL_NEW ({len(ALL_NEW_FEATURE_CLASS_COLUMNS)})")
    print(f"survivorship: {SURVIVORSHIP_BIAS_NOTE}")

    if BARS_CACHE_PATH.is_file() and os.environ.get("AIHF_META_RELOAD_BARS", "1") == "1":
        print(f"loading bars cache: {BARS_CACHE_PATH}")
        with BARS_CACHE_PATH.open("rb") as handle:
            cached = pickle.load(handle)
        bars = cached["bars"]
        kept = cached["kept"]
        dropped = cached["dropped"]
    else:
        bars, kept, dropped = _download_broad_universe(
            BROAD_LIQUID_CANDIDATE_UNIVERSE,
            provider=provider,
            request_start=request_start,
            request_end=request_end,
            frequency=settings.frequency,
        )
        BARS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with BARS_CACHE_PATH.open("wb") as handle:
            pickle.dump({"bars": bars, "kept": kept, "dropped": dropped}, handle)
        print(f"wrote bars cache: {BARS_CACHE_PATH}")

    print(f"kept_n: {len(kept)}  dropped_n: {len(dropped)}")

    run_settings = settings.model_copy(update={"universe": kept})
    report = run_meta_labeling_triage(bars, run_settings)
    c = report.classification

    print("--- OOS meta-labeling metrics ---")
    print(
        f"test_start={report.test_start}  n_train={report.n_train}  "
        f"n_test={report.n_test}  n_accepted={report.n_accepted}  "
        f"accept_rate={report.accept_rate:.4f}"
    )
    print(
        f"Precision={c.precision:.6f}  Recall={c.recall:.6f}  F1={c.f1:.6f}  "
        f"base_rate={c.base_rate:.6f}  lift={c.lift:.4f}"
    )
    print(
        f"TP={c.n_true_positive} FP={c.n_false_positive} "
        f"FN={c.n_false_negative} TN={c.n_true_negative}"
    )
    filt = (
        "None"
        if report.filtered_sharpe_oos is None
        else f"{report.filtered_sharpe_oos:.6f}"
    )
    print(
        f"primary_sharpe_oos={report.primary_sharpe_oos:.6f}  "
        f"filtered_sharpe_oos={filt}"
    )
    print(
        f"precision_above_baserate={report.precision_above_baserate}  "
        f"filtered_sharpe_beats_primary={report.filtered_sharpe_beats_primary}"
    )
    print(f"interpretation: {report.interpretation}")
    print(f"note: {report.interpretation_note}")
    print(f"trial_sharpe_logged: {report.trial_sharpe_logged}")
    print(f"counts_as_research_trial: {report.counts_as_research_trial}")
    print("=== HANDOFF ===")
    print("module: aihedgefund.research.meta_labeling")
    print("script: scripts/run_meta_labeling_triage.py")
    if report.interpretation == "candidate_for_gate":
        print("next: CPCV/DSR gate on this meta-labeling config (separate prompt)")
    else:
        print("next: document search stop; no further free-OHLCV feature/label variants")


if __name__ == "__main__":
    main()
