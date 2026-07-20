"""Manual live CPCV/DSR gate for meta-labeling; excluded from pytest/CI (Yahoo).

Pre-registered single run (no post-hoc retune / seed fishing).
Candidate: SMA-10 primary + Triple-Barrier meta-labels + LGBM binary accept
on ALL_NEW_FEATURE_CLASS_COLUMNS, BROAD_LIQUID_CANDIDATE_UNIVERSE.
CPCV N=6 k=2, horizon=vertical_bars=10.

Permutation null size: env ``AIHF_GATE_N_PERMUTATIONS`` (default **10** for
practical live runs on ~1.3M events). Module constant ``N_PERMUTATIONS=100``
remains the planned / documented full-null size for offline tests.

Supports resume via ``artifacts/meta_gate_checkpoint.json`` and dataset pickle.

This validation run does NOT increment N_RESEARCH_TRIALS (already row 24).
"""

from __future__ import annotations

import json
import os
import pickle
import time
from datetime import UTC, datetime
from datetime import time as dt_time
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import pandas as pd

from aihedgefund.core.bus import InProcessMessageBus
from aihedgefund.core.config import load_settings
from aihedgefund.core.runtime import FrozenClock
from aihedgefund.core.schemas import BarFrame, BaselineDataset, GateVerdict, MarketDataRequest
from aihedgefund.data.adapters import YFinanceProvider
from aihedgefund.data.provider import DataUnavailableError
from aihedgefund.data.quality import DataQualityError, DataQualityGate
from aihedgefund.features.feature_classes import ALL_NEW_FEATURE_CLASS_COLUMNS
from aihedgefund.research.meta_labeling import PRIMARY_MA_WINDOW
from aihedgefund.research.meta_labeling_gate import (
    COUNTS_AS_NEW_RESEARCH_TRIAL,
    CPCV_PARAM_NOTE,
    DSR_THRESHOLD,
    N_PERMUTATIONS,
    N_TRIALS,
    PERM_SEED,
    SEED,
    VERTICAL_BARS,
    binary_params_from_settings,
    interpret_corrected_verdict,
    permute_labels_within_dates,
    prepare_meta_gate_inputs,
    run_meta_labeling_gate,
)
from aihedgefund.research.research_trials import research_trial_sharpe_variance
from aihedgefund.research.universes import (
    BROAD_LIQUID_CANDIDATE_UNIVERSE,
    SURVIVORSHIP_BIAS_NOTE,
)

# Planned full-null size lives on the module (``N_PERMUTATIONS=100``). Live runs
# default to a smaller M via env — ~1.3M events make M=100 impractically slow.
_DEFAULT_LIVE_N_PERMUTATIONS: Final[int] = 10

DOWNLOAD_BATCH_SIZE: Final[int] = 50
BATCH_PAUSE_SEC: Final[float] = 1.5
BARS_CACHE_PATH: Final[Path] = Path("artifacts/meta_labeling_bars.pkl")
DATASET_CACHE_PATH: Final[Path] = Path("artifacts/meta_gate_dataset.pkl")
CHECKPOINT_PATH: Final[Path] = Path("artifacts/meta_gate_checkpoint.json")


def _live_n_permutations() -> int:
    """Live null size from ``AIHF_GATE_N_PERMUTATIONS`` (default 10)."""
    raw = os.environ.get("AIHF_GATE_N_PERMUTATIONS", "").strip()
    if not raw:
        return _DEFAULT_LIVE_N_PERMUTATIONS
    value = int(raw)
    if value < 1:
        msg = f"AIHF_GATE_N_PERMUTATIONS must be >= 1, got {value}"
        raise ValueError(msg)
    return value


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
    """Batch Yahoo fetch with per-symbol fallback."""
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


def _print_gate_block(title: str, verdict: GateVerdict) -> None:
    print(f"--- {title} ---")
    print(f"verdict: {verdict.verdict}")
    print(f"dsr: {verdict.dsr}")
    print(f"observed_sharpe: {verdict.deflated.observed_sharpe}")
    print(f"n_obs_T: {verdict.deflated.n_obs}")
    print(f"sr0: {verdict.deflated.sr0}")
    print(f"var_trial_sharpes: {verdict.deflated.var_trial_sharpes}")
    print(f"n_trials: {verdict.n_trials}")
    print(f"path_sharpe_mean: {verdict.path_sharpe_mean}")
    print(f"path_sharpe_std: {verdict.path_sharpe_std}")


def _load_checkpoint() -> dict[str, Any]:
    if not CHECKPOINT_PATH.is_file():
        return {}
    with CHECKPOINT_PATH.open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def _save_checkpoint(payload: dict[str, Any]) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def main() -> None:
    """Fetch or reload bars; run meta-labeling CPCV/DSR gate + permutation null."""
    _enable_insecure_yfinance_ssl_if_needed()
    base = load_settings()
    research = base.research.model_copy(update={"seed": SEED})
    settings = base.model_copy(
        update={
            "universe": BROAD_LIQUID_CANDIDATE_UNIVERSE,
            "research": research,
        }
    )
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

    print("=== meta-labeling CPCV/DSR gate (hardened) ===")
    print(f"primary: SMA-{PRIMARY_MA_WINDOW} + Triple-Barrier meta + LGBM binary")
    print(f"features: ALL_NEW ({len(ALL_NEW_FEATURE_CLASS_COLUMNS)})")
    print(f"vertical_bars / CPCV horizon: {VERTICAL_BARS}")
    print(f"seed: {SEED}")
    print(
        f"n_trials: {N_TRIALS} (does NOT increment; counts_as_new={COUNTS_AS_NEW_RESEARCH_TRIAL})"
    )
    print(f"cpcv: {CPCV_PARAM_NOTE}")
    print(f"dsr_threshold: {DSR_THRESHOLD}")
    print(f"survivorship: {SURVIVORSHIP_BIAS_NOTE}")

    checkpoint = _load_checkpoint()

    if DATASET_CACHE_PATH.is_file() and os.environ.get("AIHF_META_RELOAD_DATASET", "1") == "1":
        print(f"loading dataset cache: {DATASET_CACHE_PATH}")
        with DATASET_CACHE_PATH.open("rb") as handle:
            cached_ds = pickle.load(handle)
        dataset = cached_ds["dataset"]
        bet_returns = cached_ds["bet_returns"]
        bar_timestamps = cached_ds["bar_timestamps"]
        kept = cached_ds["kept"]
        dropped = cached_ds["dropped"]
    else:
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
        print("assembling meta-label dataset...")
        dataset, bet_returns, bar_timestamps = prepare_meta_gate_inputs(
            bars, settings.labels
        )
        with DATASET_CACHE_PATH.open("wb") as handle:
            pickle.dump(
                {
                    "dataset": dataset,
                    "bet_returns": bet_returns,
                    "bar_timestamps": bar_timestamps,
                    "kept": kept,
                    "dropped": dropped,
                },
                handle,
            )
        print(f"wrote dataset cache: {DATASET_CACHE_PATH}")

    print(f"kept_n: {len(kept)}  dropped_n: {len(dropped)}")
    print(f"n_events: {len(dataset.features)}")
    universe = kept

    params = binary_params_from_settings(settings)
    var_trial = research_trial_sharpe_variance()
    print(f"var_trial_sharpes: {var_trial}")

    gate_kwargs = dict(
        model_params=params,
        num_boost_round=research.num_boost_round,
        seed=SEED,
        universe=universe,
        start=settings.start,
        end=settings.end,
        frequency=settings.frequency,
        bar_timestamps=bar_timestamps,
    )

    if "real_dsr" in checkpoint and "real_summary" in checkpoint:
        real_dsr = float(checkpoint["real_dsr"])
        print("--- meta-labeling gate (from checkpoint) ---")
        for key, value in checkpoint["real_summary"].items():
            print(f"{key}: {value}")
    else:
        verdict = run_meta_labeling_gate(dataset, bet_returns, **gate_kwargs)
        _print_gate_block(f"meta-labeling gate (h={VERTICAL_BARS})", verdict)
        real_dsr = float(verdict.dsr)
        checkpoint = {
            "real_dsr": real_dsr,
            "real_summary": {
                "verdict": verdict.verdict,
                "dsr": float(verdict.dsr),
                "observed_sharpe": float(verdict.deflated.observed_sharpe),
                "n_obs_T": int(verdict.deflated.n_obs),
                "sr0": float(verdict.deflated.sr0),
                "var_trial_sharpes": float(verdict.deflated.var_trial_sharpes),
                "n_trials": int(verdict.n_trials),
                "path_sharpe_mean": float(verdict.path_sharpe_mean),
                "path_sharpe_std": float(verdict.path_sharpe_std),
            },
            "null_dsrs": [],
        }
        _save_checkpoint(checkpoint)

    n_perm = _live_n_permutations()
    null_dsrs: list[float] = [float(x) for x in checkpoint.get("null_dsrs", [])]
    # Truncate if a prior longer null was interrupted.
    if len(null_dsrs) > n_perm:
        null_dsrs = null_dsrs[:n_perm]
    print(
        f"--- permutation null M={n_perm} (have {len(null_dsrs)}; "
        f"planned constant N_PERMUTATIONS={N_PERMUTATIONS}; "
        f"override via AIHF_GATE_N_PERMUTATIONS) ---"
    )
    rng = np.random.default_rng(PERM_SEED)
    for i in range(n_perm):
        perm_label = permute_labels_within_dates(dataset.label, rng=rng)
        if i < len(null_dsrs):
            continue
        perm_dataset = BaselineDataset(
            features=dataset.features,
            label=perm_label,
            horizon=dataset.horizon,
            feature_columns=dataset.feature_columns,
        )
        perm_verdict = run_meta_labeling_gate(
            perm_dataset,
            bet_returns,
            **gate_kwargs,
        )
        dsr_i = float(perm_verdict.dsr)
        null_dsrs.append(dsr_i)
        checkpoint["null_dsrs"] = null_dsrs
        _save_checkpoint(checkpoint)
        if (i + 1) % 5 == 0 or i == 0 or i + 1 == n_perm:
            print(f"perm {i + 1}/{n_perm}: dsr={dsr_i:.6g}", flush=True)

    null_arr = np.asarray(null_dsrs, dtype=np.float64)
    q50 = float(np.quantile(null_arr, 0.50))
    q95 = float(np.quantile(null_arr, 0.95))
    percentile = float(100.0 * np.mean(null_arr < real_dsr))
    corrected, gate_ja, beats_null, null_broken = interpret_corrected_verdict(
        real_dsr=real_dsr,
        null_q95=q95,
    )
    print(f"null_dsr_q50: {q50}")
    print(f"null_dsr_q95: {q95}")
    print(f"null_dsr_min: {float(null_arr.min())}")
    print(f"null_dsr_max: {float(null_arr.max())}")
    print(f"real_dsr: {real_dsr}")
    print(f"real_percentile_vs_null: {percentile:.2f}")
    print(f"beats_null_q95: {beats_null}")
    print(f"gate_dsr_ge_0_95: {gate_ja}")
    if null_broken:
        print(
            "GATE_BROKEN: null DSR 95% quantile is already >= 0.9 — "
            "selection-bias correction still inflated"
        )
    print(f"corrected_verdict: {corrected}")

    print("=== INTERPRETATION ===")
    print(f"corrected_verdict: {corrected}")
    if corrected == "JA":
        print("handoff: Phase-3 Slice-2 (vectorbt / Nautilus / paper) WITH meta-labeling")
    else:
        print(
            "handoff: STOP free-yfinance OHLCV signal search; "
            "meta-labeling failed hardened gate; no Phase-4 without validated signal"
        )
    print("=== HANDOFF ===")
    print("module: aihedgefund.research.meta_labeling_gate")
    print("script: scripts/run_gate_on_meta_labeling.py")


if __name__ == "__main__":
    main()
