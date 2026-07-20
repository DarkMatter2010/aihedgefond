"""Manual live CPCV/DSR gate for all_new features; excluded from pytest/CI (Yahoo).

Pre-registered single run (no post-hoc retune / seed fishing).
Primary: ALL_NEW_FEATURE_CLASS_COLUMNS (15), BROAD_LIQUID_CANDIDATE_UNIVERSE,
h=21 + M=100 permutation null. Secondary: h=2 gate DSR for context only.

This validation run does NOT increment N_RESEARCH_TRIALS (already counted in
the feature-class triage table).
"""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from datetime import time as dt_time
from typing import Final, Literal

import numpy as np
import pandas as pd

from aihedgefund.core.bus import InProcessMessageBus
from aihedgefund.core.config import load_settings
from aihedgefund.core.runtime import FrozenClock
from aihedgefund.core.schemas import BarFrame, BaselineDataset, GateVerdict, MarketDataRequest
from aihedgefund.data.adapters import YFinanceProvider
from aihedgefund.data.provider import DataUnavailableError
from aihedgefund.data.quality import DataQualityError, DataQualityGate
from aihedgefund.research.all_new_gate import (
    COUNTS_AS_NEW_RESEARCH_TRIAL,
    CPCV_PARAM_NOTE,
    DSR_THRESHOLD,
    GATE_CANDIDATE_FEATURE_COLUMNS,
    N_PERMUTATIONS,
    N_TRIALS,
    PERM_SEED,
    PRIMARY_HORIZON,
    SECONDARY_HORIZON,
    SEED,
    assemble_all_new_dataset,
    bar_timestamps_from_bars,
    interpret_corrected_verdict,
    permute_labels_within_dates,
    run_all_new_gate,
)
from aihedgefund.research.baseline import build_lgbm_params
from aihedgefund.research.research_trials import research_trial_sharpe_variance
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


def _print_gate_block(label: str, verdict: GateVerdict) -> None:
    """Print core gate fields from a GateVerdict."""
    print(f"--- {label} ---")
    print(f"path_sharpe_mean: {verdict.path_sharpe_mean}")
    print(f"path_sharpe_std: {verdict.path_sharpe_std}")
    print(f"observed_sharpe: {verdict.deflated.observed_sharpe}")
    print(f"n_obs_T: {verdict.deflated.n_obs}")
    print(f"sr0: {verdict.deflated.sr0}")
    print(f"var_trial_sharpes: {verdict.deflated.var_trial_sharpes}")
    print(f"n_trials: {verdict.n_trials}")
    print(f"dsr: {verdict.dsr}")
    print(f"verdict: {verdict.verdict}")


def main() -> None:
    """Fetch broad universe once; primary h=21 gate+null; secondary h=2 gate."""
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

    print("=== all_new CPCV/DSR gate (hardened) ===")
    print(f"candidate_features: {len(GATE_CANDIDATE_FEATURE_COLUMNS)} (ALL_NEW only)")
    print(f"primary_horizon: {PRIMARY_HORIZON}")
    print(f"secondary_horizon: {SECONDARY_HORIZON} (context only)")
    print(f"seed: {SEED}")
    print(
        f"n_trials: {N_TRIALS} (does NOT increment; counts_as_new={COUNTS_AS_NEW_RESEARCH_TRIAL})"
    )
    print(f"cpcv: {CPCV_PARAM_NOTE}")
    print(f"dsr_threshold: {DSR_THRESHOLD}")
    print(f"survivorship: {SURVIVORSHIP_BIAS_NOTE}")

    bars, kept, dropped = _download_broad_universe(
        BROAD_LIQUID_CANDIDATE_UNIVERSE,
        provider=provider,
        request_start=request_start,
        request_end=request_end,
        frequency=settings.frequency,
    )
    print(f"kept_n: {len(kept)}  dropped_n: {len(dropped)}")
    bar_timestamps = bar_timestamps_from_bars(bars)
    universe = kept

    params = build_lgbm_params(
        seed=SEED,
        learning_rate=research.learning_rate,
        num_leaves=research.num_leaves,
        min_data_in_leaf=research.min_data_in_leaf,
        feature_fraction=research.feature_fraction,
        bagging_fraction=research.bagging_fraction,
        bagging_freq=research.bagging_freq,
    )
    var_trial = research_trial_sharpe_variance()
    print(f"var_trial_sharpes: {var_trial} (RESEARCH_TRIAL_SHARPES)")

    # --- Primary h=21 ---
    primary = assemble_all_new_dataset(bars, horizon=PRIMARY_HORIZON)
    primary_verdict = run_all_new_gate(
        primary,
        model_params=params,
        num_boost_round=research.num_boost_round,
        seed=SEED,
        universe=universe,
        start=settings.start,
        end=settings.end,
        frequency=settings.frequency,
        bar_timestamps=bar_timestamps,
    )
    _print_gate_block(f"primary h={PRIMARY_HORIZON}", primary_verdict)
    real_dsr = float(primary_verdict.dsr)

    print(f"--- permutation null M={N_PERMUTATIONS} (h={PRIMARY_HORIZON}) ---")
    rng = np.random.default_rng(PERM_SEED)
    null_dsrs: list[float] = []
    for i in range(N_PERMUTATIONS):
        perm_label = permute_labels_within_dates(primary.label, rng=rng)
        perm_dataset = BaselineDataset(
            features=primary.features,
            label=perm_label,
            horizon=primary.horizon,
            feature_columns=primary.feature_columns,
        )
        perm_verdict = run_all_new_gate(
            perm_dataset,
            model_params=params,
            num_boost_round=research.num_boost_round,
            seed=SEED,
            universe=universe,
            start=settings.start,
            end=settings.end,
            frequency=settings.frequency,
            bar_timestamps=bar_timestamps,
        )
        dsr_i = float(perm_verdict.dsr)
        null_dsrs.append(dsr_i)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"perm {i + 1}/{N_PERMUTATIONS}: dsr={dsr_i:.6g}")

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

    # --- Secondary h=2 (context only) ---
    secondary = assemble_all_new_dataset(bars, horizon=SECONDARY_HORIZON)
    secondary_verdict = run_all_new_gate(
        secondary,
        model_params=params,
        num_boost_round=research.num_boost_round,
        seed=SEED,
        universe=universe,
        start=settings.start,
        end=settings.end,
        frequency=settings.frequency,
        bar_timestamps=bar_timestamps,
    )
    _print_gate_block(
        f"secondary h={SECONDARY_HORIZON} (context only; not the fishing target)",
        secondary_verdict,
    )

    print("=== INTERPRETATION ===")
    print(f"primary_corrected_verdict: {corrected}")
    if corrected == "JA":
        print(
            "handoff: Phase-3 Slice-2 (vectorbt / Nautilus / paper) WITH "
            "ALL_NEW_FEATURE_CLASS_COLUMNS"
        )
    else:
        print(
            "handoff: STOP free-yfinance OHLCV feature hunting; "
            "no CPCV/DSR-validated signal on this dataset; no Phase-4 without signal"
        )
    print("=== HANDOFF ===")
    print("module: aihedgefund.research.all_new_gate")
    print("script: scripts/run_gate_on_all_new_features.py")
    print(f"feature_set: ALL_NEW_FEATURE_CLASS_COLUMNS ({len(GATE_CANDIDATE_FEATURE_COLUMNS)})")


if __name__ == "__main__":
    main()
