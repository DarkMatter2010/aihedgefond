"""Phase-2 Abschluss-Probe: Cross-Sectional-Momentum auf breitem Universe.

Vorab-registrierter EINMAL-Lauf (kein Nachjustieren nach Sicht der Ergebnisse).
Diagnose-Skript — bewusst nicht Teil von pytest/CI.

Bindet bestehende Research-APIs:
  make_forward_return_labels, time_embargo_split, train_baseline / build_lgbm_params,
  compute_ic_metrics, assemble_baseline_dataset.

Survivorship-Hinweis: ``BROAD_LIQUID_CANDIDATE_UNIVERSE`` (alias
``CANDIDATE_UNIVERSE``) ist die heutige S&P-500-nahe Liste
(milder Survivorship-Bias). Continuity-Filter gilt as-of TRAIN_END (keine
OOS-Look-Ahead-Selektion). Jedes positive Ergebnis ist PROVISORISCH und muss
in Phase 3 unter CPCV/DSR bestätigt werden.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time
from typing import Final, Literal

import numpy as np
import pandas as pd

from aihedgefund.core.bus import InProcessMessageBus
from aihedgefund.core.config import load_settings
from aihedgefund.core.runtime import FrozenClock
from aihedgefund.core.schemas import (
    BarFrame,
    BaselineDataset,
    CorporateActionInput,
    MarketDataRequest,
)
from aihedgefund.data.adapters.yfinance import YFinanceProvider
from aihedgefund.data.corporate_actions import adjust_corporate_actions
from aihedgefund.data.provider import DataUnavailableError
from aihedgefund.data.quality import DataQualityError, DataQualityGate
from aihedgefund.research.baseline import build_lgbm_params, predict_scores, train_baseline
from aihedgefund.research.dataset import assemble_baseline_dataset
from aihedgefund.research.forward_labels import make_forward_return_labels
from aihedgefund.research.metrics import compute_ic_metrics
from aihedgefund.research.model_hash import compute_model_hash
from aihedgefund.research.split import time_embargo_split
from aihedgefund.research.universes import BROAD_LIQUID_CANDIDATE_UNIVERSE

# ---------------------------------------------------------------------------
# Vorab-Registrierung (FIX — nicht ändern nach Ergebnis-Sicht)
# ---------------------------------------------------------------------------

DATA_START: Final[date] = date(2019, 1, 2)
DATA_END: Final[date] = date(2025, 1, 2)  # yfinance end exclusive → last bar ~2024-12-31
TRAIN_END: Final[date] = date(2023, 6, 30)
TEST_START: Final[date] = date(2024, 1, 2)
TEST_WINDOW_END: Final[date] = date(2024, 12, 31)
HORIZONS: Final[tuple[int, ...]] = (63, 126)
RANK_IC_MATERIAL_THRESHOLD: Final[float] = 0.02
SKIP_DAYS: Final[int] = 21
MOM_WINDOWS: Final[tuple[int, ...]] = (63, 126, 252)
VOL_WINDOW: Final[int] = 63
REV_WINDOW: Final[int] = 21
DOWNLOAD_BATCH_SIZE: Final[int] = 50
FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "mom_63_skip21",
    "mom_126_skip21",
    "mom_252_skip21",
    "rev_21",
    "vol_63",
)

# Shared with universe-breadth diagnostic — do not fork a second copy.
CANDIDATE_UNIVERSE: tuple[str, ...] = BROAD_LIQUID_CANDIDATE_UNIVERSE


def _as_of_adjusted_close(bars: BarFrame, symbol: str) -> pd.Series:
    """Point-in-time total-return close via Phase-1 corporate-action transform."""
    frame = bars.bars[symbol]
    return adjust_corporate_actions(
        CorporateActionInput(
            raw_close=frame["close"],
            splits=bars.splits[symbol],
            dividends=bars.dividends[symbol],
        )
    ).as_of_adjusted


def _cross_section_zscore(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    """Z-standardize each feature across symbols within each timestamp."""
    if list(feature_matrix.index.names) != ["timestamp", "symbol"]:
        msg = "feature matrix index names must be ('timestamp', 'symbol')"
        raise ValueError(msg)

    def _zscore_block(block: pd.DataFrame) -> pd.DataFrame:
        values = block.to_numpy(dtype=float)
        means = np.nanmean(values, axis=0)
        stds = np.nanstd(values, axis=0, ddof=0)
        stds = np.where(stds > 0.0, stds, np.nan)
        z = (values - means) / stds
        z = np.where(np.isfinite(z), z, 0.0)
        return pd.DataFrame(z, index=block.index, columns=block.columns)

    parts: list[pd.DataFrame] = []
    for _ts, group in feature_matrix.groupby(level="timestamp", sort=True):
        # drop symbol level for block math, restore MultiIndex after
        flat = group.droplevel("timestamp")
        if len(flat) < 2:
            zeros = pd.DataFrame(0.0, index=group.index, columns=group.columns)
            parts.append(zeros)
            continue
        z_flat = _zscore_block(flat)
        z_flat.index = group.index
        parts.append(z_flat)
    return pd.concat(parts).sort_index().astype("float64")


def build_momentum_features(bars: BarFrame) -> pd.DataFrame:
    """Causal momentum / reversal / vol features; then cross-sectional z-scores."""
    per_symbol: list[pd.DataFrame] = []
    for symbol in bars.bars:
        close = _as_of_adjusted_close(bars, symbol).astype(float)
        logret = np.log(close / close.shift(1))
        columns: dict[str, pd.Series] = {}
        for window in MOM_WINDOWS:
            # Formation ending at t-SKIP: close[t-SKIP] / close[t-SKIP-window] - 1
            lagged = close.shift(SKIP_DAYS)
            columns[f"mom_{window}_skip{SKIP_DAYS}"] = lagged / lagged.shift(window) - 1.0
        columns[f"rev_{REV_WINDOW}"] = close / close.shift(REV_WINDOW) - 1.0
        columns[f"vol_{VOL_WINDOW}"] = logret.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std()
        frame = pd.DataFrame(columns)
        frame.index = pd.MultiIndex.from_arrays(
            [frame.index, [symbol] * len(frame)],
            names=("timestamp", "symbol"),
        )
        per_symbol.append(frame)

    if not per_symbol:
        msg = "no symbols available for feature construction"
        raise ValueError(msg)

    raw = pd.concat(per_symbol).sort_index()
    raw = raw.loc[:, list(FEATURE_COLUMNS)]
    # Drop incomplete lookback rows before CS z-score so breadth is not inflated by NaNs
    complete = raw.notna().all(axis=1)
    raw = raw.loc[complete]
    if raw.empty:
        msg = "momentum feature matrix empty after lookback drop"
        raise ValueError(msg)
    return _cross_section_zscore(raw)


def _has_continuous_history(
    frame: pd.DataFrame,
    *,
    calendar: pd.DatetimeIndex,
) -> bool:
    """True iff ``frame`` has a finite close on every date in ``calendar``."""
    if frame.empty or "close" not in frame.columns:
        return False
    aligned = frame.reindex(calendar)
    close = aligned["close"]
    return bool(close.notna().all() and np.isfinite(close.to_numpy(dtype=float)).all())


def _download_universe(
    symbols: tuple[str, ...],
    *,
    provider: YFinanceProvider,
    request_start: datetime,
    request_end: datetime,
    frequency: Literal["1d"],
) -> tuple[BarFrame, tuple[str, ...], tuple[str, ...]]:
    """Load Yahoo once in batches; keep symbols with continuous history + quality pass.

    Returns ``(bars, kept_symbols, dropped_symbols)``.
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
            # Fall back to per-symbol so one bad ticker does not kill the batch.
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
            continue

        for symbol in batch:
            collected_bars[symbol] = batch_bars.bars[symbol]
            collected_divs[symbol] = batch_bars.dividends[symbol]
            collected_splits[symbol] = batch_bars.splits[symbol]

    if "SPY" not in collected_bars and "AAPL" not in collected_bars:
        msg = "reference calendar symbol missing after download"
        raise RuntimeError(msg)

    # Prefer SPY calendar when available; else densest large-cap.
    reference_symbol = "SPY" if "SPY" in collected_bars else "AAPL"
    if reference_symbol not in collected_bars:
        reference_symbol = next(iter(collected_bars))
    calendar = collected_bars[reference_symbol].index
    # Restrict calendar to [DATA_START, DATA_END).
    start_ts = pd.Timestamp(DATA_START, tz="UTC")
    end_ts = pd.Timestamp(DATA_END, tz="UTC")
    calendar = calendar[(calendar >= start_ts) & (calendar < end_ts)]
    min_bars = 252 + SKIP_DAYS + max(HORIZONS) + 50
    if len(calendar) < min_bars:
        msg = f"reference calendar too short: {len(calendar)} bars"
        raise RuntimeError(msg)

    # Continuity is evaluated as-of TRAIN_END only. Requiring unbroken closes
    # through the OOS window would look ahead from the training decision point.
    train_end_ts = pd.Timestamp(TRAIN_END, tz="UTC")
    continuity_calendar = calendar[calendar <= train_end_ts]
    if continuity_calendar.empty:
        msg = "continuity calendar empty after as-of TRAIN_END restriction"
        raise RuntimeError(msg)

    kept: list[str] = []
    for symbol, frame in collected_bars.items():
        if symbol == "SPY":
            # SPY is calendar reference only if it was injected; not in S&P list usually
            pass
        if not _has_continuous_history(frame, calendar=continuity_calendar):
            dropped.append(symbol)
            continue
        # Trim to full research calendar so all frames share identical timestamps.
        # OOS gaps remain NaN and are handled downstream (not a universe reject).
        collected_bars[symbol] = frame.reindex(calendar)
        collected_divs[symbol] = collected_divs[symbol].reindex(calendar, fill_value=0.0)
        collected_splits[symbol] = collected_splits[symbol].reindex(calendar, fill_value=0.0)
        kept.append(symbol)

    # SPY is not in CANDIDATE_UNIVERSE — exclude from research universe if present
    kept = [s for s in kept if s in set(CANDIDATE_UNIVERSE)]
    if len(kept) < 100:
        msg = f"history filter left only {len(kept)} symbols; aborting"
        raise RuntimeError(msg)

    kept_tuple = tuple(sorted(kept))
    bars = BarFrame(
        bars={s: collected_bars[s] for s in kept_tuple},
        dividends={s: collected_divs[s] for s in kept_tuple},
        splits={s: collected_splits[s] for s in kept_tuple},
    )
    return bars, kept_tuple, tuple(sorted(set(dropped)))


def _restrict_test_window(dataset: BaselineDataset, *, test_end: date) -> BaselineDataset:
    """Drop test rows after the registered OOS window end (still within DATA_END)."""
    timestamps = dataset.features.index.get_level_values("timestamp")
    dates = np.array([ts.date() for ts in timestamps])
    mask = dates <= test_end
    if not mask.any():
        msg = "test window restriction removed all rows"
        raise ValueError(msg)
    return BaselineDataset(
        features=dataset.features.iloc[mask],
        label=dataset.label.iloc[mask],
        horizon=dataset.horizon,
        feature_columns=dataset.feature_columns,
    )


def run_one_horizon(
    bars: BarFrame,
    feature_matrix: pd.DataFrame,
    *,
    horizon: int,
    seed: int,
    learning_rate: float,
    num_leaves: int,
    min_data_in_leaf: int,
    feature_fraction: float,
    bagging_fraction: float,
    bagging_freq: int,
    num_boost_round: int,
    ic_positive_threshold: float,
    min_cs_breadth_for_reliable_ic: int,
) -> dict[str, object]:
    """Train/evaluate one registered horizon on the shared feature matrix."""
    embargo_days = horizon
    labels, _meta = make_forward_return_labels(bars, horizon=horizon)
    dataset = assemble_baseline_dataset(
        feature_matrix,
        labels,
        horizon=horizon,
        feature_columns=FEATURE_COLUMNS,
    )
    train, test, split_def = time_embargo_split(
        dataset,
        train_end=TRAIN_END,
        test_start=TEST_START,
        embargo_days=embargo_days,
        horizon=horizon,
    )
    # Cap OOS at end of 2024 (registered test window)
    test = _restrict_test_window(test, test_end=TEST_WINDOW_END)

    params = build_lgbm_params(
        seed=seed,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_data_in_leaf=min_data_in_leaf,
        feature_fraction=feature_fraction,
        bagging_fraction=bagging_fraction,
        bagging_freq=bagging_freq,
    )
    hyperparams = dict(params)
    hyperparams["num_boost_round"] = num_boost_round
    hyperparams["horizon"] = horizon
    hyperparams["embargo_days"] = embargo_days
    model_hash = compute_model_hash(
        features=FEATURE_COLUMNS,
        hyperparameters=hyperparams,
        universe=tuple(sorted(bars.bars)),
        start=DATA_START,
        end=DATA_END,
        frequency="1d",
        seed=seed,
    )
    model = train_baseline(train, params=params, num_boost_round=num_boost_round)
    predictions = predict_scores(model, test.features, model_hash=model_hash)
    metrics = compute_ic_metrics(
        predictions.scores,
        test.label,
        ic_positive_threshold=ic_positive_threshold,
        min_cs_breadth_for_reliable_ic=min_cs_breadth_for_reliable_ic,
    )
    n_symbols = int(test.features.index.get_level_values("symbol").nunique())
    test_timestamps = test.features.index.get_level_values("timestamp")
    if len(test_timestamps) == 0:
        msg = "test window has no timestamps after split/restrict"
        raise ValueError(msg)
    test_dates = pd.DatetimeIndex(test_timestamps).tz_convert("UTC").date
    actual_test_start = min(test_dates)
    actual_test_end = max(test_dates)
    return {
        "horizon": horizon,
        "embargo_days": embargo_days,
        "ic_mean": metrics.ic_mean,
        "rank_ic_mean": metrics.rank_ic_mean,
        "icir": metrics.icir,
        "rank_icir": metrics.rank_icir,
        "n_symbols": n_symbols,
        "cross_section_width_median": metrics.median_cs_breadth,
        "test_window": (f"{actual_test_start.isoformat()} → {actual_test_end.isoformat()}"),
        "train_end": TRAIN_END.isoformat(),
        "test_start": split_def.test_start.isoformat(),
        "train_rows": split_def.train_rows,
        "test_rows": len(test.features),
        "n_dates": metrics.n_dates,
        "cs_breadth_warning": metrics.cs_breadth_warning,
    }


def format_results_table(rows: list[dict[str, object]]) -> str:
    """Render a plain-text result table for Slack / stdout."""
    header = (
        f"{'horizon':>8}  {'ic_mean':>10}  {'rank_ic_mean':>12}  {'icir':>10}  "
        f"{'n_symbols':>9}  {'cs_width_med':>12}  {'test_window'}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        icir = row["icir"]
        icir_s = f"{icir:.6f}" if isinstance(icir, float) else "None"
        lines.append(
            f"{row['horizon']:>8}  {row['ic_mean']:>10.6f}  {row['rank_ic_mean']:>12.6f}  "
            f"{icir_s:>10}  {row['n_symbols']:>9}  "
            f"{row['cross_section_width_median']:>12.1f}  {row['test_window']}"
        )
    return "\n".join(lines)


def verdict_from_rows(rows: list[dict[str, object]]) -> tuple[bool, str]:
    """JA only if every horizon has Rank-IC > threshold with consistent sign."""
    if not rows:
        return False, "NEIN — keine Ergebnisse"
    rank_ics = [float(row["rank_ic_mean"]) for row in rows]  # type: ignore[arg-type]
    all_material = all(ric > RANK_IC_MATERIAL_THRESHOLD for ric in rank_ics)
    signs = {1 if ric > 0 else (-1 if ric < 0 else 0) for ric in rank_ics}
    consistent = len(signs) == 1 and 0 not in signs
    if all_material and consistent:
        return True, (
            f"JA — Rank-IC materiell > {RANK_IC_MATERIAL_THRESHOLD} auf allen Horizonten "
            f"und Vorzeichen konsistent ({'+' if rank_ics[0] > 0 else '-'})"
        )
    reasons: list[str] = []
    if not all_material:
        reasons.append(
            f"Rank-IC nicht auf allen Horizonten > {RANK_IC_MATERIAL_THRESHOLD} "
            f"(values={[round(v, 6) for v in rank_ics]})"
        )
    if not consistent:
        reasons.append(f"Vorzeichen inkonsistent (values={[round(v, 6) for v in rank_ics]})")
    return False, "NEIN — " + "; ".join(reasons)


def main() -> None:
    """Load Yahoo once, run both registered horizons, print table + verdict."""
    settings = load_settings()
    research = settings.research
    request_start = datetime.combine(DATA_START, time.min, tzinfo=UTC)
    request_end = datetime.combine(DATA_END, time.min, tzinfo=UTC)
    clock = FrozenClock(request_end)
    bus = InProcessMessageBus()
    provider = YFinanceProvider(
        settings.symbol_aliases,
        bus,
        DataQualityGate(settings.quality, bus, clock=clock),
        clock=clock,
    )

    # Include SPY as calendar reference even though it is not in the research universe.
    download_symbols = tuple(dict.fromkeys(("SPY", *CANDIDATE_UNIVERSE)))
    print(
        f"loading {len(CANDIDATE_UNIVERSE)} candidate symbols "
        f"({DATA_START} → {DATA_END}, batches of {DOWNLOAD_BATCH_SIZE})..."
    )
    bars, kept, dropped = _download_universe(
        download_symbols,
        provider=provider,
        request_start=request_start,
        request_end=request_end,
        frequency=settings.frequency,
    )
    print(f"candidates: {len(CANDIDATE_UNIVERSE)}")
    print(f"kept after continuous-history (as-of {TRAIN_END}) + quality filter: {len(kept)}")
    print(f"dropped: {len(dropped)}")
    print(
        "NOTE: milder Survivorship-Bias (heutige S&P-500-Mitglieder) — "
        "jedes positive Ergebnis ist PROVISORISCH (Phase-3 CPCV/DSR)."
    )

    feature_matrix = build_momentum_features(bars)
    print(f"feature rows: {len(feature_matrix)}  columns: {list(FEATURE_COLUMNS)}")

    rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        print(f"running horizon={horizon} embargo_days={horizon}...")
        row = run_one_horizon(
            bars,
            feature_matrix,
            horizon=horizon,
            seed=research.seed,
            learning_rate=research.learning_rate,
            num_leaves=research.num_leaves,
            min_data_in_leaf=research.min_data_in_leaf,
            feature_fraction=research.feature_fraction,
            bagging_fraction=research.bagging_fraction,
            bagging_freq=research.bagging_freq,
            num_boost_round=research.num_boost_round,
            ic_positive_threshold=research.ic_positive_threshold,
            min_cs_breadth_for_reliable_ic=research.min_cs_breadth_for_reliable_ic,
        )
        rows.append(row)
        print(
            f"  h={horizon}: ic_mean={row['ic_mean']:.6f} "
            f"rank_ic={row['rank_ic_mean']:.6f} icir={row['icir']}"
        )

    table = format_results_table(rows)
    ok, verdict = verdict_from_rows(rows)
    print()
    print("=== RESULT TABLE ===")
    print(table)
    print()
    print(f"VERDICT: {verdict}")
    print()
    print("=== HANDOFF ===")
    print(f"n_symbols_after_filter: {len(kept)}")
    print(f"train_end: {TRAIN_END.isoformat()}")
    print(f"test_start: {TEST_START.isoformat()}")
    print(f"test_window_end: {TEST_WINDOW_END.isoformat()}")
    print(f"data_start: {DATA_START.isoformat()}")
    print(f"data_end_exclusive: {DATA_END.isoformat()}")
    print(
        "features: mom_63/126/252 with 21d skip; rev_21; vol_63; "
        "cross-sectionally z-scored per date"
    )
    print("labels: make_forward_return_labels horizon in {63,126}; embargo_days=horizon")
    print(
        f"model: train_baseline seed={research.seed} deterministic=True "
        f"num_boost_round={research.num_boost_round}"
    )
    print(f"phase3_justified: {'YES' if ok else 'NO'}")
    # Machine-readable block for Slack posting
    print("=== TICKERS_KEPT_COUNT ===")
    print(len(kept))
    print("=== TICKERS_KEPT ===")
    print(",".join(kept))


if __name__ == "__main__":
    main()
