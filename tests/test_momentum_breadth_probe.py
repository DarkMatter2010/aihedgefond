"""Offline tests for Phase-2 momentum breadth probe helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aihedgefund.core.schemas import BarFrame

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_momentum_breadth_probe.py"
_SPEC = importlib.util.spec_from_file_location("run_momentum_breadth_probe", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe)


def _ohlcv(close: np.ndarray, *, start: str = "2019-01-02") -> pd.DataFrame:
    index = pd.date_range(start, periods=len(close), freq="B", tz="UTC")
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.001,
            "low": np.minimum(open_, close) * 0.999,
            "close": close,
            "adj_close": close,
            "volume": np.full(len(close), 1_000_000.0),
        },
        index=index,
    )


def _bars_from_closes(closes: dict[str, np.ndarray]) -> BarFrame:
    frames = {symbol: _ohlcv(values) for symbol, values in closes.items()}
    return BarFrame(
        bars=frames,
        dividends={s: pd.Series(0.0, index=f.index) for s, f in frames.items()},
        splits={s: pd.Series(0.0, index=f.index) for s, f in frames.items()},
    )


def test_momentum_features_point_in_time_truncation() -> None:
    """Truncating history after t must not change mom/rev/vol values at t."""
    rows = 400
    rng = np.random.default_rng(42)
    base = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, rows)))
    closes = {
        "AAA": base,
        "BBB": base * 1.05 + rng.normal(0.0, 0.3, rows),
        "CCC": base * 0.97 + rng.normal(0.0, 0.25, rows),
    }
    bars = _bars_from_closes(closes)
    full = probe.build_momentum_features(bars)

    # Anchor after full lookback (252 + 21) so all FEATURE_COLUMNS are finite.
    anchor = bars.bars["AAA"].index[320]
    for column in probe.FEATURE_COLUMNS:
        assert pd.notna(full.loc[(anchor, "AAA"), column])

    truncated = probe.build_momentum_features(
        BarFrame(
            bars={s: f.loc[:anchor] for s, f in bars.bars.items()},
            dividends={s: d.loc[:anchor] for s, d in bars.dividends.items()},
            splits={s: sp.loc[:anchor] for s, sp in bars.splits.items()},
        )
    )
    for column in probe.FEATURE_COLUMNS:
        assert full.loc[(anchor, "AAA"), column] == pytest.approx(
            truncated.loc[(anchor, "AAA"), column]
        )


def test_momentum_features_ignore_post_anchor_spike() -> None:
    """Feature values at t must ignore a synthetic close spike strictly after t."""
    rows = 400
    rng = np.random.default_rng(7)
    wave = np.sin(np.linspace(0.0, 10.0 * np.pi, rows))
    close = 100.0 + 2.0 * wave + rng.normal(0.0, 0.05, rows)
    bars = _bars_from_closes({"AAA": close, "BBB": close * 1.01, "CCC": close * 0.99})
    anchor_pos = 320
    anchor = bars.bars["AAA"].index[anchor_pos]
    baseline = probe.build_momentum_features(bars)

    spiked = close.copy()
    spiked[anchor_pos + 1 :] = close[anchor_pos + 1 :] * 3.0
    spiked_bars = _bars_from_closes({"AAA": spiked, "BBB": close * 1.01, "CCC": close * 0.99})
    after = probe.build_momentum_features(spiked_bars)

    for column in probe.FEATURE_COLUMNS:
        assert baseline.loc[(anchor, "AAA"), column] == pytest.approx(
            after.loc[(anchor, "AAA"), column]
        )


def test_continuity_filter_allows_oos_gap_after_train_end() -> None:
    """Symbols with gaps only after TRAIN_END must remain eligible (no OOS look-ahead)."""
    # Build a calendar spanning well before TRAIN_END through DATA_END.
    full_index = pd.date_range("2019-01-02", "2024-12-31", freq="B", tz="UTC")
    assert full_index[full_index <= pd.Timestamp(probe.TRAIN_END, tz="UTC")].size > 300

    continuous = pd.Series(100.0, index=full_index)
    gappy = continuous.copy()
    # Gap strictly inside the OOS window (after TEST_START).
    oos_gap_start = pd.Timestamp("2024-06-03", tz="UTC")
    oos_gap_end = pd.Timestamp("2024-06-14", tz="UTC")
    gappy.loc[oos_gap_start:oos_gap_end] = np.nan

    continuous_frame = _ohlcv(continuous.to_numpy(), start="2019-01-02")
    continuous_frame.index = full_index
    gappy_frame = continuous_frame.copy()
    gappy_frame["close"] = gappy
    gappy_frame["adj_close"] = gappy
    gappy_frame["open"] = gappy
    gappy_frame["high"] = gappy
    gappy_frame["low"] = gappy

    continuity_calendar = full_index[full_index <= pd.Timestamp(probe.TRAIN_END, tz="UTC")]
    full_calendar = full_index[
        (full_index >= pd.Timestamp(probe.DATA_START, tz="UTC"))
        & (full_index < pd.Timestamp(probe.DATA_END, tz="UTC"))
    ]

    assert probe._has_continuous_history(gappy_frame, calendar=continuity_calendar)
    assert not probe._has_continuous_history(gappy_frame, calendar=full_calendar)
    assert probe._has_continuous_history(continuous_frame, calendar=continuity_calendar)
    assert probe._has_continuous_history(continuous_frame, calendar=full_calendar)


@pytest.mark.parametrize(
    ("rows", "expect_ok", "needle"),
    [
        (
            [
                {"rank_ic_mean": 0.03},
                {"rank_ic_mean": 0.04},
            ],
            True,
            "JA",
        ),
        (
            [
                {"rank_ic_mean": 0.03},
                {"rank_ic_mean": -0.04},
            ],
            False,
            "Vorzeichen",
        ),
        (
            [
                {"rank_ic_mean": 0.01},
                {"rank_ic_mean": 0.015},
            ],
            False,
            "nicht auf allen Horizonten",
        ),
        (
            [
                {"rank_ic_mean": -0.03},
                {"rank_ic_mean": -0.04},
            ],
            False,
            "nicht auf allen Horizonten",
        ),
        ([], False, "keine Ergebnisse"),
    ],
)
def test_verdict_from_rows_threshold_and_sign(
    rows: list[dict[str, object]],
    expect_ok: bool,
    needle: str,
) -> None:
    ok, message = probe.verdict_from_rows(rows)
    assert ok is expect_ok
    assert needle in message
