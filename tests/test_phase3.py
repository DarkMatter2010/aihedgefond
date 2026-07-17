"""Offline, deterministic definition-of-done tests for Phase 3 Slice 1."""

from __future__ import annotations

from datetime import date
from math import sqrt

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from aihedgefund.core.schemas import BaselineDataset, CPCVConfig
from aihedgefund.research.baseline import build_lgbm_params
from aihedgefund.research.cpcv import combinatorial_purged_splits, subset_by_positions
from aihedgefund.research.deflated_sharpe import (
    deflated_sharpe,
    deflated_sharpe_from_moments,
    expected_max_sharpe,
    sharpe_ratio,
)
from aihedgefund.research.gate import run_overfitting_gate, scores_to_strategy_returns

SEED = 20260717
HORIZON = 2


def _mini_dataset(
    *,
    n_days: int = 60,
    n_symbols: int = 4,
    horizon: int = HORIZON,
    seed: int = SEED,
) -> BaselineDataset:
    """Build a tiny deterministic BaselineDataset for CPCV unit tests."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B", tz="UTC")
    # Keep enough trailing room so label ends are mostly in-calendar.
    usable = dates[: n_days - horizon]
    rows: list[tuple[pd.Timestamp, str]] = []
    feat_rows: list[list[float]] = []
    labels: list[float] = []
    feature_columns = ("f0", "f1")
    for _symbol_i, symbol in enumerate([f"S{i}" for i in range(n_symbols)]):
        noise = rng.normal(0.0, 0.01, len(usable))
        for t_i, ts in enumerate(usable):
            rows.append((ts, symbol))
            feat_rows.append(
                [float(noise[t_i]), float(rng.normal(0.0, 1.0))]
            )
            labels.append(float(noise[t_i] * 0.1 + rng.normal(0.0, 0.01)))
    index = pd.MultiIndex.from_tuples(rows, names=("timestamp", "symbol"))
    features = pd.DataFrame(feat_rows, index=index, columns=list(feature_columns), dtype="float64")
    label = pd.Series(labels, index=index, dtype="float64", name=f"forward_return_{horizon}")
    features = features.sort_index()
    label = label.loc[features.index]
    return BaselineDataset(
        features=features,
        label=label,
        horizon=horizon,
        feature_columns=feature_columns,
    )


def test_cpcv_config_hard_fails_when_k_ge_n() -> None:
    """CPCVConfig rejects k >= N at the DTO boundary."""
    with pytest.raises(ValidationError):
        CPCVConfig(n_blocks=3, n_test_blocks=3, embargo_days=2, horizon=2)


def test_purge_removes_label_overlapping_train_samples() -> None:
    """Purge drops train rows whose [t0, t1] overlaps the test span."""
    dataset = _mini_dataset(n_days=40, n_symbols=2, horizon=2)
    config = CPCVConfig(n_blocks=4, n_test_blocks=1, embargo_days=2, horizon=2)
    result = combinatorial_purged_splits(dataset, config)

    features = dataset.features.sort_index()
    row_ts = pd.DatetimeIndex(features.index.get_level_values("timestamp"))
    # Reconstruct t1 the same way CPCV does for the assert.
    from aihedgefund.research.cpcv import _label_end_times

    t1 = _label_end_times(features.index, horizon=2)

    for fold in result.folds:
        test_ts = row_ts[list(fold.test_positions)]
        test_min, test_max = test_ts.min(), test_ts.max()
        train_pos = list(fold.train_positions)
        # No surviving train row may overlap the test span via its label window.
        for pos in train_pos:
            assert not (row_ts[pos] <= test_max and t1[pos] >= test_min), (
                f"fold {fold.fold_id}: train row {pos} overlaps test "
                f"[{test_min}, {test_max}] via t1={t1[pos]}"
            )
        # And at least one row was purged on interior folds (constructed so
        # horizon>0 forces overlap near block edges).
        assert fold.purged_train_count >= 0
        # Constructed case: with horizon=2, interior folds must purge the
        # pre-test label windows that would otherwise leak into training.
        if min(fold.test_block_ids) > 0:
            assert fold.purged_train_count > 0


def test_embargo_zone_boundaries() -> None:
    """Embargo removes the next ``embargo_days`` trading timestamps after test_end.

    Calendar-day embargo would miss Fri→Mon: ``Friday + timedelta(days=1)`` is
    Saturday, so Monday stays in train. Trading-bar embargo must drop Monday
    when ``embargo_days >= 1``.
    """
    dataset = _mini_dataset(n_days=50, n_symbols=2, horizon=2)
    embargo_days = 5
    config = CPCVConfig(
        n_blocks=5,
        n_test_blocks=1,
        embargo_days=embargo_days,
        horizon=2,
    )
    result = combinatorial_purged_splits(dataset, config)
    features = dataset.features.sort_index()
    row_ts = pd.DatetimeIndex(features.index.get_level_values("timestamp"))
    timestamps = pd.DatetimeIndex(row_ts.unique()).sort_values()
    n = len(timestamps)
    base, rem = divmod(n, config.n_blocks)
    sizes = [base + (1 if i < rem else 0) for i in range(config.n_blocks)]
    block_ids = np.repeat(np.arange(config.n_blocks), sizes)

    saw_friday_monday = False
    for fold in result.folds:
        test_block = fold.test_block_ids[0]
        block_mask = block_ids == test_block
        test_end_iloc = int(np.flatnonzero(block_mask)[-1])
        test_end = timestamps[test_end_iloc]
        embargo_start = test_end_iloc + 1
        embargo_stop = min(embargo_start + embargo_days, len(timestamps))
        embargo_ts = set(timestamps[embargo_start:embargo_stop])
        train_set = set(fold.train_positions)
        test_set = set(fold.test_positions)
        for pos, ts in enumerate(row_ts):
            if pos in test_set:
                continue
            if ts in embargo_ts:
                assert pos not in train_set, (
                    f"fold {fold.fold_id}: embargo row {ts} still in train"
                )
                assert fold.embargoed_train_count > 0

        # Friday → Monday: next trading bar after Friday must be embargoed.
        if test_end.dayofweek == 4 and embargo_start < len(timestamps):
            monday = timestamps[embargo_start]
            assert monday.dayofweek == 0, (
                f"expected Monday after Friday test_end={test_end}, got {monday}"
            )
            monday_positions = [
                pos for pos, ts in enumerate(row_ts) if ts == monday and pos not in test_set
            ]
            assert monday_positions, f"no Monday rows for {monday}"
            for pos in monday_positions:
                assert pos not in train_set, (
                    f"fold {fold.fold_id}: Monday {monday} after Friday "
                    f"test_end must be embargoed (trading-bar, not calendar)"
                )
            saw_friday_monday = True

    assert saw_friday_monday, (
        "fixture must include a Friday test_end so Fri→Mon embargo is asserted"
    )


def test_gate_hard_fails_on_undefined_path_sharpe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undefined path Sharpe (<2 obs or zero variance) must hard-fail, not default to 0."""
    dataset = _mini_dataset(n_days=40, n_symbols=4, horizon=2, seed=SEED)
    config = CPCVConfig(n_blocks=3, n_test_blocks=1, embargo_days=2, horizon=2)
    params = build_lgbm_params(
        seed=SEED,
        learning_rate=0.1,
        num_leaves=8,
        min_data_in_leaf=5,
        feature_fraction=1.0,
        bagging_fraction=1.0,
        bagging_freq=0,
    )
    kwargs = dict(
        dataset=dataset,
        cpcv_config=config,
        model_params=params,
        num_boost_round=10,
        n_trials=12,
        seed=SEED,
        universe=("S0", "S1", "S2", "S3"),
        start=date(2024, 1, 2),
        end=date(2024, 3, 29),
        frequency="1d",
    )

    def _zero_variance_returns(
        scores: pd.Series,
        forward_returns: pd.Series,
    ) -> pd.Series:
        ts = scores.index.get_level_values("timestamp").unique().sort_values()
        return pd.Series(
            0.0,
            index=pd.DatetimeIndex(ts, name="timestamp"),
            dtype="float64",
            name="strategy_return",
        )

    monkeypatch.setattr(
        "aihedgefund.research.gate.scores_to_strategy_returns",
        _zero_variance_returns,
    )
    with pytest.raises(ValueError, match=r"fold \d+: path Sharpe undefined"):
        run_overfitting_gate(**kwargs)

    def _single_obs_returns(
        scores: pd.Series,
        forward_returns: pd.Series,
    ) -> pd.Series:
        _ = forward_returns
        ts0 = scores.index.get_level_values("timestamp").unique().sort_values()[0]
        return pd.Series(
            [0.01],
            index=pd.DatetimeIndex([ts0], name="timestamp"),
            dtype="float64",
            name="strategy_return",
        )

    monkeypatch.setattr(
        "aihedgefund.research.gate.scores_to_strategy_returns",
        _single_obs_returns,
    )
    with pytest.raises(ValueError, match=r"fold \d+: path Sharpe undefined"):
        run_overfitting_gate(**kwargs)

def test_dsr_matches_bailey_lopez_de_prado_2014_example() -> None:
    """Sanity-check DSR against the published numerical example.

    Bailey & López de Prado (2014): annualized SR*=2.5, T=1250, N=100,
    V[{SR_ann}]=0.5, skew=-3, kurtosis=10 → DSR ≈ 0.900.
    With N=46 the same inputs give DSR ≈ 0.9505.
    """
    obs_per_year = 250
    sr_nonann = 2.5 / sqrt(obs_per_year)
    var_nonann = 0.5 / obs_per_year
    report_100 = deflated_sharpe_from_moments(
        observed_sharpe=sr_nonann,
        n_obs=1250,
        skewness=-3.0,
        kurtosis=10.0,
        n_trials=100,
        var_trial_sharpes=var_nonann,
    )
    assert report_100.dsr == pytest.approx(0.9004, abs=5e-4)

    report_46 = deflated_sharpe_from_moments(
        observed_sharpe=sr_nonann,
        n_obs=1250,
        skewness=-3.0,
        kurtosis=10.0,
        n_trials=46,
        var_trial_sharpes=var_nonann,
    )
    assert report_46.dsr == pytest.approx(0.9505, abs=5e-4)

    # expected_max_sharpe hard-fails on invalid n_trials.
    with pytest.raises(ValueError, match="n_trials"):
        expected_max_sharpe(1, 0.01)


def test_pure_noise_strategy_dsr_leq_zero() -> None:
    """A strongly losing noise series must yield DSR <= 0 (gate rejects)."""
    rng = np.random.default_rng(SEED)
    # Large negative drift → SR* << 0 << SR0 → Φ underflows to 0.0.
    returns = rng.normal(loc=-0.05, scale=0.01, size=800)
    report = deflated_sharpe(returns, n_trials=12, var_trial_sharpes=0.002)
    assert report.dsr <= 0.0
    assert report.observed_sharpe < 0.0


def test_true_signal_strategy_dsr_gt_zero() -> None:
    """A strong positive-edge series must yield DSR > 0 (gate accepts)."""
    rng = np.random.default_rng(SEED + 1)
    returns = rng.normal(loc=0.02, scale=0.01, size=800)
    report = deflated_sharpe(returns, n_trials=12, var_trial_sharpes=0.0005)
    assert report.dsr > 0.0
    assert report.observed_sharpe > report.sr0


def test_sharpe_ratio_hard_fails_on_zero_variance() -> None:
    """Zero-variance returns are a hard fail, not a silent inf."""
    with pytest.raises(ValueError, match="non-zero"):
        sharpe_ratio([0.1, 0.1, 0.1])


def test_scores_to_strategy_returns_schema() -> None:
    """Strategy returns are a timestamp-indexed float series."""
    idx = pd.MultiIndex.from_product(
        [
            pd.date_range("2024-01-02", periods=3, freq="B", tz="UTC"),
            ["A", "B"],
        ],
        names=("timestamp", "symbol"),
    )
    scores = pd.Series([1.0, -1.0, 0.5, -0.5, 2.0, -2.0], index=idx, dtype="float64")
    fwd = pd.Series([0.01, -0.01, 0.02, -0.02, 0.03, -0.03], index=idx, dtype="float64")
    out = scores_to_strategy_returns(scores, fwd)
    assert list(out.index.names) == ["timestamp"]
    assert len(out) == 3
    assert out.dtype == np.float64


def test_gate_verdict_schema_and_reproducibility() -> None:
    """Full gate returns a validated GateVerdict and is seed-stable."""
    dataset = _mini_dataset(n_days=80, n_symbols=4, horizon=2, seed=SEED)
    config = CPCVConfig(n_blocks=4, n_test_blocks=1, embargo_days=2, horizon=2)
    params = build_lgbm_params(
        seed=SEED,
        learning_rate=0.1,
        num_leaves=8,
        min_data_in_leaf=5,
        feature_fraction=1.0,
        bagging_fraction=1.0,
        bagging_freq=0,
    )
    kwargs = dict(
        dataset=dataset,
        cpcv_config=config,
        model_params=params,
        num_boost_round=20,
        n_trials=12,
        seed=SEED,
        universe=("S0", "S1", "S2", "S3"),
        start=date(2024, 1, 2),
        end=date(2024, 4, 30),
        frequency="1d",
    )
    first = run_overfitting_gate(**kwargs)
    second = run_overfitting_gate(**kwargs)
    assert first.verdict in {"JA", "NEIN"}
    assert first.verdict == ("JA" if first.dsr > 0.0 else "NEIN")
    assert first.n_trials == 12
    assert first.cpcv.n_folds == 4
    assert first.dsr == second.dsr
    assert first.path_sharpe_mean == second.path_sharpe_mean
    assert len(first.path_results) == first.cpcv.n_folds


def test_subset_by_positions_preserves_schema() -> None:
    """iloc subsetting keeps BaselineDataset invariants."""
    dataset = _mini_dataset(n_days=30, n_symbols=2)
    config = CPCVConfig(n_blocks=3, n_test_blocks=1, embargo_days=2, horizon=2)
    fold = combinatorial_purged_splits(dataset, config).folds[0]
    train = subset_by_positions(dataset, fold.train_positions)
    assert train.horizon == dataset.horizon
    assert train.feature_columns == dataset.feature_columns
    assert len(train.features) == len(fold.train_positions)
