"""Offline, deterministic definition-of-done tests for Phase 2."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from aihedgefund.core.bus import InProcessMessageBus
from aihedgefund.core.config import load_settings
from aihedgefund.core.schemas import BarFrame, BaselineDataset, Phase2Sidecar
from aihedgefund.features.pipeline import FEATURE_COLUMNS, FeaturePipeline
from aihedgefund.research.adapters.filesystem import FilesystemModelArtifactAdapter
from aihedgefund.research.baseline import (
    build_lgbm_params,
    predict_scores,
    train_baseline,
)
from aihedgefund.research.dataset import assemble_baseline_dataset
from aihedgefund.research.forward_labels import make_forward_return_labels
from aihedgefund.research.metrics import BREADTH_WARNING, compute_ic_metrics
from aihedgefund.research.model_hash import compute_model_hash
from aihedgefund.research.run_baseline import (
    SIDECAR_FILENAME,
    build_hyperparams,
    load_sidecar,
    run_baseline,
)
from aihedgefund.research.split import time_embargo_split

SEED = 20260716
HORIZON = 5
CREATED_AT = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)


def _synthetic_symbol_frame(symbol_seed: int, rows: int = 160) -> pd.DataFrame:
    """Deterministic business-day GBM bars for one symbol."""
    rng = np.random.default_rng(symbol_seed)
    index = pd.date_range("2024-01-02", periods=rows, freq="B", tz="UTC")
    returns = rng.normal(0.0005, 0.015, rows)
    close = 100.0 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0] * 0.999, close[:-1]]
    high = np.maximum(open_, close) * (1.001 + rng.uniform(0.0, 0.002, rows))
    low = np.minimum(open_, close) * (0.999 - rng.uniform(0.0, 0.002, rows))
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": rng.integers(1_000_000, 5_000_000, rows).astype(float),
        },
        index=index,
    )


def multi_symbol_bars(symbols: tuple[str, ...] = ("AAA", "BBB", "CCC", "DDD")) -> BarFrame:
    """Build a multi-symbol BarFrame with zero corporate actions."""
    frames = {
        symbol: _synthetic_symbol_frame(SEED + idx)
        for idx, symbol in enumerate(symbols)
    }
    return BarFrame(
        bars=frames,
        dividends={symbol: pd.Series(0.0, index=frame.index) for symbol, frame in frames.items()},
        splits={symbol: pd.Series(0.0, index=frame.index) for symbol, frame in frames.items()},
    )


def phase2_settings(tmp_path: Path):
    """Settings with split dates aligned to the synthetic 2024 fixture."""
    base = load_settings()
    research = base.research.model_copy(
        update={
            "horizon": HORIZON,
            "embargo_days": HORIZON,
            "seed": SEED,
            "train_end": date(2024, 4, 30),
            "test_start": date(2024, 5, 8),
            "num_boost_round": 25,
            "min_data_in_leaf": 5,
            "ic_positive_threshold": 0.02,
            "min_cs_breadth_for_reliable_ic": 30,
            "strategy_id": "phase2-test-baseline",
        }
    )
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    return base.model_copy(
        update={
            "start": date(2024, 1, 1),
            "end": date(2024, 12, 31),
            "universe": ("AAA", "BBB", "CCC", "DDD"),
            "artifact_root": artifact_root,
            "research": research,
        }
    )


def test_research_settings_loaded_from_yaml() -> None:
    settings = load_settings()
    assert settings.research.horizon == 5
    assert settings.research.embargo_days >= settings.research.horizon
    assert settings.research.seed == 42
    assert settings.research.strategy_id == "phase2-lgbm-baseline"
    assert len(settings.universe) == 50
    assert len(settings.universe) >= settings.research.min_cs_breadth_for_reliable_ic


def _nyse_new_year_holiday_dataset() -> BaselineDataset:
    """Business-day bars around Dec-2022/Jan-2023 without Observed New Year.

    2023-01-02 was an Observed New Year holiday (Monday). A plain ``freq="B"``
    index would include it and understate the trading-bar gap.
    """
    bdays = pd.bdate_range("2022-12-01", "2023-01-31", tz="UTC")
    holiday = pd.Timestamp("2023-01-02", tz="UTC")
    assert holiday in bdays
    calendar = bdays.delete(bdays.get_loc(holiday))
    assert holiday not in calendar

    symbol = "AAA"
    n = len(calendar)
    multi = pd.MultiIndex.from_arrays(
        [calendar, [symbol] * n],
        names=["timestamp", "symbol"],
    )
    features = pd.DataFrame(
        {column: np.linspace(0.0, 1.0, n, dtype=np.float64) for column in FEATURE_COLUMNS},
        index=multi,
    )
    label = pd.Series(np.zeros(n, dtype=np.float64), index=multi, name="forward_return_5")
    return BaselineDataset(
        features=features,
        label=label,
        horizon=HORIZON,
        feature_columns=FEATURE_COLUMNS,
    )


def test_production_split_survives_nyse_new_year_holiday() -> None:
    """Production train_end/test_start must keep bar_gap > horizon across 2023-01-02.

    Empirically: train_end=2022-12-30 + test_start=2023-01-09 → bar_gap=5 == horizon
    (leakage); test_start=2023-01-10 → bar_gap=6 > horizon.
    """
    settings = load_settings()
    assert settings.research.train_end == date(2022, 12, 30)
    assert settings.research.test_start == date(2023, 1, 10)

    dataset = _nyse_new_year_holiday_dataset()
    train, test, split_def = time_embargo_split(
        dataset,
        train_end=date(2022, 12, 30),
        test_start=date(2023, 1, 10),
        embargo_days=HORIZON,
        horizon=HORIZON,
    )
    assert split_def.train_rows > 0
    assert split_def.test_rows > 0
    assert set(train.features.index).isdisjoint(set(test.features.index))

    with pytest.raises(ValueError, match="bar gap"):
        time_embargo_split(
            dataset,
            train_end=date(2022, 12, 30),
            test_start=date(2023, 1, 9),
            embargo_days=HORIZON,
            horizon=HORIZON,
        )


def test_research_settings_hard_fail_when_embargo_lt_horizon(tmp_path: Path) -> None:
    config = Path("src/aihedgefund/config/limits.yaml").read_text(encoding="utf-8")
    invalid = tmp_path / "bad_research.yaml"
    invalid.write_text(
        config.replace("embargo_days: 5", "embargo_days: 2"),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="embargo_days"):
        load_settings(invalid)


def test_forward_return_label_formula_and_tail_drop() -> None:
    bars = multi_symbol_bars(("AAA", "BBB"))
    labels, meta = make_forward_return_labels(bars, horizon=HORIZON)

    assert meta.horizon == HORIZON
    assert meta.close_adj_source == "as_of_adjusted"
    assert labels.isna().sum() == 0
    assert list(labels.index.names) == ["timestamp", "symbol"]

    for symbol, frame in bars.bars.items():
        symbol_labels = labels.xs(symbol, level="symbol")
        assert len(symbol_labels) == len(frame) - HORIZON
        last_kept = frame.index[-(HORIZON + 1)]
        assert symbol_labels.index.max() == last_kept
        for missing_ts in frame.index[-HORIZON:]:
            assert missing_ts not in symbol_labels.index

        close = frame["close"].astype(float)
        expected = close.shift(-HORIZON) / close - 1.0
        expected = expected.iloc[:-HORIZON]
        pd.testing.assert_series_equal(
            symbol_labels,
            expected,
            check_names=False,
        )


def test_embargo_split_rejects_overlapping_label_windows() -> None:
    bars = multi_symbol_bars()
    bus = InProcessMessageBus()
    features = FeaturePipeline(bus).compute(bars)
    labels, _ = make_forward_return_labels(bars, horizon=HORIZON)
    dataset = assemble_baseline_dataset(features, labels, horizon=HORIZON)

    with pytest.raises(ValueError, match="calendar gap|overlap|bar gap"):
        time_embargo_split(
            dataset,
            train_end=date(2024, 5, 15),
            test_start=date(2024, 5, 16),
            embargo_days=HORIZON,
            horizon=HORIZON,
        )


def test_embargo_split_rejects_bar_gap_equal_horizon() -> None:
    """calendar_gap can exceed embargo while trading-bar gap == horizon.

    Business-day example: train_end=2024-04-26, test_start=2024-05-03,
    horizon=5 → calendar_gap=7 >= 5, but bar_gap=5 == horizon (leakage).
    """
    bars = multi_symbol_bars()
    bus = InProcessMessageBus()
    features = FeaturePipeline(bus).compute(bars)
    labels, _ = make_forward_return_labels(bars, horizon=HORIZON)
    dataset = assemble_baseline_dataset(features, labels, horizon=HORIZON)

    with pytest.raises(ValueError, match="bar gap"):
        time_embargo_split(
            dataset,
            train_end=date(2024, 4, 26),
            test_start=date(2024, 5, 3),
            embargo_days=HORIZON,
            horizon=HORIZON,
        )


def test_embargo_split_has_no_label_window_overlap() -> None:
    bars = multi_symbol_bars()
    bus = InProcessMessageBus()
    features = FeaturePipeline(bus).compute(bars)
    labels, _ = make_forward_return_labels(bars, horizon=HORIZON)
    dataset = assemble_baseline_dataset(features, labels, horizon=HORIZON)

    train, test, split_def = time_embargo_split(
        dataset,
        train_end=date(2024, 4, 30),
        test_start=date(2024, 5, 8),
        embargo_days=HORIZON,
        horizon=HORIZON,
    )

    assert split_def.train_rows > 0
    assert split_def.test_rows > 0
    full_index = dataset.features.index
    for symbol in sorted(set(train.features.index.get_level_values("symbol"))):
        full_symbol_ts = pd.DatetimeIndex(
            full_index.get_level_values("timestamp")[
                full_index.get_level_values("symbol") == symbol
            ].unique()
        ).sort_values()
        max_tr = train.features.xs(symbol, level="symbol").index.max()
        min_te = test.features.xs(symbol, level="symbol").index.min()
        train_pos = full_symbol_ts.get_loc(max_tr)
        test_pos = full_symbol_ts.get_loc(min_te)
        assert isinstance(train_pos, int) and isinstance(test_pos, int)
        assert test_pos - train_pos > HORIZON
    assert set(train.features.index).isdisjoint(set(test.features.index))


def test_determinism_identical_predictions_and_model_hash(tmp_path: Path) -> None:
    bars = multi_symbol_bars()
    settings = phase2_settings(tmp_path)
    research = settings.research
    bus = InProcessMessageBus()
    features = FeaturePipeline(bus).compute(bars)
    labels, _ = make_forward_return_labels(bars, horizon=research.horizon)
    dataset = assemble_baseline_dataset(features, labels, horizon=research.horizon)
    train, test, _ = time_embargo_split(
        dataset,
        train_end=research.train_end,
        test_start=research.test_start,
        embargo_days=research.embargo_days,
        horizon=research.horizon,
    )
    hyperparams = build_hyperparams(research)
    model_hash = compute_model_hash(
        features=FEATURE_COLUMNS,
        hyperparameters=hyperparams,
        universe=settings.universe,
        start=settings.start,
        end=settings.end,
        frequency=settings.frequency,
        seed=research.seed,
    )
    params = {k: v for k, v in hyperparams.items() if k != "num_boost_round"}

    first = train_baseline(train, params=params, num_boost_round=research.num_boost_round)
    second = train_baseline(train, params=params, num_boost_round=research.num_boost_round)
    pred_a = predict_scores(first, test.features, model_hash=model_hash)
    pred_b = predict_scores(second, test.features, model_hash=model_hash)

    assert pred_a.model_hash == pred_b.model_hash == model_hash
    np.testing.assert_array_equal(pred_a.scores.to_numpy(), pred_b.scores.to_numpy())


def test_ic_sanity_signal_positive_noise_near_zero() -> None:
    index = pd.MultiIndex.from_product(
        [
            pd.date_range("2024-06-03", periods=40, freq="B", tz="UTC"),
            ("AAA", "BBB", "CCC", "DDD", "EEE"),
        ],
        names=("timestamp", "symbol"),
    )
    rng = np.random.default_rng(SEED)
    latent = pd.Series(rng.normal(0.0, 1.0, len(index)), index=index)
    forward = latent + rng.normal(0.0, 0.05, len(index))
    signal_scores = latent
    noise_scores = pd.Series(rng.normal(0.0, 1.0, len(index)), index=index)

    signal_metrics = compute_ic_metrics(
        signal_scores,
        forward,
        ic_positive_threshold=0.02,
        min_cs_breadth_for_reliable_ic=30,
    )
    noise_metrics = compute_ic_metrics(
        noise_scores,
        forward,
        ic_positive_threshold=0.02,
        min_cs_breadth_for_reliable_ic=30,
    )

    assert signal_metrics.ic_mean > 0.5
    assert signal_metrics.rank_ic_mean > 0.5
    assert signal_metrics.ic_materially_positive is True
    assert abs(noise_metrics.ic_mean) < 0.2
    assert signal_metrics.cs_breadth_warning is True
    assert BREADTH_WARNING in signal_metrics.warnings


def test_artifact_roundtrip_predictions_match(tmp_path: Path) -> None:
    bars = multi_symbol_bars()
    settings = phase2_settings(tmp_path)
    adapter = FilesystemModelArtifactAdapter(settings.artifact_root)
    pipeline = FeaturePipeline(InProcessMessageBus())

    (
        _train,
        test,
        _split,
        predictions,
        metrics,
        artifact_dir,
        sidecar,
    ) = run_baseline(
        bars,
        settings,
        feature_pipeline=pipeline,
        artifact_adapter=adapter,
        created_at=CREATED_AT,
        git_commit="test-commit-phase2",
    )

    assert (artifact_dir / "model.txt").is_file()
    assert (artifact_dir / "metadata.json").is_file()
    assert (artifact_dir / SIDECAR_FILENAME).is_file()
    assert isinstance(sidecar, Phase2Sidecar)
    assert sidecar.horizon == HORIZON
    assert sidecar.git_commit == "test-commit-phase2"
    assert sidecar.feature_list == FEATURE_COLUMNS
    assert "lightgbm" in sidecar.lib_versions
    assert metrics.median_cs_breadth >= 1.0

    loaded_model, loaded_meta = adapter.load_booster(predictions.model_hash)
    assert loaded_meta.model_hash == predictions.model_hash
    reloaded = predict_scores(
        loaded_model,
        test.features,
        model_hash=predictions.model_hash,
    )
    np.testing.assert_array_equal(
        predictions.scores.to_numpy(),
        reloaded.scores.to_numpy(),
    )
    loaded_sidecar = load_sidecar(artifact_dir)
    assert loaded_sidecar.model_hash == sidecar.model_hash
    assert loaded_sidecar.metrics.ic_mean == sidecar.metrics.ic_mean


def test_run_baseline_is_deterministic_across_calls(tmp_path: Path) -> None:
    bars = multi_symbol_bars()
    settings_a = phase2_settings(tmp_path / "a")
    settings_b = phase2_settings(tmp_path / "b")

    result_a = run_baseline(
        bars,
        settings_a,
        feature_pipeline=FeaturePipeline(InProcessMessageBus()),
        artifact_adapter=FilesystemModelArtifactAdapter(settings_a.artifact_root),
        created_at=CREATED_AT,
        git_commit="determinism-check",
    )
    result_b = run_baseline(
        bars,
        settings_b,
        feature_pipeline=FeaturePipeline(InProcessMessageBus()),
        artifact_adapter=FilesystemModelArtifactAdapter(settings_b.artifact_root),
        created_at=CREATED_AT,
        git_commit="determinism-check",
    )

    _train_a, _test_a, _split_a, pred_a, metrics_a, _dir_a, sidecar_a = result_a
    _train_b, _test_b, _split_b, pred_b, metrics_b, _dir_b, sidecar_b = result_b

    assert pred_a.model_hash == pred_b.model_hash
    assert sidecar_a.model_hash == sidecar_b.model_hash
    np.testing.assert_array_equal(pred_a.scores.to_numpy(), pred_b.scores.to_numpy())
    assert metrics_a.ic_mean == metrics_b.ic_mean
    assert metrics_a.rank_ic_mean == metrics_b.rank_ic_mean
    assert metrics_a.icir == metrics_b.icir


def test_phase2_modules_do_not_import_yfinance() -> None:
    import aihedgefund.research.baseline as baseline
    import aihedgefund.research.dataset as dataset
    import aihedgefund.research.forward_labels as forward_labels
    import aihedgefund.research.metrics as metrics
    import aihedgefund.research.run_baseline as run_mod
    import aihedgefund.research.split as split

    for module in (baseline, dataset, forward_labels, metrics, run_mod, split):
        assert "yfinance" not in module.__dict__
        assert "yfinance" not in getattr(module, "__file__", "")


def test_build_lgbm_params_are_deterministic_regressor() -> None:
    params = build_lgbm_params(
        seed=SEED,
        learning_rate=0.05,
        num_leaves=31,
        min_data_in_leaf=5,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
    )
    assert params["objective"] == "regression"
    assert params["deterministic"] is True
    assert params["force_col_wise"] is True
    assert params["num_threads"] == 1
    assert params["seed"] == SEED
