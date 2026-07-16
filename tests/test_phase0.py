"""Definition-of-done tests for Phase 0."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import lightgbm as lgb
import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from aihedgefund.core.bus import InProcessMessageBus
from aihedgefund.core.config import Settings, load_settings
from aihedgefund.core.ports import BrokerPort, DataVendorPort, ModelArtifactPort
from aihedgefund.core.schemas import (
    Event,
    FeatureValue,
    FeatureVector,
    Fill,
    ModelArtifactLoadResult,
    ModelArtifactMetadata,
    ModelArtifactSaveRequest,
    Order,
    OrderSide,
    OrderType,
    RiskCheck,
    Signal,
)
from aihedgefund.research.adapters.filesystem import FilesystemModelArtifactAdapter
from aihedgefund.research.model_hash import compute_model_hash

NOW = datetime(2026, 1, 2, 15, 30, tzinfo=UTC)
ORDER_ID = UUID("00000000-0000-4000-8000-000000000001")
ARTIFACT_CREATED_AT = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)
FEATURE_NAMES = ("momentum_20", "rsi_14")
ARTIFACT_SEED = 42


class SignalGenerated(Event):
    """Dummy integration event carrying a typed payload."""

    signal: Signal


class LogSubscriber:
    """Test subscriber recording deterministic delivery order."""

    def __init__(self) -> None:
        self.received: list[SignalGenerated] = []

    def __call__(self, event: SignalGenerated) -> None:
        self.received.append(event)


def make_signal() -> Signal:
    """Create a valid deterministic signal."""
    return Signal(
        signal_id=UUID("00000000-0000-4000-8000-000000000002"),
        symbol="AAPL",
        timestamp=NOW,
        value=0.75,
        model_version="phase0-test",
    )


def make_order() -> Order:
    """Create a valid deterministic command."""
    return Order(
        order_id=ORDER_ID,
        symbol="AAPL",
        timestamp=NOW,
        side=OrderSide.BUY,
        quantity=Decimal("10"),
        order_type=OrderType.MARKET,
    )


def make_fill() -> Fill:
    """Create a valid deterministic fill."""
    return Fill(
        fill_id=UUID("00000000-0000-4000-8000-000000000003"),
        order_id=ORDER_ID,
        symbol="AAPL",
        fill_price=Decimal("195.25"),
        filled_qty=Decimal("10"),
        timestamp=NOW,
    )


def make_risk_check() -> RiskCheck:
    """Create a valid deterministic risk result."""
    return RiskCheck(
        risk_check_id=UUID("00000000-0000-4000-8000-000000000004"),
        order_id=ORDER_ID,
        passed=False,
        violated_limits=("max_position_size",),
        timestamp=NOW,
    )


def make_feature_vector() -> FeatureVector:
    """Create a valid deterministic feature vector."""
    return FeatureVector(
        feature_vector_id=UUID("00000000-0000-4000-8000-000000000005"),
        symbol="AAPL",
        timestamp=NOW,
        features=(
            FeatureValue(name="momentum_20d", value=1.25),
            FeatureValue(name="volatility_20d", value=0.18),
        ),
        feature_set_version="phase0-test",
    )


def test_signal_generated_reaches_log_subscriber() -> None:
    bus = InProcessMessageBus()
    log = LogSubscriber()
    signal = make_signal()
    event = SignalGenerated(timestamp=NOW, signal=signal)

    bus.subscribe_event(SignalGenerated, log)
    bus.publish_event(event)

    assert log.received == [event]
    assert log.received[0].signal == signal


def test_command_and_event_channels_are_isolated() -> None:
    bus = InProcessMessageBus()
    commands: list[Order] = []
    events: list[Signal] = []
    order = make_order()
    signal = make_signal()

    bus.subscribe_command(Order, commands.append)
    bus.subscribe_event(Signal, events.append)
    bus.publish_command(order)

    assert commands == [order]
    assert events == []

    bus.publish_event(signal)

    assert commands == [order]
    assert events == [signal]

    with pytest.raises(TypeError):
        bus.publish_event(order)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        bus.publish_command(signal)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "factory",
    [make_signal, make_order, make_fill, make_risk_check, make_feature_vector],
)
def test_required_schemas_instantiate_and_are_frozen(
    factory: Callable[[], BaseModel],
) -> None:
    dto = factory()

    assert isinstance(dto, BaseModel)
    with pytest.raises(ValidationError, match="frozen"):
        dto.timestamp = NOW  # type: ignore[attr-defined,misc]


@pytest.mark.parametrize(
    "schema, invalid_data",
    [
        (
            Signal,
            {
                "symbol": "AAPL",
                "timestamp": NOW,
                "value": "not-a-number",
                "model_version": "v1",
            },
        ),
        (
            Order,
            {
                "symbol": "AAPL",
                "timestamp": NOW,
                "side": OrderSide.BUY,
                "quantity": "10",
                "order_type": OrderType.MARKET,
            },
        ),
        (
            Fill,
            {
                "order_id": ORDER_ID,
                "symbol": "AAPL",
                "fill_price": Decimal("1"),
                "filled_qty": "10",
                "timestamp": NOW,
            },
        ),
        (
            RiskCheck,
            {
                "order_id": ORDER_ID,
                "passed": "yes",
                "violated_limits": (),
                "timestamp": NOW,
            },
        ),
        (
            FeatureVector,
            {
                "symbol": "AAPL",
                "timestamp": NOW,
                "features": ({"name": "momentum", "value": "high"},),
                "feature_set_version": "v1",
            },
        ),
    ],
)
def test_required_schemas_reject_wrong_types(
    schema: type[BaseModel],
    invalid_data: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        schema.model_validate(invalid_data)


def configured_max_position(settings: Settings) -> float:
    """Stand-in module that consumes only typed configuration."""
    return settings.trading_limits.max_position_size


def test_config_loader_exposes_typed_trading_limit() -> None:
    settings = load_settings()

    assert isinstance(settings, Settings)
    assert configured_max_position(settings) == 100000.0
    assert settings.universe == ("AAPL", "MSFT", "SPY")
    assert settings.feature_flags["audit_logging"] is True


def test_config_loader_hard_fails_for_invalid_root(tmp_path: Path) -> None:
    invalid_config = tmp_path / "invalid.yaml"
    invalid_config.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a mapping"):
        load_settings(invalid_config)


@pytest.mark.parametrize("port", [DataVendorPort, BrokerPort])
def test_ports_are_abstract(port: type[object]) -> None:
    with pytest.raises(TypeError):
        port()


def _hash_inputs() -> dict[str, object]:
    return {
        "features": FEATURE_NAMES,
        "hyperparameters": {"learning_rate": 0.05, "num_leaves": 31},
        "universe": ("MSFT", "AAPL"),
        "start": date(2015, 1, 1),
        "end": date(2026, 1, 1),
        "frequency": "1d",
        "seed": ARTIFACT_SEED,
    }


def _train_dummy_booster(
    feature_names: tuple[str, ...] = FEATURE_NAMES,
) -> lgb.Booster:
    rng = np.random.default_rng(ARTIFACT_SEED)
    features = rng.normal(size=(64, len(feature_names)))
    labels = (features[:, 0] + 0.5 * features[:, 1] > 0).astype(np.float64)
    dataset = lgb.Dataset(
        features,
        label=labels,
        feature_name=list(feature_names),
        free_raw_data=False,
    )
    return lgb.train(
        {
            "objective": "binary",
            "verbosity": -1,
            "seed": ARTIFACT_SEED,
            "deterministic": True,
            "num_threads": 1,
        },
        dataset,
        num_boost_round=5,
    )


def _make_metadata(model_hash: str | None = None) -> ModelArtifactMetadata:
    inputs = _hash_inputs()
    resolved_hash = model_hash or compute_model_hash(**inputs)  # type: ignore[arg-type]
    return ModelArtifactMetadata(
        model_hash=resolved_hash,
        strategy_id="phase0-dummy",
        created_at=ARTIFACT_CREATED_AT,
        universe=("AAPL", "MSFT"),
        features=FEATURE_NAMES,
        hyperparameters={"learning_rate": 0.05, "num_leaves": 31},
        seed=ARTIFACT_SEED,
        start=date(2015, 1, 1),
        end=date(2026, 1, 1),
        frequency="1d",
        model_format="lightgbm_native",
        model_file="model.txt",
        phase=0,
    )


def test_settings_expose_artifact_root_from_yaml() -> None:
    settings = load_settings()

    assert settings.artifact_root == Path("artifacts/")


def test_settings_hard_fail_when_artifact_root_missing(tmp_path: Path) -> None:
    config_path = tmp_path / "limits.yaml"
    config_path.write_text(
        "\n".join(
            [
                "trading_limits:",
                "  max_position_size: 100000.0",
                "  max_sector_exposure: 0.25",
                "  max_turnover: 0.20",
                "  max_daily_drawdown: 0.05",
                "universe: [AAPL]",
                "feature_flags: {audit_logging: true}",
                "start: 2015-01-01",
                "end: 2026-01-01",
                "frequency: 1d",
                "symbol_aliases: {}",
                "quality:",
                "  max_nan_ratio: 0.0",
                "  max_abs_logret: 0.30",
                "  stale_bars: 5",
                "  zscore_cap: 8.0",
                "  max_last_bar_age_days: 7",
                "labels:",
                "  vol_span: 20",
                "  cusum_threshold: 0.02",
                "  pt: 1.0",
                "  sl: 1.0",
                "  vertical_bars: 10",
                "fracdiff:",
                "  d: 0.40",
                "  thresh: 0.0001",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="artifact_root"):
        load_settings(config_path)


def test_model_artifact_metadata_is_frozen_and_utc() -> None:
    metadata = _make_metadata()

    assert metadata.model_format == "lightgbm_native"
    assert metadata.seed == ARTIFACT_SEED
    assert metadata.created_at.tzinfo == UTC
    with pytest.raises(ValidationError, match="frozen"):
        metadata.phase = 1  # type: ignore[misc]


def test_model_artifact_port_is_abstract() -> None:
    with pytest.raises(TypeError):
        ModelArtifactPort()  # type: ignore[misc]


def test_model_hash_is_deterministic_across_calls() -> None:
    inputs = _hash_inputs()
    first = compute_model_hash(**inputs)  # type: ignore[arg-type]
    second = compute_model_hash(**inputs)  # type: ignore[arg-type]
    same_order_different_hyperparam_key_order = compute_model_hash(
        features=FEATURE_NAMES,
        hyperparameters={"num_leaves": 31, "learning_rate": 0.05},
        universe=("AAPL", "MSFT"),
        start=date(2015, 1, 1),
        end=date(2026, 1, 1),
        frequency="1d",
        seed=ARTIFACT_SEED,
    )

    assert first == second
    assert first == same_order_different_hyperparam_key_order
    assert len(first) == 64


def test_model_hash_changes_when_feature_order_changes() -> None:
    inputs = _hash_inputs()
    baseline = compute_model_hash(**inputs)  # type: ignore[arg-type]
    reordered = compute_model_hash(
        features=("rsi_14", "momentum_20"),
        hyperparameters={"learning_rate": 0.05, "num_leaves": 31},
        universe=("MSFT", "AAPL"),
        start=date(2015, 1, 1),
        end=date(2026, 1, 1),
        frequency="1d",
        seed=ARTIFACT_SEED,
    )

    assert baseline != reordered


def test_model_hash_changes_when_hyperparameter_changes() -> None:
    inputs = _hash_inputs()
    baseline = compute_model_hash(**inputs)  # type: ignore[arg-type]
    changed = compute_model_hash(
        features=FEATURE_NAMES,
        hyperparameters={"learning_rate": 0.1, "num_leaves": 31},
        universe=("MSFT", "AAPL"),
        start=date(2015, 1, 1),
        end=date(2026, 1, 1),
        frequency="1d",
        seed=ARTIFACT_SEED,
    )

    assert baseline != changed


def test_model_hash_changes_when_seed_changes() -> None:
    inputs = _hash_inputs()
    baseline = compute_model_hash(**inputs)  # type: ignore[arg-type]
    changed = compute_model_hash(
        features=FEATURE_NAMES,
        hyperparameters={"learning_rate": 0.05, "num_leaves": 31},
        universe=("MSFT", "AAPL"),
        start=date(2015, 1, 1),
        end=date(2026, 1, 1),
        frequency="1d",
        seed=ARTIFACT_SEED + 1,
    )

    assert baseline != changed


def test_filesystem_model_artifact_round_trip(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    adapter = FilesystemModelArtifactAdapter(artifact_root)
    model = _train_dummy_booster()
    sample = np.array([[0.25, -0.5], [1.0, 0.1]], dtype=np.float64)
    expected = model.predict(sample)

    metadata = _make_metadata()
    artifact_dir = adapter.save_booster(model, metadata)
    loaded_model, loaded_metadata = adapter.load_booster(metadata.model_hash)

    assert artifact_dir == artifact_root / "models" / "phase0-dummy" / metadata.model_hash
    assert (artifact_dir / "model.txt").is_file()
    assert (artifact_dir / "metadata.json").is_file()
    assert loaded_metadata == metadata
    np.testing.assert_allclose(loaded_model.predict(sample), expected)

    port_result = adapter.load(metadata.model_hash)
    assert isinstance(port_result, ModelArtifactLoadResult)
    assert port_result.metadata == metadata


def test_save_hard_fails_for_missing_artifact_root(tmp_path: Path) -> None:
    missing_root = tmp_path / "does-not-exist"
    adapter = FilesystemModelArtifactAdapter(missing_root)
    model = _train_dummy_booster()
    metadata = _make_metadata()

    with pytest.raises(FileNotFoundError, match="artifact_root does not exist"):
        adapter.save_booster(model, metadata)


def test_save_hard_fails_for_unwritable_artifact_root(tmp_path: Path) -> None:
    artifact_root = tmp_path / "readonly-artifacts"
    artifact_root.mkdir()
    artifact_root.chmod(0o555)
    adapter = FilesystemModelArtifactAdapter(artifact_root)
    model = _train_dummy_booster()
    metadata = _make_metadata()

    try:
        with pytest.raises(PermissionError, match="artifact_root is not writable"):
            adapter.save_booster(model, metadata)
    finally:
        artifact_root.chmod(0o755)


def test_load_hard_fails_for_unknown_model_hash(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    adapter = FilesystemModelArtifactAdapter(artifact_root)

    with pytest.raises(FileNotFoundError, match="model artifact not found"):
        adapter.load("0" * 64)


def test_save_hard_fails_for_model_hash_mismatch(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    adapter = FilesystemModelArtifactAdapter(artifact_root)
    model = _train_dummy_booster()
    metadata = _make_metadata().model_copy(update={"model_hash": "0" * 64})

    with pytest.raises(ValueError, match="model_hash mismatch"):
        adapter.save_booster(model, metadata)


def test_save_hard_fails_for_booster_feature_mismatch(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    adapter = FilesystemModelArtifactAdapter(artifact_root)
    model = _train_dummy_booster(feature_names=("alpha", "beta"))
    metadata = _make_metadata()

    with pytest.raises(ValueError, match="booster features"):
        adapter.save_booster(model, metadata)


def test_load_hard_fails_for_metadata_feature_mismatch(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    adapter = FilesystemModelArtifactAdapter(artifact_root)
    model = _train_dummy_booster()
    metadata = _make_metadata()
    adapter.save_booster(model, metadata)

    tampered_features = ("rsi_14", "momentum_20")
    tampered_hash = compute_model_hash(
        features=tampered_features,
        hyperparameters=metadata.hyperparameters,
        universe=metadata.universe,
        start=metadata.start,
        end=metadata.end,
        frequency=metadata.frequency,
        seed=metadata.seed,
    )
    tampered = metadata.model_copy(
        update={"features": tampered_features, "model_hash": tampered_hash}
    )
    metadata_path = (
        artifact_root / "models" / metadata.strategy_id / metadata.model_hash / "metadata.json"
    )
    metadata_path.write_text(tampered.model_dump_json(), encoding="utf-8")

    with pytest.raises(ValueError, match="metadata model_hash .* does not match"):
        adapter.load(metadata.model_hash)


def test_load_hard_fails_for_booster_metadata_feature_drift(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    adapter = FilesystemModelArtifactAdapter(artifact_root)
    model = _train_dummy_booster()
    metadata = _make_metadata()
    request = ModelArtifactSaveRequest(
        model_blob=model.model_to_string().encode("utf-8"),
        metadata=metadata,
    )
    adapter.save(request)

    other_model = _train_dummy_booster(feature_names=("alpha", "beta"))
    model_path = (
        artifact_root / "models" / metadata.strategy_id / metadata.model_hash / "model.txt"
    )
    model_path.write_bytes(other_model.model_to_string().encode("utf-8"))

    with pytest.raises(ValueError, match="booster features"):
        adapter.load(metadata.model_hash)
