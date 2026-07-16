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
    ModelArtifactMetadata,
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
        "features": ("momentum_20", "rsi_14"),
        "hyperparameters": {"learning_rate": 0.05, "num_leaves": 31},
        "universe": ("MSFT", "AAPL"),
        "start": date(2015, 1, 1),
        "end": date(2026, 1, 1),
        "frequency": "1d",
    }


def _train_dummy_booster() -> lgb.Booster:
    rng = np.random.default_rng(42)
    features = rng.normal(size=(64, 2))
    labels = (features[:, 0] + 0.5 * features[:, 1] > 0).astype(np.float64)
    dataset = lgb.Dataset(features, label=labels, free_raw_data=False)
    return lgb.train(
        {
            "objective": "binary",
            "verbosity": -1,
            "seed": 42,
            "deterministic": True,
            "num_threads": 1,
        },
        dataset,
        num_boost_round=5,
    )


def _make_metadata(model_hash: str) -> ModelArtifactMetadata:
    return ModelArtifactMetadata(
        model_hash=model_hash,
        strategy_id="phase0-dummy",
        created_at=ARTIFACT_CREATED_AT,
        universe=("AAPL", "MSFT"),
        features=("momentum_20", "rsi_14"),
        hyperparameters={"learning_rate": 0.05, "num_leaves": 31},
        model_format="lightgbm_native",
        model_file="model.txt",
        phase=0,
    )


def test_settings_expose_artifact_root_default() -> None:
    settings = load_settings()

    assert settings.artifact_root == Path("artifacts/")


def test_model_artifact_metadata_is_frozen_and_utc() -> None:
    metadata = _make_metadata("abc123")

    assert metadata.model_format == "lightgbm_native"
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
    shuffled = compute_model_hash(
        features=("rsi_14", "momentum_20"),
        hyperparameters={"num_leaves": 31, "learning_rate": 0.05},
        universe=("AAPL", "MSFT"),
        start=date(2015, 1, 1),
        end=date(2026, 1, 1),
        frequency="1d",
    )

    assert first == second
    assert first == shuffled
    assert len(first) == 64


def test_model_hash_changes_when_hyperparameter_changes() -> None:
    inputs = _hash_inputs()
    baseline = compute_model_hash(**inputs)  # type: ignore[arg-type]
    changed = compute_model_hash(
        features=("momentum_20", "rsi_14"),
        hyperparameters={"learning_rate": 0.1, "num_leaves": 31},
        universe=("MSFT", "AAPL"),
        start=date(2015, 1, 1),
        end=date(2026, 1, 1),
        frequency="1d",
    )

    assert baseline != changed


def test_filesystem_model_artifact_round_trip(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    adapter = FilesystemModelArtifactAdapter(artifact_root)
    model = _train_dummy_booster()
    sample = np.array([[0.25, -0.5], [1.0, 0.1]], dtype=np.float64)
    expected = model.predict(sample)

    model_hash = compute_model_hash(**_hash_inputs())  # type: ignore[arg-type]
    metadata = _make_metadata(model_hash)
    artifact_dir = adapter.save(model, metadata)
    loaded_model, loaded_metadata = adapter.load(model_hash)

    assert artifact_dir == artifact_root / "models" / "phase0-dummy" / model_hash
    assert (artifact_dir / "model.txt").is_file()
    assert (artifact_dir / "metadata.json").is_file()
    assert loaded_metadata == metadata
    np.testing.assert_allclose(loaded_model.predict(sample), expected)


def test_save_hard_fails_for_missing_artifact_root(tmp_path: Path) -> None:
    missing_root = tmp_path / "does-not-exist"
    adapter = FilesystemModelArtifactAdapter(missing_root)
    model = _train_dummy_booster()
    metadata = _make_metadata("missing-root-hash")

    with pytest.raises(FileNotFoundError, match="artifact_root does not exist"):
        adapter.save(model, metadata)


def test_save_hard_fails_for_unwritable_artifact_root(tmp_path: Path) -> None:
    artifact_root = tmp_path / "readonly-artifacts"
    artifact_root.mkdir()
    artifact_root.chmod(0o555)
    adapter = FilesystemModelArtifactAdapter(artifact_root)
    model = _train_dummy_booster()
    metadata = _make_metadata("readonly-hash")

    try:
        with pytest.raises(PermissionError, match="artifact_root is not writable"):
            adapter.save(model, metadata)
    finally:
        artifact_root.chmod(0o755)


def test_load_hard_fails_for_unknown_model_hash(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    adapter = FilesystemModelArtifactAdapter(artifact_root)

    with pytest.raises(FileNotFoundError, match="model artifact not found"):
        adapter.load("does-not-exist-hash")
