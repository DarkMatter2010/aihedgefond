"""Definition-of-done tests for Phase 0."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone
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
from aihedgefund.research.adapters.filesystem import (
    FilesystemModelArtifactAdapter,
    compute_model_hash,
)

NOW = datetime(2026, 1, 2, 15, 30, tzinfo=UTC)
ORDER_ID = UUID("00000000-0000-4000-8000-000000000001")


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


@pytest.mark.parametrize("port", [DataVendorPort, BrokerPort, ModelArtifactPort])
def test_ports_are_abstract(port: type[object]) -> None:
    with pytest.raises(TypeError):
        port()


MODEL_FEATURES = ("feature_b", "feature_a")
MODEL_HYPERPARAMETERS: dict[str, object] = {
    "learning_rate": 0.1,
    "num_leaves": 4,
    "seed": 17,
    "verbosity": -1,
}
MODEL_SAMPLE = np.array([[0.15, 0.85], [0.75, 0.25]], dtype=np.float64)


@pytest.fixture(scope="module")
def dummy_lightgbm_model() -> lgb.Booster:
    """Train a small deterministic native model for persistence tests."""
    features = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.2, 0.8],
            [0.8, 0.2],
            [0.1, 0.9],
            [0.9, 0.1],
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 1, 1, 0, 1, 1, 1, 1], dtype=np.float64)
    return lgb.train(
        {
            "objective": "binary",
            "learning_rate": 0.1,
            "num_leaves": 4,
            "min_data_in_leaf": 1,
            "seed": 17,
            "num_threads": 1,
            "verbosity": -1,
        },
        lgb.Dataset(features, label=labels),
        num_boost_round=5,
    )


def make_model_metadata(settings: Settings) -> ModelArtifactMetadata:
    """Create metadata whose hash is derived from the configured training inputs."""
    model_hash = compute_model_hash(
        features=MODEL_FEATURES,
        hyperparameters=MODEL_HYPERPARAMETERS,
        universe=settings.universe,
        settings=settings,
    )
    return ModelArtifactMetadata(
        model_hash=model_hash,
        strategy_id="phase0-artifact-test",
        created_at=NOW,
        universe=settings.universe,
        features=MODEL_FEATURES,
        hyperparameters=MODEL_HYPERPARAMETERS,
        model_format="lightgbm_native",
        model_file="model.txt",
        phase=0,
    )


def test_model_artifact_metadata_contract_and_config_key() -> None:
    settings = load_settings()
    metadata = make_model_metadata(settings)

    assert settings.artifact_root == Path("artifacts")
    assert tuple(ModelArtifactMetadata.model_fields) == (
        "model_hash",
        "strategy_id",
        "created_at",
        "universe",
        "features",
        "hyperparameters",
        "model_format",
        "model_file",
        "phase",
    )
    assert metadata.created_at.tzinfo is UTC
    with pytest.raises(ValidationError, match="created_at must use UTC"):
        ModelArtifactMetadata.model_validate(
            {
                **metadata.model_dump(),
                "created_at": datetime(
                    2026,
                    1,
                    2,
                    16,
                    30,
                    tzinfo=timezone(timedelta(hours=1)),
                ),
            }
        )


def test_model_hash_is_deterministic_across_order_and_calls() -> None:
    settings = load_settings()
    hashes = {
        compute_model_hash(
            features=features,
            hyperparameters=hyperparameters,
            universe=universe,
            settings=settings,
        )
        for features, hyperparameters, universe in (
            (MODEL_FEATURES, MODEL_HYPERPARAMETERS, settings.universe),
            (
                tuple(reversed(MODEL_FEATURES)),
                dict(reversed(tuple(MODEL_HYPERPARAMETERS.items()))),
                tuple(reversed(settings.universe)),
            ),
            (MODEL_FEATURES, MODEL_HYPERPARAMETERS, settings.universe),
        )
    }

    assert len(hashes) == 1


def test_model_hash_changes_with_hyperparameter() -> None:
    settings = load_settings()
    original_hash = compute_model_hash(
        features=MODEL_FEATURES,
        hyperparameters=MODEL_HYPERPARAMETERS,
        universe=settings.universe,
        settings=settings,
    )
    changed_hash = compute_model_hash(
        features=MODEL_FEATURES,
        hyperparameters={**MODEL_HYPERPARAMETERS, "num_leaves": 8},
        universe=settings.universe,
        settings=settings,
    )

    assert changed_hash != original_hash


def test_lightgbm_artifact_round_trip_preserves_predictions(
    tmp_path: Path,
    dummy_lightgbm_model: lgb.Booster,
) -> None:
    settings = load_settings()
    metadata = make_model_metadata(settings)
    adapter = FilesystemModelArtifactAdapter(tmp_path)

    artifact_directory = adapter.save_model(dummy_lightgbm_model, metadata)
    loaded_model, loaded_metadata = adapter.load_model(metadata.model_hash)

    assert artifact_directory == (tmp_path / "models" / metadata.strategy_id / metadata.model_hash)
    assert (artifact_directory / "model.txt").is_file()
    assert (artifact_directory / "metadata.json").is_file()
    assert loaded_metadata == metadata
    np.testing.assert_allclose(
        loaded_model.predict(MODEL_SAMPLE),
        dummy_lightgbm_model.predict(MODEL_SAMPLE),
        rtol=0,
        atol=0,
    )


def test_model_save_hard_fails_for_missing_or_unwritable_root(
    tmp_path: Path,
    dummy_lightgbm_model: lgb.Booster,
) -> None:
    metadata = make_model_metadata(load_settings())
    missing_root = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="artifact_root does not exist"):
        FilesystemModelArtifactAdapter(missing_root).save_model(dummy_lightgbm_model, metadata)

    unwritable_root = tmp_path / "unwritable"
    unwritable_root.mkdir()
    unwritable_root.chmod(0o500)
    try:
        with pytest.raises(PermissionError, match="artifact_root is not writable"):
            FilesystemModelArtifactAdapter(unwritable_root).save_model(
                dummy_lightgbm_model, metadata
            )
    finally:
        unwritable_root.chmod(0o700)


def test_model_load_hard_fails_for_unknown_hash(tmp_path: Path) -> None:
    adapter = FilesystemModelArtifactAdapter(tmp_path)

    with pytest.raises(FileNotFoundError, match="model hash not found"):
        adapter.load_model("missing-model-hash")
