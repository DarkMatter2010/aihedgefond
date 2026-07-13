"""Definition-of-done tests for Phase 0."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import BaseModel, ValidationError

from aihedgefund.core.bus import InProcessMessageBus
from aihedgefund.core.config import Settings, load_settings
from aihedgefund.core.ports import BrokerPort, DataVendorPort
from aihedgefund.core.schemas import (
    Event,
    FeatureValue,
    FeatureVector,
    Fill,
    Order,
    OrderSide,
    OrderType,
    RiskCheck,
    Signal,
)

NOW = datetime(2026, 1, 2, 15, 30, tzinfo=timezone.utc)
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


@pytest.mark.parametrize("port", [DataVendorPort, BrokerPort])
def test_ports_are_abstract(port: type[object]) -> None:
    with pytest.raises(TypeError):
        port()
