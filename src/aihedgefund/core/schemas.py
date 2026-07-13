"""Vendor-neutral DTOs shared across module boundaries."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import StrEnum
from typing import Annotated
from uuid import UUID, uuid4

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

Symbol = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, pattern=r"^[A-Z0-9._-]+$"),
]
PositiveDecimal = Annotated[Decimal, Field(gt=0)]
NonNegativeDecimal = Annotated[Decimal, Field(ge=0)]
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class BoundaryDTO(BaseModel):
    """Immutable, strict base for every object crossing a module boundary."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


class Message(BoundaryDTO):
    """Base for timestamped bus messages."""

    timestamp: AwareDatetime

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_be_utc(cls, value: datetime) -> datetime:
        """Reject non-UTC timestamps and normalize valid values to UTC."""
        if value.utcoffset() != timedelta(0):
            msg = "timestamp must use UTC"
            raise ValueError(msg)
        return value.astimezone(timezone.utc)


class Command(Message):
    """An intent to change state."""


class Event(Message):
    """An immutable fact that has already occurred."""


class OrderSide(StrEnum):
    """Supported order directions."""

    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    """Supported foundational order types."""

    MARKET = "market"
    LIMIT = "limit"


class Signal(Event):
    """Model output for one instrument."""

    signal_id: UUID = Field(default_factory=uuid4)
    symbol: Symbol
    value: Annotated[float, Field(allow_inf_nan=False)]
    model_version: NonEmptyText


class Order(Command):
    """Broker-independent order command."""

    order_id: UUID = Field(default_factory=uuid4)
    symbol: Symbol
    side: OrderSide
    quantity: PositiveDecimal
    order_type: OrderType
    limit_price: PositiveDecimal | None = None

    @model_validator(mode="after")
    def validate_limit_price(self) -> Order:
        """Require limit prices exactly when the order type needs one."""
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            msg = "limit_price is required for limit orders"
            raise ValueError(msg)
        if self.order_type is OrderType.MARKET and self.limit_price is not None:
            msg = "limit_price is not allowed for market orders"
            raise ValueError(msg)
        return self


class Fill(Event):
    """Execution fact emitted after a broker fills an order."""

    fill_id: UUID = Field(default_factory=uuid4)
    order_id: UUID
    symbol: Symbol
    fill_price: PositiveDecimal
    filled_qty: PositiveDecimal


class RiskCheck(Event):
    """Result of evaluating an order against configured limits."""

    risk_check_id: UUID = Field(default_factory=uuid4)
    order_id: UUID
    passed: bool
    violated_limits: tuple[NonEmptyText, ...] = ()

    @model_validator(mode="after")
    def result_must_match_violations(self) -> RiskCheck:
        """Prevent contradictory risk outcomes."""
        if self.passed and self.violated_limits:
            msg = "a passed risk check cannot contain violated limits"
            raise ValueError(msg)
        if not self.passed and not self.violated_limits:
            msg = "a failed risk check must contain at least one violated limit"
            raise ValueError(msg)
        return self


class FeatureValue(BoundaryDTO):
    """One named numerical feature."""

    name: NonEmptyText
    value: Annotated[float, Field(allow_inf_nan=False)]


class FeatureVector(Event):
    """Versioned feature values for one instrument and timestamp."""

    feature_vector_id: UUID = Field(default_factory=uuid4)
    symbol: Symbol
    features: Annotated[tuple[FeatureValue, ...], Field(min_length=1)]
    feature_set_version: NonEmptyText

    @field_validator("features")
    @classmethod
    def feature_names_must_be_unique(
        cls, value: tuple[FeatureValue, ...]
    ) -> tuple[FeatureValue, ...]:
        """Reject ambiguous duplicate feature names."""
        names = [feature.name for feature in value]
        if len(names) != len(set(names)):
            msg = "feature names must be unique"
            raise ValueError(msg)
        return value


class OHLCVRequest(BoundaryDTO):
    """Typed request for historical bars."""

    symbol: Symbol
    start: AwareDatetime
    end: AwareDatetime

    @field_validator("start", "end")
    @classmethod
    def timestamps_must_be_utc(cls, value: datetime) -> datetime:
        """Require UTC request boundaries."""
        if value.utcoffset() != timedelta(0):
            msg = "request timestamps must use UTC"
            raise ValueError(msg)
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def end_must_follow_start(self) -> OHLCVRequest:
        """Reject empty and reversed time windows."""
        if self.end <= self.start:
            msg = "end must be later than start"
            raise ValueError(msg)
        return self


class OHLCVBar(BoundaryDTO):
    """Vendor-neutral market-data response item."""

    symbol: Symbol
    timestamp: AwareDatetime
    open: PositiveDecimal
    high: PositiveDecimal
    low: PositiveDecimal
    close: PositiveDecimal
    volume: NonNegativeDecimal

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_be_utc(cls, value: datetime) -> datetime:
        """Require UTC market-data timestamps."""
        if value.utcoffset() != timedelta(0):
            msg = "timestamp must use UTC"
            raise ValueError(msg)
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def prices_must_form_valid_bar(self) -> OHLCVBar:
        """Reject bars whose high/low do not contain every traded price."""
        prices = (self.open, self.close, self.low, self.high)
        if self.high != max(prices) or self.low != min(prices):
            msg = "high and low must bound open and close"
            raise ValueError(msg)
        return self


class Position(BoundaryDTO):
    """Broker-neutral current position."""

    symbol: Symbol
    quantity: Decimal
    average_price: NonNegativeDecimal
