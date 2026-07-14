"""Vendor-neutral DTOs shared across module boundaries."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID, uuid4

import pandas as pd
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
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


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
        return value.astimezone(UTC)


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
        return value.astimezone(UTC)

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
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def prices_must_form_valid_bar(self) -> OHLCVBar:
        """Reject bars whose high/low do not contain every traded price."""
        prices = (self.open, self.close, self.low, self.high)
        if self.high != max(prices) or self.low != min(prices):
            msg = "high and low must bound open and close"
            raise ValueError(msg)
        return self


class BarFrame(BoundaryDTO):
    """Canonical per-symbol OHLCV frames and unadjusted corporate actions."""

    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    bars: dict[Symbol, pd.DataFrame]
    dividends: dict[Symbol, pd.Series]
    splits: dict[Symbol, pd.Series]

    @model_validator(mode="after")
    def validate_canonical_frames(self) -> BarFrame:
        """Require canonical columns, UTC indexes, and aligned action series."""
        expected_columns = ("open", "high", "low", "close", "adj_close", "volume")
        symbols = set(self.bars)
        if not symbols:
            msg = "at least one symbol frame is required"
            raise ValueError(msg)
        if set(self.dividends) != symbols or set(self.splits) != symbols:
            msg = "bar and corporate-action symbols must match"
            raise ValueError(msg)
        for symbol, frame in self.bars.items():
            if not isinstance(frame.index, pd.DatetimeIndex) or str(frame.index.tz) != "UTC":
                msg = f"{symbol} bar index must be a UTC DatetimeIndex"
                raise ValueError(msg)
            if tuple(frame.columns) != expected_columns:
                msg = f"{symbol} columns must be {expected_columns}"
                raise ValueError(msg)
            for action_name, action in (
                ("dividends", self.dividends[symbol]),
                ("splits", self.splits[symbol]),
            ):
                if not isinstance(action.index, pd.DatetimeIndex) or str(action.index.tz) != "UTC":
                    msg = f"{symbol} {action_name} index must be a UTC DatetimeIndex"
                    raise ValueError(msg)
                if not action.index.equals(frame.index):
                    msg = f"{symbol} {action_name} index must align with bars"
                    raise ValueError(msg)
        return self


class QualityReport(BoundaryDTO):
    """Successful quality-gate measurements for one symbol."""

    symbol: Symbol
    rows: Annotated[int, Field(gt=0)]
    nan_ratios: dict[NonEmptyText, Annotated[float, Field(ge=0, le=1)]]
    max_abs_log_return: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    max_return_zscore: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    last_timestamp: AwareDatetime
    passed: Literal[True] = True

    @field_validator("last_timestamp")
    @classmethod
    def last_timestamp_must_be_utc(cls, value: datetime) -> datetime:
        """Require a UTC quality-report boundary."""
        if value.utcoffset() != timedelta(0):
            msg = "last_timestamp must use UTC"
            raise ValueError(msg)
        return value.astimezone(UTC)


class LabeledSample(BoundaryDTO):
    """One purge-friendly label with its overlap-aware training weight."""

    label: Literal[-1, 0, 1]
    t0: AwareDatetime
    t1: AwareDatetime
    ret: FiniteFloat
    weight: Annotated[float, Field(ge=0, allow_inf_nan=False)]

    @field_validator("t0", "t1")
    @classmethod
    def label_timestamps_must_be_utc(cls, value: datetime) -> datetime:
        """Require UTC event boundaries."""
        if value.utcoffset() != timedelta(0):
            msg = "label timestamps must use UTC"
            raise ValueError(msg)
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def touch_must_not_precede_event(self) -> LabeledSample:
        """Reject impossible event intervals."""
        if self.t1 < self.t0:
            msg = "t1 must not precede t0"
            raise ValueError(msg)
        return self


class DataIngested(Event):
    """Fact emitted after canonical vendor data is validated."""

    symbols: Annotated[tuple[Symbol, ...], Field(min_length=1)]
    rows: dict[Symbol, Annotated[int, Field(gt=0)]]


class QualityGateFailed(Event):
    """Fact emitted immediately before bad market data is rejected."""

    symbol: Symbol
    reason: NonEmptyText


class QualityReportProduced(Event):
    """Fact emitted after a market-data frame passes every quality rule."""

    report: QualityReport


class FeaturesComputed(Event):
    """Fact emitted after a causal feature matrix is produced."""

    symbols: Annotated[tuple[Symbol, ...], Field(min_length=1)]
    rows: Annotated[int, Field(gt=0)]


class LabelsComputed(Event):
    """Fact emitted after purge-friendly labels are produced."""

    samples: Annotated[int, Field(gt=0)]


class Position(BoundaryDTO):
    """Broker-neutral current position."""

    symbol: Symbol
    quantity: Decimal
    average_price: NonNegativeDecimal
