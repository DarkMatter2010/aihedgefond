"""Vendor-neutral DTOs shared across module boundaries."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID, uuid4

import pandas as pd
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
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


class ModelTrainingConfig(BoundaryDTO):
    """Deterministic training inputs required for model reproduction."""

    seed: Annotated[int, Field(ge=0)]
    hyperparameters: dict[NonEmptyText, JsonValue]


class ModelArtifactMetadata(BoundaryDTO):
    """Reproducibility metadata stored beside one native model artifact."""

    model_hash: NonEmptyText
    strategy_id: NonEmptyText
    created_at: AwareDatetime
    universe: tuple[Symbol, ...]
    features: tuple[NonEmptyText, ...]
    training_config: ModelTrainingConfig
    model_format: Literal["lightgbm_native"]
    model_file: NonEmptyText
    phase: Annotated[int, Field(ge=0)]

    @field_validator("created_at")
    @classmethod
    def created_at_must_be_utc(cls, value: datetime) -> datetime:
        """Reject non-UTC creation timestamps and normalize UTC values."""
        if value.utcoffset() != timedelta(0):
            msg = "created_at must use UTC"
            raise ValueError(msg)
        return value.astimezone(UTC)

    @field_validator("model_hash", "strategy_id")
    @classmethod
    def artifact_directory_names_must_be_safe(cls, value: str) -> str:
        """Require metadata-derived directories to be single safe path segments."""
        if value in {".", ".."} or "/" in value or "\\" in value:
            msg = "artifact directory names must be single path segments"
            raise ValueError(msg)
        return value

    @field_validator("model_file")
    @classmethod
    def model_file_must_be_relative_filename(cls, value: str) -> str:
        """Reject absolute, nested, and traversal model paths."""
        if value in {".", ".."} or "/" in value or "\\" in value:
            msg = "model_file must be a relative filename"
            raise ValueError(msg)
        return value


class SaveModelArtifactRequest(BoundaryDTO):
    """Vendor-neutral payload and metadata to persist."""

    model_data: Annotated[bytes, Field(min_length=1)]
    metadata: ModelArtifactMetadata


class SaveModelArtifactResult(BoundaryDTO):
    """Location created for one persisted model artifact."""

    artifact_directory: Path


class LoadModelArtifactRequest(BoundaryDTO):
    """Identity of one model artifact to restore."""

    model_hash: NonEmptyText

    @field_validator("model_hash")
    @classmethod
    def model_hash_must_be_safe(cls, value: str) -> str:
        """Require the hash lookup to remain within one path segment."""
        if value in {".", ".."} or "/" in value or "\\" in value:
            msg = "model_hash must be a single path segment"
            raise ValueError(msg)
        return value


class LoadModelArtifactResult(BoundaryDTO):
    """Vendor-neutral persisted model payload and validated metadata."""

    model_data: Annotated[bytes, Field(min_length=1)]
    metadata: ModelArtifactMetadata


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


class MarketDataRequest(BoundaryDTO):
    """Typed multi-symbol request at the Phase 1 ingest boundary."""

    symbols: Annotated[tuple[Symbol, ...], Field(min_length=1)]
    start: AwareDatetime
    end: AwareDatetime
    frequency: Literal["1d"]

    @field_validator("symbols")
    @classmethod
    def symbols_must_be_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject ambiguous duplicate symbols."""
        if len(value) != len(set(value)):
            msg = "market-data symbols must be unique"
            raise ValueError(msg)
        return value

    @field_validator("start", "end")
    @classmethod
    def timestamps_must_be_utc(cls, value: datetime) -> datetime:
        """Require UTC request boundaries."""
        if value.utcoffset() != timedelta(0):
            msg = "market-data request timestamps must use UTC"
            raise ValueError(msg)
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def end_must_follow_start(self) -> MarketDataRequest:
        """Reject empty and reversed ingest windows."""
        if self.end <= self.start:
            msg = "market-data request end must be later than start"
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


class CorporateActionInput(BoundaryDTO):
    """Aligned raw prices and action facts entering the pure CA transform."""

    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    raw_close: pd.Series
    splits: pd.Series
    dividends: pd.Series

    @model_validator(mode="after")
    def validate_series(self) -> CorporateActionInput:
        """Require aligned, ordered, finite action inputs."""
        if self.raw_close.empty:
            msg = "raw_close must not be empty"
            raise ValueError(msg)
        if not self.raw_close.index.equals(self.splits.index) or not self.raw_close.index.equals(
            self.dividends.index
        ):
            msg = "close, splits, and dividends must have identical indexes"
            raise ValueError(msg)
        if not isinstance(self.raw_close.index, pd.DatetimeIndex):
            msg = "corporate-action index must be a DatetimeIndex"
            raise ValueError(msg)
        if self.raw_close.index.has_duplicates or not self.raw_close.index.is_monotonic_increasing:
            msg = "corporate-action index must be unique and increasing"
            raise ValueError(msg)
        try:
            raw_close = self.raw_close.astype(float)
            splits = self.splits.astype(float)
            dividends = self.dividends.astype(float)
        except (TypeError, ValueError) as exc:
            msg = "corporate-action series must be numeric"
            raise ValueError(msg) from exc
        if raw_close.isna().any() or splits.isna().any() or dividends.isna().any():
            msg = "corporate-action inputs must not contain NaNs"
            raise ValueError(msg)
        if (raw_close <= 0).any() or (splits < 0).any() or (dividends < 0).any():
            msg = "prices must be positive and actions must be non-negative"
            raise ValueError(msg)
        return self


class CorporateActionAdjustment(BoundaryDTO):
    """Forward-adjusted price outputs whose row uses only known actions."""

    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    raw_close: pd.Series
    split_adjusted: pd.Series
    as_of_adjusted: pd.Series

    @model_validator(mode="after")
    def series_must_align(self) -> CorporateActionAdjustment:
        """Require all output series to share the raw-close index."""
        if not self.raw_close.index.equals(
            self.split_adjusted.index
        ) or not self.raw_close.index.equals(self.as_of_adjusted.index):
            msg = "corporate-action outputs must have identical indexes"
            raise ValueError(msg)
        return self


class PointInTimeFrame(BoundaryDTO):
    """Typed wrapper for mutable tables crossing the PIT module boundary."""

    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    frame: pd.DataFrame


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


class IngestedMarketData(BoundaryDTO):
    """Typed payload emitted after an ingest request passes quality checks."""

    request: MarketDataRequest
    data: BarFrame


class QualityFailure(BoundaryDTO):
    """Typed payload describing a rejected symbol frame."""

    symbol: Symbol
    reason: NonEmptyText


class FeatureMatrixPayload(BoundaryDTO):
    """Typed metadata for the intentionally DataFrame-returning feature contract."""

    symbols: Annotated[tuple[Symbol, ...], Field(min_length=1)]
    rows: Annotated[int, Field(gt=0)]
    columns: Annotated[tuple[NonEmptyText, ...], Field(min_length=1)]
    dtypes: Annotated[tuple[Literal["float64"], ...], Field(min_length=1)]

    @model_validator(mode="after")
    def columns_and_dtypes_must_align(self) -> FeatureMatrixPayload:
        """Require one explicit dtype per feature column."""
        if len(self.columns) != len(self.dtypes):
            msg = "feature columns and dtypes must have equal lengths"
            raise ValueError(msg)
        return self


class LabelBatch(BoundaryDTO):
    """Typed payload containing the labels produced by one run."""

    samples: Annotated[tuple[LabeledSample, ...], Field(min_length=1)]


class DataIngested(Event):
    """Fact emitted after canonical vendor data is validated."""

    payload: IngestedMarketData

    @property
    def symbols(self) -> tuple[str, ...]:
        """Expose requested symbols for Phase 1 compatibility."""
        return self.payload.request.symbols

    @property
    def rows(self) -> dict[str, int]:
        """Expose validated row counts for Phase 1 compatibility."""
        return {symbol: len(frame) for symbol, frame in self.payload.data.bars.items()}


class QualityGateFailed(Event):
    """Fact emitted immediately before bad market data is rejected."""

    payload: QualityFailure

    @property
    def symbol(self) -> str:
        """Expose the rejected symbol for compatibility."""
        return self.payload.symbol

    @property
    def reason(self) -> str:
        """Expose the rejection reason for compatibility."""
        return self.payload.reason


class QualityReportProduced(Event):
    """Fact emitted after a market-data frame passes every quality rule."""

    report: QualityReport


class FeaturesComputed(Event):
    """Fact emitted after a causal feature matrix is produced."""

    payload: FeatureMatrixPayload

    @property
    def symbols(self) -> tuple[str, ...]:
        """Expose computed symbols for compatibility."""
        return self.payload.symbols

    @property
    def rows(self) -> int:
        """Expose the feature row count for compatibility."""
        return self.payload.rows


class LabelsComputed(Event):
    """Fact emitted after purge-friendly labels are produced."""

    payload: LabelBatch

    @property
    def samples(self) -> int:
        """Expose the label count for compatibility."""
        return len(self.payload.samples)


class Position(BoundaryDTO):
    """Broker-neutral current position."""

    symbol: Symbol
    quantity: Decimal
    average_price: NonNegativeDecimal
