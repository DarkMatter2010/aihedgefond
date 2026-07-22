"""Vendor-neutral DTOs shared across module boundaries."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
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


class Form4Request(BoundaryDTO):
    """Multi-symbol Form 4 ingest window (UTC).

    PIT rule: features must key off ``filed_at`` / acceptance time, never
    ``transaction_date`` alone (filings may lag the trade by up to ~2 days).
    """

    symbols: Annotated[tuple[Symbol, ...], Field(min_length=1)]
    start: AwareDatetime
    end: AwareDatetime

    @field_validator("symbols")
    @classmethod
    def symbols_must_be_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject ambiguous duplicate symbols."""
        if len(value) != len(set(value)):
            msg = "form4 symbols must be unique"
            raise ValueError(msg)
        return value

    @field_validator("start", "end")
    @classmethod
    def timestamps_must_be_utc(cls, value: datetime) -> datetime:
        """Require UTC request boundaries."""
        if value.utcoffset() != timedelta(0):
            msg = "form4 request timestamps must use UTC"
            raise ValueError(msg)
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def end_must_follow_start(self) -> Form4Request:
        """Reject empty and reversed ingest windows."""
        if self.end <= self.start:
            msg = "form4 request end must be later than start"
            raise ValueError(msg)
        return self


class Form4Record(BoundaryDTO):
    """One insider transaction row extracted from a Form 4 filing.

    ``filed_at`` is the SEC acceptance timestamp (PIT-safe). ``transaction_date``
    is the trade date inside the form and is informational only.
    """

    symbol: Symbol
    cik: NonEmptyText
    accession: NonEmptyText
    filed_at: AwareDatetime
    transaction_date: date | None
    transaction_code: NonEmptyText
    shares: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    price: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None
    acquired_disposed: Literal["A", "D"]
    reporting_owner: NonEmptyText | None = None

    @field_validator("filed_at")
    @classmethod
    def filed_at_must_be_utc(cls, value: datetime) -> datetime:
        """Require UTC filing acceptance timestamps."""
        if value.utcoffset() != timedelta(0):
            msg = "filed_at must use UTC"
            raise ValueError(msg)
        return value.astimezone(UTC)


class Form4Frame(BoundaryDTO):
    """Validated Form 4 rows for a request; empty records is normal (no activity)."""

    records: tuple[Form4Record, ...]
    symbols_queried: Annotated[tuple[Symbol, ...], Field(min_length=1)]
    symbols_without_filings: tuple[Symbol, ...] = ()

    @model_validator(mode="after")
    def without_filings_must_be_subset(self) -> Form4Frame:
        """``symbols_without_filings`` must be a subset of queried symbols."""
        queried = set(self.symbols_queried)
        missing = set(self.symbols_without_filings)
        if not missing.issubset(queried):
            msg = "symbols_without_filings must be a subset of symbols_queried"
            raise ValueError(msg)
        return self


class IngestedForm4Data(BoundaryDTO):
    """Typed payload for a successful Form 4 ingest."""

    request: Form4Request
    data: Form4Frame


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
    """Aligned vendor closes (split-continuous) and action facts for the CA transform."""

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
        if (
            not self.raw_close.index.equals(self.splits.index)
            or not self.raw_close.index.equals(self.dividends.index)
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
        if (
            not self.raw_close.index.equals(self.split_adjusted.index)
            or not self.raw_close.index.equals(self.as_of_adjusted.index)
        ):
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


class ExtremeReturnFlag(BoundaryDTO):
    """One statistically or magnitude-extreme but physically plausible log-return."""

    symbol: Symbol
    bar_timestamp: AwareDatetime
    log_return: FiniteFloat
    z_score: Annotated[float, Field(ge=0, allow_inf_nan=False)]

    @field_validator("bar_timestamp")
    @classmethod
    def bar_timestamp_must_be_utc(cls, value: datetime) -> datetime:
        """Require a UTC bar timestamp for extreme-return flags."""
        if value.utcoffset() != timedelta(0):
            msg = "bar_timestamp must use UTC"
            raise ValueError(msg)
        return value.astimezone(UTC)


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
        return {
            symbol: len(frame)
            for symbol, frame in self.payload.data.bars.items()
        }


class Form4Ingested(Event):
    """Fact emitted after Form 4 rows are quality-checked."""

    payload: IngestedForm4Data

    @property
    def symbols(self) -> tuple[str, ...]:
        """Expose queried symbols."""
        return self.payload.request.symbols

    @property
    def n_records(self) -> int:
        """Expose validated transaction row count."""
        return len(self.payload.data.records)


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


class ExtremeReturnFlagged(Event):
    """Soft flag for a real crash-scale return that stays in the sample."""

    payload: ExtremeReturnFlag

    @property
    def symbol(self) -> str:
        """Expose the flagged symbol for subscribers."""
        return self.payload.symbol


class QualityReportProduced(Event):
    """Fact emitted after a market-data frame passes every hard-fail quality rule."""

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


class ModelArtifactMetadata(BoundaryDTO):
    """Versioned metadata persisted beside a trained model artifact."""

    model_hash: Annotated[
        str,
        StringConstraints(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"),
    ]
    strategy_id: NonEmptyText
    created_at: AwareDatetime
    universe: Annotated[tuple[NonEmptyText, ...], Field(min_length=1)]
    features: Annotated[tuple[NonEmptyText, ...], Field(min_length=1)]
    hyperparameters: dict[NonEmptyText, object]
    seed: Annotated[int, Field(ge=0)]
    start: date
    end: date
    frequency: Literal["1d"]
    model_format: Literal["lightgbm_native"]
    model_file: NonEmptyText
    phase: Annotated[int, Field(ge=0)]

    @field_validator("created_at")
    @classmethod
    def created_at_must_be_utc(cls, value: datetime) -> datetime:
        """Require UTC artifact timestamps."""
        if value.utcoffset() != timedelta(0):
            msg = "created_at must use UTC"
            raise ValueError(msg)
        return value.astimezone(UTC)

    @field_validator("universe", "features", mode="before")
    @classmethod
    def freeze_sequence(cls, value: object) -> object:
        """Convert list payloads to immutable tuples at the boundary."""
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def end_must_follow_start(self) -> ModelArtifactMetadata:
        """Reject empty or reversed training windows in artifact metadata."""
        if self.end <= self.start:
            msg = "end must be later than start"
            raise ValueError(msg)
        return self


class ModelArtifactSaveRequest(BoundaryDTO):
    """Vendor-neutral save payload for model artifact persistence."""

    model_blob: bytes
    metadata: ModelArtifactMetadata


class ModelArtifactLoadResult(BoundaryDTO):
    """Vendor-neutral load payload returned by model artifact adapters."""

    model_blob: bytes
    metadata: ModelArtifactMetadata


class ForwardReturnLabelMeta(BoundaryDTO):
    """Typed metadata for dense leak-free forward-return labels."""

    horizon: Annotated[int, Field(ge=1)]
    rows: Annotated[int, Field(ge=0)]
    symbols: Annotated[tuple[Symbol, ...], Field(min_length=1)]
    close_adj_source: Literal["as_of_adjusted"]


class BaselineDataset(BoundaryDTO):
    """Aligned Phase-1 features and dense forward-return labels."""

    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    features: pd.DataFrame
    label: pd.Series
    horizon: Annotated[int, Field(ge=1)]
    feature_columns: Annotated[tuple[NonEmptyText, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def features_and_label_must_align(self) -> BaselineDataset:
        """Require identical MultiIndex rows and finite float64 feature columns."""
        if not isinstance(self.features.index, pd.MultiIndex):
            msg = "baseline features must use a MultiIndex"
            raise ValueError(msg)
        if list(self.features.index.names) != ["timestamp", "symbol"]:
            msg = "baseline features index names must be ('timestamp', 'symbol')"
            raise ValueError(msg)
        if tuple(self.features.columns) != self.feature_columns:
            msg = "baseline feature columns must match feature_columns metadata"
            raise ValueError(msg)
        if not self.label.index.equals(self.features.index):
            msg = "baseline label index must equal features index"
            raise ValueError(msg)
        if self.features.isna().any().any() or self.label.isna().any():
            msg = "baseline dataset must not contain NaNs"
            raise ValueError(msg)
        return self


class SplitDefinition(BoundaryDTO):
    """Fixed calendar train/test split with embargo semantics."""

    train_end: date
    test_start: date
    embargo_days: Annotated[int, Field(ge=1)]
    horizon: Annotated[int, Field(ge=1)]
    train_rows: Annotated[int, Field(ge=0)]
    test_rows: Annotated[int, Field(ge=0)]
    train_max_timestamp: AwareDatetime | None = None
    test_min_timestamp: AwareDatetime | None = None

    @field_validator("train_max_timestamp", "test_min_timestamp")
    @classmethod
    def split_timestamps_must_be_utc(
        cls, value: datetime | None
    ) -> datetime | None:
        """Require UTC split boundaries when present."""
        if value is None:
            return None
        if value.utcoffset() != timedelta(0):
            msg = "split timestamps must use UTC"
            raise ValueError(msg)
        return value.astimezone(UTC)


class PredictionOutput(BoundaryDTO):
    """Model scores aligned to a MultiIndex feature frame."""

    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    scores: pd.Series
    model_hash: Annotated[
        str,
        StringConstraints(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"),
    ]

    @model_validator(mode="after")
    def scores_must_be_finite(self) -> PredictionOutput:
        """Reject empty or non-finite prediction series."""
        if self.scores.empty:
            msg = "prediction scores must not be empty"
            raise ValueError(msg)
        if self.scores.isna().any():
            msg = "prediction scores must not contain NaNs"
            raise ValueError(msg)
        return self


class ICMetricsReport(BoundaryDTO):
    """Cross-sectional IC diagnostics for one OOS prediction set."""

    ic_mean: FiniteFloat
    rank_ic_mean: FiniteFloat
    icir: FiniteFloat | None
    rank_icir: FiniteFloat | None
    median_cs_breadth: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    cs_breadth_warning: bool
    ic_materially_positive: bool
    ic_positive_threshold: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    n_dates: Annotated[int, Field(ge=0)]
    warnings: tuple[NonEmptyText, ...] = ()


class Phase2Sidecar(BoundaryDTO):
    """JSON sidecar persisted beside the native LightGBM artifact."""

    model_hash: Annotated[
        str,
        StringConstraints(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"),
    ]
    git_commit: NonEmptyText
    universe: Annotated[tuple[NonEmptyText, ...], Field(min_length=1)]
    feature_list: Annotated[tuple[NonEmptyText, ...], Field(min_length=1)]
    hyperparams: dict[NonEmptyText, object]
    horizon: Annotated[int, Field(ge=1)]
    split_def: SplitDefinition
    data_range: dict[NonEmptyText, NonEmptyText]
    lib_versions: dict[NonEmptyText, NonEmptyText]
    seed: Annotated[int, Field(ge=0)]
    metrics: ICMetricsReport
    strategy_id: NonEmptyText
    phase: Literal[2] = 2

    @field_validator("universe", "feature_list", mode="before")
    @classmethod
    def freeze_sidecar_sequences(cls, value: object) -> object:
        """Convert list payloads to immutable tuples at the boundary."""
        if isinstance(value, list):
            return tuple(value)
        return value


class CPCVConfig(BoundaryDTO):
    """Combinatorial purged cross-validation parameters.

    Unique timestamps are partitioned into ``n_blocks`` contiguous blocks.
    Every combination of ``n_test_blocks`` blocks forms one test fold; the
    remainder is training after purge + embargo.
    """

    n_blocks: Annotated[int, Field(ge=2)]
    n_test_blocks: Annotated[int, Field(ge=1)]
    embargo_days: Annotated[int, Field(ge=0)]
    horizon: Annotated[int, Field(ge=1)]

    @model_validator(mode="after")
    def test_blocks_must_fit(self) -> CPCVConfig:
        """Require k < N so every fold retains a non-empty train pool."""
        if self.n_test_blocks >= self.n_blocks:
            msg = "n_test_blocks must be < n_blocks"
            raise ValueError(msg)
        return self


class CPCVFold(BoundaryDTO):
    """One combinatorial fold after purge and embargo.

    ``train_positions`` / ``test_positions`` are integer locations into the
    row order of the input ``BaselineDataset`` (``iloc`` indices).
    """

    fold_id: Annotated[int, Field(ge=0)]
    test_block_ids: Annotated[tuple[Annotated[int, Field(ge=0)], ...], Field(min_length=1)]
    train_positions: Annotated[tuple[Annotated[int, Field(ge=0)], ...], Field(min_length=1)]
    test_positions: Annotated[tuple[Annotated[int, Field(ge=0)], ...], Field(min_length=1)]
    purged_train_count: Annotated[int, Field(ge=0)]
    embargoed_train_count: Annotated[int, Field(ge=0)]


class CPCVSplitResult(BoundaryDTO):
    """Full CPCV enumeration for one dataset."""

    config: CPCVConfig
    n_folds: Annotated[int, Field(ge=1)]
    n_timestamps: Annotated[int, Field(ge=1)]
    folds: Annotated[tuple[CPCVFold, ...], Field(min_length=1)]


class SharpeReport(BoundaryDTO):
    """Non-annualized Sharpe plus higher moments used by DSR."""

    sharpe: FiniteFloat
    n_obs: Annotated[int, Field(ge=2)]
    skewness: FiniteFloat
    kurtosis: FiniteFloat  # Pearson (normal == 3)


class DeflatedSharpeReport(BoundaryDTO):
    """Bailey & López de Prado (2014) Deflated Sharpe Ratio result."""

    observed_sharpe: FiniteFloat
    sr0: FiniteFloat
    dsr: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    n_trials: Annotated[int, Field(ge=1)]
    var_trial_sharpes: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    n_obs: Annotated[int, Field(ge=2)]
    skewness: FiniteFloat
    kurtosis: FiniteFloat


class GatePathResult(BoundaryDTO):
    """Per-CPCV-fold OOS path diagnostics."""

    fold_id: Annotated[int, Field(ge=0)]
    sharpe: FiniteFloat
    n_return_obs: Annotated[int, Field(ge=1)]
    mean_return: FiniteFloat


class GateVerdict(BoundaryDTO):
    """Phase-3 overfitting gate: DSR >= 0.95 → JA, otherwise NEIN.

    ``dsr`` is Φ(z) from Bailey & López de Prado (2014). The 0.95 threshold
    is the conventional confidence level that the true SR exceeds 0 after
    selection-bias deflation. ``dsr > 0`` is not used — Φ(z) is almost always
    strictly positive for finite z and would rubber-stamp near-noise signals.
    """

    verdict: Literal["JA", "NEIN"]
    dsr: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    n_trials: Annotated[int, Field(ge=1)]
    path_sharpe_mean: FiniteFloat
    path_sharpe_std: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    path_results: Annotated[tuple[GatePathResult, ...], Field(min_length=1)]
    deflated: DeflatedSharpeReport
    cpcv: CPCVSplitResult
    horizon: Annotated[int, Field(ge=1)]
    seed: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def verdict_must_match_dsr(self) -> GateVerdict:
        """Hard-bind JA/NEIN to the DSR >= 0.95 rule."""
        expected: Literal["JA", "NEIN"] = "JA" if self.dsr >= 0.95 else "NEIN"
        if self.verdict != expected:
            msg = f"verdict {self.verdict!r} inconsistent with dsr={self.dsr}"
            raise ValueError(msg)
        return self
