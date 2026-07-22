"""Abstract vendor-facing ports for future adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from aihedgefund.core.schemas import (
    Fill,
    Form4Frame,
    Form4Request,
    ModelArtifactLoadResult,
    ModelArtifactSaveRequest,
    OHLCVBar,
    OHLCVRequest,
    Order,
    Position,
)


class DataVendorPort(ABC):
    """Port implemented by market-data adapters."""

    @abstractmethod
    def get_ohlcv(self, request: OHLCVRequest) -> tuple[OHLCVBar, ...]:
        """Return validated bars for a typed request."""


class InsiderFilingPort(ABC):
    """Port implemented by SEC Form 4 (insider) filing adapters.

    Point-in-time note: only ``filed_at`` (acceptance time) is safe to use as
    the availability timestamp. ``transaction_date`` may precede filing by up
    to ~2 business days and must not drive feature availability.
    """

    @abstractmethod
    def get_form4(self, request: Form4Request) -> Form4Frame:
        """Return validated Form 4 transaction rows for the request window."""


class BrokerPort(ABC):
    """Port implemented by broker adapters."""

    @abstractmethod
    def submit_order(self, order: Order) -> Fill | None:
        """Submit an order and return an immediate fill when one exists."""

    @abstractmethod
    def get_positions(self) -> tuple[Position, ...]:
        """Return current broker positions as immutable DTOs."""


class ModelArtifactPort(ABC):
    """Port implemented by trained-model persistence adapters."""

    @abstractmethod
    def save(self, request: ModelArtifactSaveRequest) -> Path:
        """Persist a native model blob and metadata; return the artifact directory."""

    @abstractmethod
    def load(self, model_hash: str) -> ModelArtifactLoadResult:
        """Load a model blob and metadata by hash; raise if the artifact is missing."""
