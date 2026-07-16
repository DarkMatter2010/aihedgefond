"""Abstract vendor-facing ports for future adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from aihedgefund.core.schemas import (
    Fill,
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
