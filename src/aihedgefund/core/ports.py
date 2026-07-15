"""Abstract infrastructure ports implemented by concrete adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from aihedgefund.core.schemas import (
    Fill,
    LoadModelArtifactRequest,
    LoadModelArtifactResult,
    OHLCVBar,
    OHLCVRequest,
    Order,
    Position,
    SaveModelArtifactRequest,
    SaveModelArtifactResult,
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
    """Vendor-neutral port for persisting and restoring trained models."""

    @abstractmethod
    def save_model(self, request: SaveModelArtifactRequest) -> SaveModelArtifactResult:
        """Persist a validated model payload and metadata."""

    @abstractmethod
    def load_model(self, request: LoadModelArtifactRequest) -> LoadModelArtifactResult:
        """Restore a uniquely identified validated model artifact."""
