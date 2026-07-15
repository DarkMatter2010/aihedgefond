"""Abstract infrastructure ports implemented by concrete adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import lightgbm as lgb

from aihedgefund.core.schemas import (
    Fill,
    ModelArtifactMetadata,
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
    """Port for persisting and restoring native trained models."""

    @abstractmethod
    def save_model(self, model: lgb.Booster, metadata: ModelArtifactMetadata) -> Path:
        """Persist a model with its validated reproducibility metadata."""

    @abstractmethod
    def load_model(self, model_hash: str) -> tuple[lgb.Booster, ModelArtifactMetadata]:
        """Restore the uniquely identified model and validated metadata."""
