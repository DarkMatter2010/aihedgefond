"""Injectable runtime providers for IDs and wall-clock time."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from random import Random
from typing import Protocol
from uuid import UUID, uuid4


class IdProvider(Protocol):
    """Boundary provider for newly minted DTO identifiers."""

    def new_id(self) -> UUID:
        """Return the next identifier."""


class Clock(Protocol):
    """Boundary provider for current UTC time."""

    def now(self) -> datetime:
        """Return the current UTC timestamp."""


@dataclass(frozen=True)
class Uuid4IdProvider:
    """Production identifier provider preserving UUID4 behavior."""

    def new_id(self) -> UUID:
        """Mint a random UUID4."""
        return uuid4()


@dataclass
class SeededIdProvider:
    """Deterministic identifier provider for repeatable runs."""

    seed: int
    _random: Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._random = Random(self.seed)

    def new_id(self) -> UUID:
        """Mint a deterministic UUID4-shaped identifier."""
        return UUID(int=self._random.getrandbits(128), version=4)


@dataclass(frozen=True)
class SystemClock:
    """Production UTC wall clock."""

    def now(self) -> datetime:
        """Return current UTC time."""
        return datetime.now(UTC)


@dataclass(frozen=True)
class FrozenClock:
    """Clock fixed at one UTC timestamp for deterministic runs."""

    timestamp: datetime

    def __post_init__(self) -> None:
        if self.timestamp.utcoffset() != timedelta(0):
            msg = "frozen timestamp must use UTC"
            raise ValueError(msg)
        object.__setattr__(self, "timestamp", self.timestamp.astimezone(UTC))

    def now(self) -> datetime:
        """Return the configured timestamp."""
        return self.timestamp
