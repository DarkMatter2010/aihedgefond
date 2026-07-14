"""Vendor-neutral market-data provider port and bounded fallback orchestration."""

from __future__ import annotations

from abc import ABC, abstractmethod

from aihedgefund.core.schemas import BarFrame, MarketDataRequest


class DataUnavailableError(RuntimeError):
    """Raised when no provider can return complete canonical market data."""


class MarketDataProvider(ABC):
    """Port implemented by concrete OHLCV vendor adapters."""

    @abstractmethod
    def get_ohlcv(
        self,
        request: MarketDataRequest,
    ) -> BarFrame:
        """Return complete canonical bars and raw corporate actions."""


class ProviderChain(MarketDataProvider):
    """Try each provider with bounded attempts and fail after all are exhausted."""

    def __init__(
        self,
        providers: tuple[MarketDataProvider, ...],
        *,
        max_attempts: int = 2,
    ) -> None:
        if not providers:
            msg = "at least one market-data provider is required"
            raise ValueError(msg)
        if max_attempts < 1:
            msg = "max_attempts must be at least one"
            raise ValueError(msg)
        self._providers = providers
        self._max_attempts = max_attempts

    def get_ohlcv(
        self,
        request: MarketDataRequest,
    ) -> BarFrame:
        """Return the first complete response, never a partial or NaN-filled frame."""
        failures: list[str] = []
        for provider in self._providers:
            for attempt in range(1, self._max_attempts + 1):
                try:
                    result = provider.get_ohlcv(request)
                    self._assert_complete(result, request)
                except Exception as exc:  # noqa: BLE001 - adapters expose heterogeneous failures
                    failures.append(f"{type(provider).__name__} attempt {attempt}: {exc}")
                    continue
                return result
        detail = "; ".join(failures)
        msg = f"all market-data providers failed: {detail}"
        raise DataUnavailableError(msg)

    @staticmethod
    def _assert_complete(result: BarFrame, request: MarketDataRequest) -> None:
        returned_symbols = set(result.bars)
        requested_symbols = set(request.symbols)
        if returned_symbols != requested_symbols:
            missing = sorted(requested_symbols - returned_symbols)
            unexpected = sorted(returned_symbols - requested_symbols)
            msg = f"provider symbol mismatch; missing={missing}, unexpected={unexpected}"
            raise DataUnavailableError(msg)
        for symbol, frame in result.bars.items():
            if frame.empty or frame.isna().any(axis=None):
                msg = f"{symbol} contains an unresolved data gap"
                raise DataUnavailableError(msg)


class SecondaryProviderStub(MarketDataProvider):
    """Explicit placeholder used until a secondary vendor adapter is configured."""

    def get_ohlcv(
        self,
        request: MarketDataRequest,
    ) -> BarFrame:
        """Hard-fail instead of pretending that fallback data exists."""
        del request
        msg = "secondary market-data provider is not configured"
        raise DataUnavailableError(msg)
