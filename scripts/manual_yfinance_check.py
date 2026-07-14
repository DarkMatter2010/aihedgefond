"""Manual live yfinance smoke test; deliberately excluded from pytest collection."""

from __future__ import annotations

from datetime import UTC, datetime, time

from aihedgefund.core.bus import InProcessMessageBus
from aihedgefund.core.config import load_settings
from aihedgefund.core.schemas import MarketDataRequest
from aihedgefund.data.adapters import YFinanceProvider
from aihedgefund.data.quality import DataQualityGate


def main() -> None:
    """Fetch configured daily bars from Yahoo and print deterministic metadata."""
    settings = load_settings()
    bus = InProcessMessageBus()
    provider = YFinanceProvider(
        settings.symbol_aliases,
        bus,
        DataQualityGate(settings.quality, bus),
    )
    result = provider.get_ohlcv(
        MarketDataRequest(
            symbols=settings.universe,
            start=datetime.combine(settings.start, time.min, tzinfo=UTC),
            end=datetime.combine(settings.end, time.min, tzinfo=UTC),
            frequency=settings.frequency,
        )
    )
    for symbol, frame in result.bars.items():
        print(symbol, len(frame), frame.index.min(), frame.index.max())


if __name__ == "__main__":
    main()
