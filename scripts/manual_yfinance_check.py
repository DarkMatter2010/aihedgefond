"""Manual live yfinance smoke test; deliberately excluded from pytest collection."""

from __future__ import annotations

from datetime import UTC, datetime, time

from aihedgefund.core.bus import InProcessMessageBus
from aihedgefund.core.config import load_settings
from aihedgefund.data.adapters import YFinanceProvider


def main() -> None:
    """Fetch configured daily bars from Yahoo and print deterministic metadata."""
    settings = load_settings()
    provider = YFinanceProvider(settings.symbol_aliases, InProcessMessageBus())
    result = provider.get_ohlcv(
        settings.universe,
        datetime.combine(settings.start, time.min, tzinfo=UTC),
        datetime.combine(settings.end, time.min, tzinfo=UTC),
        settings.frequency,
    )
    for symbol, frame in result.bars.items():
        print(symbol, len(frame), frame.index.min(), frame.index.max())


if __name__ == "__main__":
    main()
