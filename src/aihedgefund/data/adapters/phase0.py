"""Compatibility bridge from the Phase 1 provider to the Phase 0 data port."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from aihedgefund.core.ports import DataVendorPort
from aihedgefund.core.schemas import MarketDataRequest, OHLCVBar, OHLCVRequest
from aihedgefund.data.provider import MarketDataProvider


class Phase0DataVendorAdapter(DataVendorPort):
    """Preserve the original single-symbol port without changing its contract."""

    def __init__(
        self,
        provider: MarketDataProvider,
        *,
        frequency: Literal["1d"] = "1d",
    ) -> None:
        self._provider = provider
        self._frequency = frequency

    def get_ohlcv(self, request: OHLCVRequest) -> tuple[OHLCVBar, ...]:
        """Map canonical Phase 1 data back to the original raw-bar DTO."""
        result = self._provider.get_ohlcv(
            MarketDataRequest(
                symbols=(request.symbol,),
                start=request.start,
                end=request.end,
                frequency=self._frequency,
            )
        )
        frame = result.bars[request.symbol]
        return tuple(
            OHLCVBar(
                symbol=request.symbol,
                timestamp=timestamp.to_pydatetime(),
                open=Decimal(str(row["open"])),
                high=Decimal(str(row["high"])),
                low=Decimal(str(row["low"])),
                close=Decimal(str(row["close"])),
                volume=Decimal(str(row["volume"])),
            )
            for timestamp, row in frame.iterrows()
        )
