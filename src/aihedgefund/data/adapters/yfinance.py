"""yfinance adapter; vendor imports are deliberately confined to this module."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import pandas as pd
import yfinance as yf

from aihedgefund.core.bus import MessageBus
from aihedgefund.core.runtime import Clock, SystemClock
from aihedgefund.core.schemas import (
    BarFrame,
    DataIngested,
    IngestedMarketData,
    MarketDataRequest,
)
from aihedgefund.data.provider import DataUnavailableError, MarketDataProvider
from aihedgefund.data.quality import DataQualityGate

CANONICAL_COLUMNS = ("open", "high", "low", "close", "adj_close", "volume")


class YFinanceProvider(MarketDataProvider):
    """Download Yahoo bars (split-continuous close) and normalize to canonical frames.

    Uses ``auto_adjust=False`` so ``close`` stays split-adjusted but not
    dividend-adjusted, while ``adj_close`` and raw action columns are retained.
    Corporate-action price transforms must not re-apply split factors to ``close``.
    """

    def __init__(
        self,
        symbol_aliases: Mapping[str, str],
        bus: MessageBus,
        quality_gate: DataQualityGate,
        *,
        clock: Clock | None = None,
    ) -> None:
        self._symbol_aliases = dict(symbol_aliases)
        self._bus = bus
        self._quality_gate = quality_gate
        self._clock = clock or SystemClock()

    def get_ohlcv(
        self,
        request: MarketDataRequest,
    ) -> BarFrame:
        """Fetch canonical UTC bars while preserving actions and both close columns."""
        symbols = request.symbols
        vendor_symbols = tuple(self._symbol_aliases.get(symbol, symbol) for symbol in symbols)
        raw = cast(
            pd.DataFrame,
            yf.download(
                tickers=list(vendor_symbols),
                start=request.start,
                end=request.end,
                interval=request.frequency,
                auto_adjust=False,
                actions=True,
                group_by="ticker",
                progress=False,
                threads=False,
            ),
        )
        if raw.empty:
            msg = "yfinance returned no rows"
            raise DataUnavailableError(msg)

        bars: dict[str, pd.DataFrame] = {}
        dividends: dict[str, pd.Series] = {}
        splits: dict[str, pd.Series] = {}
        for alias, vendor_symbol in zip(symbols, vendor_symbols, strict=True):
            vendor_frame = self._extract_symbol(raw, vendor_symbol, len(symbols))
            canonical, symbol_dividends, symbol_splits = self._canonicalize(
                vendor_frame,
                alias,
            )
            bars[alias] = canonical
            dividends[alias] = symbol_dividends
            splits[alias] = symbol_splits

        result = BarFrame(bars=bars, dividends=dividends, splits=splits)
        for symbol, frame in result.bars.items():
            self._quality_gate.validate(frame, symbol)
        self._bus.publish_event(
            DataIngested(
                timestamp=self._clock.now(),
                payload=IngestedMarketData(request=request, data=result),
            )
        )
        return result

    @staticmethod
    def _extract_symbol(
        raw: pd.DataFrame,
        vendor_symbol: str,
        requested_symbol_count: int,
    ) -> pd.DataFrame:
        if not isinstance(raw.columns, pd.MultiIndex):
            if requested_symbol_count != 1:
                msg = "yfinance returned ambiguous single-level columns"
                raise DataUnavailableError(msg)
            return raw.copy()

        first_level = raw.columns.get_level_values(0)
        second_level = raw.columns.get_level_values(1)
        if vendor_symbol in first_level:
            return cast(pd.DataFrame, raw[vendor_symbol]).copy()
        if vendor_symbol in second_level:
            return cast(pd.DataFrame, raw.xs(vendor_symbol, axis=1, level=1)).copy()
        msg = f"yfinance response omitted {vendor_symbol}"
        raise DataUnavailableError(msg)

    @staticmethod
    def _canonicalize(
        frame: pd.DataFrame,
        symbol: str,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        normalized_names = {
            column: str(column).strip().lower().replace(" ", "_") for column in frame.columns
        }
        normalized = frame.rename(columns=normalized_names)
        missing = set(CANONICAL_COLUMNS) - set(normalized.columns)
        if missing:
            msg = f"{symbol} response is missing columns: {sorted(missing)}"
            raise DataUnavailableError(msg)

        index = pd.DatetimeIndex(normalized.index)
        if index.tz is None:
            index = index.tz_localize("UTC")
        else:
            index = index.tz_convert("UTC")
        normalized.index = index

        canonical = normalized.loc[:, list(CANONICAL_COLUMNS)].astype(float)
        if canonical.empty:
            msg = f"{symbol} contains no canonical rows"
            raise DataUnavailableError(msg)

        dividends = YFinanceProvider._action_series(normalized, "dividends", canonical.index)
        splits = YFinanceProvider._action_series(normalized, "stock_splits", canonical.index)
        dividends.name = "dividends"
        splits.name = "splits"
        return canonical, dividends, splits

    @staticmethod
    def _action_series(
        frame: pd.DataFrame,
        column: str,
        index: pd.DatetimeIndex,
    ) -> pd.Series:
        if column not in frame:
            msg = f"yfinance response is missing required action column {column!r}"
            raise DataUnavailableError(msg)
        return cast(pd.Series, frame[column]).astype(float).reindex(index, fill_value=0.0)
