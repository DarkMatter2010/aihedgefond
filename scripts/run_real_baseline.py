"""Manual real-data Phase-2 baseline; deliberately excluded from pytest/CI."""

from __future__ import annotations

from datetime import UTC, datetime, time

from aihedgefund.core.bus import InProcessMessageBus
from aihedgefund.core.config import load_settings
from aihedgefund.core.runtime import FrozenClock
from aihedgefund.core.schemas import MarketDataRequest
from aihedgefund.data.adapters import YFinanceProvider
from aihedgefund.data.quality import DataQualityGate
from aihedgefund.features.pipeline import FeaturePipeline
from aihedgefund.research.adapters.filesystem import FilesystemModelArtifactAdapter
from aihedgefund.research.run_baseline import run_baseline


def main() -> None:
    """Fetch live Yahoo OHLCV for the configured universe and print IC metrics."""
    settings = load_settings()
    request_start = datetime.combine(settings.start, time.min, tzinfo=UTC)
    request_end = datetime.combine(settings.end, time.min, tzinfo=UTC)
    # Historical research windows must be validated against the request end,
    # not wall-clock time (Phase-1 quality-gate contract for backfill ingest).
    clock = FrozenClock(request_end)
    bus = InProcessMessageBus()
    provider = YFinanceProvider(
        settings.symbol_aliases,
        bus,
        DataQualityGate(settings.quality, bus, clock=clock),
        clock=clock,
    )
    bars = provider.get_ohlcv(
        MarketDataRequest(
            symbols=settings.universe,
            start=request_start,
            end=request_end,
            frequency=settings.frequency,
        )
    )
    (
        _train,
        _test,
        _split,
        _predictions,
        metrics,
        _artifact_dir,
        _sidecar,
    ) = run_baseline(
        bars,
        settings,
        feature_pipeline=FeaturePipeline(bus, clock=clock),
        artifact_adapter=FilesystemModelArtifactAdapter(settings.artifact_root),
        created_at=clock.now(),
    )

    actual_starts = [frame.index.min() for frame in bars.bars.values()]
    actual_ends = [frame.index.max() for frame in bars.bars.values()]
    data_start = min(actual_starts)
    data_end = max(actual_ends)

    print(f"ic_mean: {metrics.ic_mean}")
    print(f"rank_ic_mean: {metrics.rank_ic_mean}")
    print(f"icir: {metrics.icir}")
    print(f"ic_materially_positive: {metrics.ic_materially_positive}")
    print(f"median_cs_breadth: {metrics.median_cs_breadth}")
    print(f"cs_breadth_warning: {metrics.cs_breadth_warning}")
    print(f"n_symbols: {len(bars.bars)}")
    print(f"data_range: {data_start} -> {data_end}")


if __name__ == "__main__":
    main()
