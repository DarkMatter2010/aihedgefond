"""Manual multi-horizon IC sweep on real Yahoo data; excluded from pytest/CI."""

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
from aihedgefund.research.horizon_sweep import (
    DEFAULT_HORIZONS,
    format_sweep_table,
    run_horizon_sweep,
)


def main() -> None:
    """Fetch Yahoo OHLCV once, then sweep IC metrics across forecast horizons."""
    settings = load_settings()
    request_start = datetime.combine(settings.start, time.min, tzinfo=UTC)
    request_end = datetime.combine(settings.end, time.min, tzinfo=UTC)
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

    feature_pipeline = FeaturePipeline(bus, clock=clock)
    feature_matrix = feature_pipeline.compute(bars)
    settings.artifact_root.mkdir(parents=True, exist_ok=True)
    report = run_horizon_sweep(
        bars,
        feature_matrix,
        settings,
        horizons=DEFAULT_HORIZONS,
        artifact_adapter=FilesystemModelArtifactAdapter(settings.artifact_root),
        created_at=clock.now(),
        persist_artifact=True,
    )

    actual_starts = [frame.index.min() for frame in bars.bars.values()]
    actual_ends = [frame.index.max() for frame in bars.bars.values()]
    print(f"data_range: {min(actual_starts)} -> {max(actual_ends)}")
    print(f"n_symbols: {report.n_symbols}")
    print(f"seed: {report.seed}")
    print(f"horizons: {list(DEFAULT_HORIZONS)}")
    print()
    print(format_sweep_table(report))
    for row in report.rows:
        print(
            f"horizon={row.horizon} embargo_days={row.embargo_days} "
            f"train_end={row.train_end} test_start={row.test_start}"
        )


if __name__ == "__main__":
    main()
