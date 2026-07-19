"""Phase-2 universe-breadth diagnostic: same stack, only the universe changes.

Compares the production FEATURE_COLUMNS + LightGBM baseline on the broad
liquid candidate list vs the documented 50-name h=2 Rank-IC reference.
No CPCV / DSR gate — triage only.
"""

from __future__ import annotations

from datetime import datetime
from typing import Final, Literal

from pydantic import Field

from aihedgefund.core.config import Settings
from aihedgefund.core.schemas import (
    BarFrame,
    BoundaryDTO,
    FiniteFloat,
    ICMetricsReport,
    NonEmptyText,
)
from aihedgefund.features.pipeline import FEATURE_COLUMNS, FeaturePipeline
from aihedgefund.research.adapters.filesystem import FilesystemModelArtifactAdapter
from aihedgefund.research.run_baseline import run_baseline
from aihedgefund.research.universes import (
    BROAD_LIQUID_CANDIDATE_UNIVERSE,
    SURVIVORSHIP_BIAS_NOTE,
)

# Fixed to match the documented 50-name multi-horizon sweep candidate (h=2).
DIAGNOSTIC_HORIZON: Final[int] = 2
NARROW_UNIVERSE_N: Final[int] = 50
NARROW_H2_RANK_IC: Final[float] = 0.0176
RANK_IC_MATERIAL_THRESHOLD: Final[float] = 0.02

Interpretation = Literal["breadth_noise", "features_dead", "inconclusive"]


class UniverseBreadthDiagnosticReport(BoundaryDTO):
    """OOS triage metrics for the single-variable universe swap."""

    universe_label: NonEmptyText
    n_symbols_requested: int = Field(ge=1)
    n_symbols: int = Field(ge=1)
    horizon: int = Field(ge=1)
    seed: int = Field(ge=0)
    feature_columns: tuple[NonEmptyText, ...] = Field(min_length=1)
    metrics: ICMetricsReport
    narrow_rank_ic_reference: FiniteFloat
    narrow_universe_n: int = Field(ge=1)
    interpretation: Interpretation
    interpretation_note: NonEmptyText
    survivorship_bias_note: NonEmptyText
    counts_as_research_trial: bool = True


def interpret_breadth_vs_narrow(
    *,
    rank_ic_broad: float,
    rank_ic_narrow: float = NARROW_H2_RANK_IC,
    threshold: float = RANK_IC_MATERIAL_THRESHOLD,
) -> tuple[Interpretation, str]:
    """Apply the registered interpretation rule (relative diagnosis only)."""
    materially_positive = rank_ic_broad >= threshold
    clearly_above_narrow = rank_ic_broad > rank_ic_narrow * 1.5
    near_zero = abs(rank_ic_broad) < 0.005

    if materially_positive and clearly_above_narrow:
        return (
            "breadth_noise",
            (
                f"Rank-IC broad={rank_ic_broad:.4f} >= {threshold} and clearly "
                f"> 50-name h=2 reference {rank_ic_narrow:.4f} -> signal was "
                "noisy from thin cross-section, not dead. Next: continue on "
                "broad universe."
            ),
        )
    if near_zero or (not materially_positive and rank_ic_broad <= rank_ic_narrow):
        return (
            "features_dead",
            (
                f"Rank-IC broad={rank_ic_broad:.4f} still ~0 / not clearly above "
                f"50-name reference {rank_ic_narrow:.4f} -> features look dead. "
                "Next: expand feature classes (reversal / low-vol / illiquidity) "
                "from OHLCV."
            ),
        )
    return (
        "inconclusive",
        (
            f"Rank-IC broad={rank_ic_broad:.4f} vs narrow={rank_ic_narrow:.4f} "
            f"(threshold {threshold}): neither clearly rescued by breadth nor "
            "flat at zero. Report Pearson IC and Rank-IC signs; do not force "
            "a verdict."
        ),
    )


def settings_for_breadth_diagnostic(settings: Settings) -> Settings:
    """Copy settings with only universe + h=2 (embargo matched) changed."""
    research = settings.research.model_copy(
        update={
            "horizon": DIAGNOSTIC_HORIZON,
            "embargo_days": DIAGNOSTIC_HORIZON,
        }
    )
    return settings.model_copy(
        update={
            "universe": BROAD_LIQUID_CANDIDATE_UNIVERSE,
            "research": research,
        }
    )


def run_universe_breadth_diagnostic(
    bars: BarFrame,
    settings: Settings,
    *,
    feature_pipeline: FeaturePipeline,
    artifact_adapter: FilesystemModelArtifactAdapter,
    created_at: datetime,
    narrow_rank_ic_reference: float = NARROW_H2_RANK_IC,
) -> UniverseBreadthDiagnosticReport:
    """Train/evaluate Phase-2 baseline on ``bars``; report OOS IC triage.

    ``settings`` must already carry the broad universe and diagnostic horizon
    (see ``settings_for_breadth_diagnostic``). Uses the production
    ``FEATURE_COLUMNS`` path inside ``run_baseline`` — no new features / HPs.
    """
    if tuple(settings.universe) != BROAD_LIQUID_CANDIDATE_UNIVERSE:
        # Allow a kept subset after download drops: requested set must be the
        # broad list; bars may contain fewer symbols that passed quality.
        requested = set(BROAD_LIQUID_CANDIDATE_UNIVERSE)
        actual = set(settings.universe)
        if not actual.issubset(requested):
            msg = "settings.universe must be a subset of BROAD_LIQUID_CANDIDATE_UNIVERSE"
            raise ValueError(msg)
    if settings.research.horizon != DIAGNOSTIC_HORIZON:
        msg = f"research.horizon must be {DIAGNOSTIC_HORIZON} for this diagnostic"
        raise ValueError(msg)
    if settings.research.embargo_days != DIAGNOSTIC_HORIZON:
        msg = "research.embargo_days must equal diagnostic horizon"
        raise ValueError(msg)

    bar_symbols = tuple(sorted(bars.bars))
    if not bar_symbols:
        msg = "bars must contain at least one symbol"
        raise ValueError(msg)
    # Align settings.universe to symbols actually present in bars (download drops).
    run_settings = settings.model_copy(update={"universe": bar_symbols})

    _train, _test, _split, _preds, metrics, _dir, _sidecar = run_baseline(
        bars,
        run_settings,
        feature_pipeline=feature_pipeline,
        artifact_adapter=artifact_adapter,
        created_at=created_at,
    )
    interpretation, note = interpret_breadth_vs_narrow(
        rank_ic_broad=metrics.rank_ic_mean,
        rank_ic_narrow=narrow_rank_ic_reference,
    )
    return UniverseBreadthDiagnosticReport(
        universe_label="broad_liquid_candidate",
        n_symbols_requested=len(BROAD_LIQUID_CANDIDATE_UNIVERSE),
        n_symbols=len(bar_symbols),
        horizon=DIAGNOSTIC_HORIZON,
        seed=run_settings.research.seed,
        feature_columns=FEATURE_COLUMNS,
        metrics=metrics,
        narrow_rank_ic_reference=narrow_rank_ic_reference,
        narrow_universe_n=NARROW_UNIVERSE_N,
        interpretation=interpretation,
        interpretation_note=note,
        survivorship_bias_note=SURVIVORSHIP_BIAS_NOTE,
        counts_as_research_trial=True,
    )
