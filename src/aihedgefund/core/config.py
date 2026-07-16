"""Typed configuration models and YAML loading."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ConfigModel(BaseModel):
    """Strict and immutable configuration base."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


class TradingLimits(ConfigModel):
    """Business limits loaded from configuration."""

    max_position_size: Annotated[float, Field(gt=0)]
    max_sector_exposure: Annotated[float, Field(gt=0, le=1)]
    max_turnover: Annotated[float, Field(gt=0)]
    max_daily_drawdown: Annotated[float, Field(gt=0, le=1)]


class QualitySettings(ConfigModel):
    """Hard-fail thresholds for the market-data quality gate."""

    max_nan_ratio: Annotated[float, Field(ge=0, le=1)]
    max_abs_logret: Annotated[float, Field(gt=0)]
    stale_bars: Annotated[int, Field(ge=2)]
    zscore_cap: Annotated[float, Field(gt=0)]
    max_last_bar_age_days: Annotated[int, Field(ge=0)]


class LabelSettings(ConfigModel):
    """Event sampling and triple-barrier parameters."""

    vol_span: Annotated[int, Field(ge=2)]
    cusum_threshold: Annotated[float, Field(gt=0)]
    pt: Annotated[float, Field(gt=0)]
    sl: Annotated[float, Field(gt=0)]
    vertical_bars: Annotated[int, Field(ge=1)]


class FracDiffSettings(ConfigModel):
    """Fixed-width fractional-differentiation parameters."""

    d: Annotated[float, Field(ge=0, le=1)]
    thresh: Annotated[float, Field(gt=0, lt=1)]


class ResearchSettings(ConfigModel):
    """Phase-2 baseline research parameters (fixed splits, no proportional cuts)."""

    horizon: Annotated[int, Field(ge=1)]
    embargo_days: Annotated[int, Field(ge=1)]
    seed: Annotated[int, Field(ge=0)]
    train_end: date
    test_start: date
    strategy_id: NonEmptyText
    num_boost_round: Annotated[int, Field(ge=1)]
    learning_rate: Annotated[float, Field(gt=0, lt=1)]
    num_leaves: Annotated[int, Field(ge=2)]
    min_data_in_leaf: Annotated[int, Field(ge=1)]
    feature_fraction: Annotated[float, Field(gt=0, le=1)]
    bagging_fraction: Annotated[float, Field(gt=0, le=1)]
    bagging_freq: Annotated[int, Field(ge=0)]
    ic_positive_threshold: Annotated[float, Field(gt=0)]
    min_cs_breadth_for_reliable_ic: Annotated[int, Field(ge=1)]

    @model_validator(mode="after")
    def validate_split_and_embargo(self) -> ResearchSettings:
        """Require embargo ≥ horizon and a non-overlapping calendar split."""
        if self.embargo_days < self.horizon:
            msg = "research.embargo_days must be >= research.horizon"
            raise ValueError(msg)
        if self.test_start <= self.train_end:
            msg = "research.test_start must be later than research.train_end"
            raise ValueError(msg)
        calendar_gap = (self.test_start - self.train_end).days
        if calendar_gap < self.embargo_days:
            msg = (
                "research calendar gap (test_start - train_end) must be "
                f">= embargo_days ({self.embargo_days}); got {calendar_gap}"
            )
            raise ValueError(msg)
        return self


class Settings(ConfigModel):
    """Complete application settings, additively extended for Phase 1+2."""

    trading_limits: TradingLimits
    universe: Annotated[tuple[NonEmptyText, ...], Field(min_length=1)]
    feature_flags: dict[NonEmptyText, bool]
    start: date
    end: date
    frequency: Literal["1d"]
    symbol_aliases: dict[NonEmptyText, NonEmptyText]
    quality: QualitySettings
    labels: LabelSettings
    fracdiff: FracDiffSettings
    artifact_root: Path
    research: ResearchSettings

    @field_validator("universe", mode="before")
    @classmethod
    def freeze_yaml_sequence(cls, value: object) -> object:
        """Convert YAML's native list representation to an immutable tuple."""
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("artifact_root", mode="before")
    @classmethod
    def coerce_artifact_root(cls, value: object) -> object:
        """Accept YAML strings while keeping a typed Path in Settings."""
        if isinstance(value, str):
            return Path(value)
        return value

    @model_validator(mode="after")
    def end_must_follow_start(self) -> Settings:
        """Reject empty or reversed market-data windows."""
        if self.end <= self.start:
            msg = "end must be later than start"
            raise ValueError(msg)
        if self.research.train_end < self.start or self.research.test_start > self.end:
            msg = "research split dates must lie within settings.start/end"
            raise ValueError(msg)
        return self


DEFAULT_CONFIG_PATH = Path(__file__).parents[1] / "config" / "limits.yaml"


def load_settings(path: Path = DEFAULT_CONFIG_PATH) -> Settings:
    """Load and fully validate settings, failing on missing or malformed input."""
    with path.open(encoding="utf-8") as config_file:
        raw_config: object = yaml.safe_load(config_file)

    if not isinstance(raw_config, dict):
        msg = f"configuration root in {path} must be a mapping"
        raise ValueError(msg)

    return Settings.model_validate(raw_config)
