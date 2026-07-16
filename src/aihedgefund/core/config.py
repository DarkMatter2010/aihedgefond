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


class Settings(ConfigModel):
    """Complete application settings, additively extended for Phase 1."""

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
    artifact_root: Path = Path("artifacts/")

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
