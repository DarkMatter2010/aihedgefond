"""Typed configuration models and YAML loading."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

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


class Settings(ConfigModel):
    """Complete Phase 0 application settings."""

    trading_limits: TradingLimits
    universe: Annotated[tuple[NonEmptyText, ...], Field(min_length=1)]
    feature_flags: dict[NonEmptyText, bool]

    @field_validator("universe", mode="before")
    @classmethod
    def freeze_yaml_sequence(cls, value: object) -> object:
        """Convert YAML's native list representation to an immutable tuple."""
        if isinstance(value, list):
            return tuple(value)
        return value


DEFAULT_CONFIG_PATH = Path(__file__).parents[1] / "config" / "limits.yaml"


def load_settings(path: Path = DEFAULT_CONFIG_PATH) -> Settings:
    """Load and fully validate settings, failing on missing or malformed input."""
    with path.open(encoding="utf-8") as config_file:
        raw_config: object = yaml.safe_load(config_file)

    if not isinstance(raw_config, dict):
        msg = f"configuration root in {path} must be a mapping"
        raise ValueError(msg)

    return Settings.model_validate(raw_config)
