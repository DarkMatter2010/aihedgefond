"""Filesystem persistence for native LightGBM model artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import lightgbm as lgb
from pydantic import ValidationError

from aihedgefund.core.config import Settings
from aihedgefund.core.ports import ModelArtifactPort
from aihedgefund.core.schemas import ModelArtifactMetadata

MODEL_FILENAME = "model.txt"
METADATA_FILENAME = "metadata.json"


def compute_model_hash(
    *,
    features: tuple[str, ...],
    hyperparameters: dict[str, object],
    universe: tuple[str, ...],
    settings: Settings,
) -> str:
    """Return SHA-256 of canonical training inputs in documented field order.

    The canonical UTF-8 JSON sequence contains features (sorted), hyperparameters
    (sorted by key), universe (sorted), then settings start, end, and frequency.
    JSON object keys nested inside hyperparameter values are sorted recursively.
    """
    canonical_inputs: list[object] = [
        ["features", sorted(features)],
        [
            "hyperparameters",
            [[key, hyperparameters[key]] for key in sorted(hyperparameters)],
        ],
        ["universe", sorted(universe)],
        ["start", settings.start.isoformat()],
        ["end", settings.end.isoformat()],
        ["frequency", settings.frequency],
    ]
    try:
        canonical_json = json.dumps(
            canonical_inputs,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        msg = "model hash inputs must be finite and JSON serializable"
        raise ValueError(msg) from exc
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


class FilesystemModelArtifactAdapter(ModelArtifactPort):
    """Persist native LightGBM models and JSON metadata below one fixed root."""

    def __init__(self, artifact_root: Path) -> None:
        self._artifact_root = artifact_root

    def save_model(self, model: lgb.Booster, metadata: ModelArtifactMetadata) -> Path:
        """Save ``model.txt`` and ``metadata.json`` without root fallback."""
        self._require_writable_root()
        if metadata.model_file != MODEL_FILENAME:
            msg = f"model_file must be {MODEL_FILENAME!r}"
            raise ValueError(msg)

        artifact_directory = (
            self._artifact_root / "models" / metadata.strategy_id / metadata.model_hash
        )
        artifact_directory.mkdir(parents=True, exist_ok=True)
        model.save_model(artifact_directory / MODEL_FILENAME)
        (artifact_directory / METADATA_FILENAME).write_text(
            f"{metadata.model_dump_json(indent=2)}\n",
            encoding="utf-8",
        )
        return artifact_directory

    def load_model(self, model_hash: str) -> tuple[lgb.Booster, ModelArtifactMetadata]:
        """Load the single artifact matching ``model_hash`` or fail hard."""
        self._require_existing_root()
        artifact_directory = self._find_artifact_directory(model_hash)
        metadata_path = artifact_directory / METADATA_FILENAME
        if not metadata_path.is_file():
            msg = f"metadata not found for model hash {model_hash!r}"
            raise FileNotFoundError(msg)

        try:
            metadata = ModelArtifactMetadata.model_validate_json(
                metadata_path.read_text(encoding="utf-8")
            )
        except ValidationError as exc:
            msg = f"invalid metadata for model hash {model_hash!r}"
            raise ValueError(msg) from exc

        if metadata.model_hash != model_hash:
            msg = "metadata model_hash does not match artifact directory"
            raise ValueError(msg)
        if metadata.strategy_id != artifact_directory.parent.name:
            msg = "metadata strategy_id does not match artifact directory"
            raise ValueError(msg)
        if metadata.model_file != MODEL_FILENAME:
            msg = f"model_file must be {MODEL_FILENAME!r}"
            raise ValueError(msg)

        model_path = artifact_directory / metadata.model_file
        if not model_path.is_file():
            msg = f"native model file not found for model hash {model_hash!r}"
            raise FileNotFoundError(msg)
        return lgb.Booster(model_file=str(model_path)), metadata

    def _require_existing_root(self) -> None:
        if not self._artifact_root.exists():
            msg = f"artifact_root does not exist: {self._artifact_root}"
            raise FileNotFoundError(msg)
        if not self._artifact_root.is_dir():
            msg = f"artifact_root is not a directory: {self._artifact_root}"
            raise NotADirectoryError(msg)

    def _require_writable_root(self) -> None:
        self._require_existing_root()
        write_bits = self._artifact_root.stat().st_mode & 0o222
        if write_bits == 0 or not os.access(self._artifact_root, os.W_OK):
            msg = f"artifact_root is not writable: {self._artifact_root}"
            raise PermissionError(msg)

    def _find_artifact_directory(self, model_hash: str) -> Path:
        if not model_hash or model_hash in {".", ".."} or "/" in model_hash or "\\" in model_hash:
            msg = "model_hash must be a non-empty path segment"
            raise ValueError(msg)

        models_root = self._artifact_root / "models"
        if not models_root.is_dir():
            msg = f"model hash not found: {model_hash}"
            raise FileNotFoundError(msg)
        matches = sorted(
            strategy_directory / model_hash
            for strategy_directory in models_root.iterdir()
            if strategy_directory.is_dir() and (strategy_directory / model_hash).is_dir()
        )
        if not matches:
            msg = f"model hash not found: {model_hash}"
            raise FileNotFoundError(msg)
        if len(matches) > 1:
            msg = f"model hash is ambiguous across strategies: {model_hash}"
            raise RuntimeError(msg)
        return matches[0]
