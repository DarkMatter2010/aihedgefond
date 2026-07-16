"""Filesystem adapter for native LightGBM model artifacts."""

from __future__ import annotations

import os
from pathlib import Path

from lightgbm import Booster

from aihedgefund.core.ports import ModelArtifactPort
from aihedgefund.core.schemas import ModelArtifactMetadata

MODEL_FILENAME = "model.txt"
METADATA_FILENAME = "metadata.json"


class FilesystemModelArtifactAdapter(ModelArtifactPort):
    """Persist LightGBM boosters under ``<artifact_root>/models/...``."""

    def __init__(self, artifact_root: Path) -> None:
        self._artifact_root = artifact_root

    def save(self, model: Booster, metadata: ModelArtifactMetadata) -> Path:
        """Write ``model.txt`` and ``metadata.json``; hard-fail on bad roots."""
        self._require_writable_root()
        if metadata.model_file != MODEL_FILENAME:
            msg = f"model_file must be {MODEL_FILENAME!r} for native LightGBM artifacts"
            raise ValueError(msg)
        if metadata.model_format != "lightgbm_native":
            msg = "model_format must be 'lightgbm_native'"
            raise ValueError(msg)

        artifact_dir = (
            self._artifact_root / "models" / metadata.strategy_id / metadata.model_hash
        )
        artifact_dir.mkdir(parents=True, exist_ok=False)

        model_path = artifact_dir / MODEL_FILENAME
        model.save_model(str(model_path))

        metadata_path = artifact_dir / METADATA_FILENAME
        metadata_path.write_text(
            metadata.model_dump_json(),
            encoding="utf-8",
        )
        return artifact_dir

    def load(self, model_hash: str) -> tuple[Booster, ModelArtifactMetadata]:
        """Locate an artifact by hash and reload booster plus metadata."""
        matches = sorted(
            path
            for path in (self._artifact_root / "models").glob(f"*/{model_hash}")
            if path.is_dir()
        )
        if not matches:
            msg = f"model artifact not found for hash {model_hash!r}"
            raise FileNotFoundError(msg)
        if len(matches) > 1:
            msg = f"ambiguous model artifact hash {model_hash!r}: {matches}"
            raise FileNotFoundError(msg)

        artifact_dir = matches[0]
        model_path = artifact_dir / MODEL_FILENAME
        metadata_path = artifact_dir / METADATA_FILENAME
        if not model_path.is_file() or not metadata_path.is_file():
            msg = f"incomplete model artifact at {artifact_dir}"
            raise FileNotFoundError(msg)

        metadata = ModelArtifactMetadata.model_validate_json(
            metadata_path.read_text(encoding="utf-8")
        )
        model = Booster(model_file=str(model_path))
        return model, metadata

    def _require_writable_root(self) -> None:
        """Reject missing or non-writable artifact roots without fallback."""
        root = self._artifact_root
        if not root.exists():
            msg = f"artifact_root does not exist: {root}"
            raise FileNotFoundError(msg)
        if not root.is_dir():
            msg = f"artifact_root is not a directory: {root}"
            raise NotADirectoryError(msg)
        if not os.access(root, os.W_OK):
            msg = f"artifact_root is not writable: {root}"
            raise PermissionError(msg)
