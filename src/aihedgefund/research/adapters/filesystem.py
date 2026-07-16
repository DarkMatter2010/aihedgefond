"""Filesystem adapter for native LightGBM model artifacts."""

from __future__ import annotations

import os
from pathlib import Path

from lightgbm import Booster

from aihedgefund.core.ports import ModelArtifactPort
from aihedgefund.core.schemas import (
    ModelArtifactLoadResult,
    ModelArtifactMetadata,
    ModelArtifactSaveRequest,
)
from aihedgefund.research.model_hash import compute_model_hash

MODEL_FILENAME = "model.txt"
METADATA_FILENAME = "metadata.json"


class FilesystemModelArtifactAdapter(ModelArtifactPort):
    """Persist native LightGBM artifacts under ``<artifact_root>/models/...``."""

    def __init__(self, artifact_root: Path) -> None:
        self._artifact_root = artifact_root

    def save(self, request: ModelArtifactSaveRequest) -> Path:
        """Write ``model.txt`` and ``metadata.json``; hard-fail on bad roots."""
        self._require_writable_root()
        metadata = request.metadata
        self._validate_metadata_contract(metadata)
        self._validate_model_hash(metadata)

        artifact_dir = (
            self._artifact_root / "models" / metadata.strategy_id / metadata.model_hash
        )
        artifact_dir.mkdir(parents=True, exist_ok=False)

        model_path = artifact_dir / MODEL_FILENAME
        model_path.write_bytes(request.model_blob)

        metadata_path = artifact_dir / METADATA_FILENAME
        metadata_path.write_text(
            metadata.model_dump_json(),
            encoding="utf-8",
        )
        return artifact_dir

    def load(self, model_hash: str) -> ModelArtifactLoadResult:
        """Locate an artifact by hash and reload blob plus metadata."""
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
        if metadata.model_hash != model_hash:
            msg = (
                f"metadata model_hash {metadata.model_hash!r} does not match "
                f"requested hash {model_hash!r}"
            )
            raise ValueError(msg)
        self._validate_metadata_contract(metadata)
        self._validate_model_hash(metadata)

        model_blob = model_path.read_bytes()
        model = Booster(model_str=model_blob.decode("utf-8"))
        self._validate_booster_matches_metadata(model, metadata)
        return ModelArtifactLoadResult(model_blob=model_blob, metadata=metadata)

    def save_booster(self, model: Booster, metadata: ModelArtifactMetadata) -> Path:
        """Serialize a LightGBM booster and persist it via the port contract."""
        self._validate_booster_matches_metadata(model, metadata)
        request = ModelArtifactSaveRequest(
            model_blob=model.model_to_string().encode("utf-8"),
            metadata=metadata,
        )
        return self.save(request)

    def load_booster(self, model_hash: str) -> tuple[Booster, ModelArtifactMetadata]:
        """Load an artifact and reconstruct the LightGBM booster."""
        result = self.load(model_hash)
        model = Booster(model_str=result.model_blob.decode("utf-8"))
        self._validate_booster_matches_metadata(model, result.metadata)
        return model, result.metadata

    def _validate_metadata_contract(self, metadata: ModelArtifactMetadata) -> None:
        """Reject unsupported native-artifact contracts."""
        if metadata.model_file != MODEL_FILENAME:
            msg = f"model_file must be {MODEL_FILENAME!r} for native LightGBM artifacts"
            raise ValueError(msg)
        if metadata.model_format != "lightgbm_native":
            msg = "model_format must be 'lightgbm_native'"
            raise ValueError(msg)

    def _validate_model_hash(self, metadata: ModelArtifactMetadata) -> None:
        """Recompute identity hash from metadata fields and hard-fail on mismatch."""
        expected = compute_model_hash(
            features=metadata.features,
            hyperparameters=metadata.hyperparameters,
            universe=metadata.universe,
            start=metadata.start,
            end=metadata.end,
            frequency=metadata.frequency,
            seed=metadata.seed,
        )
        if metadata.model_hash != expected:
            msg = (
                f"model_hash mismatch: metadata has {metadata.model_hash!r}, "
                f"expected {expected!r}"
            )
            raise ValueError(msg)

    def _validate_booster_matches_metadata(
        self,
        model: Booster,
        metadata: ModelArtifactMetadata,
    ) -> None:
        """Ensure booster feature names/order match the metadata identity."""
        booster_features = tuple(model.feature_name())
        if booster_features != metadata.features:
            msg = (
                f"booster features {booster_features!r} do not match "
                f"metadata features {metadata.features!r}"
            )
            raise ValueError(msg)

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
