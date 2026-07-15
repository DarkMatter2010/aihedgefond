"""Research infrastructure adapters."""

from aihedgefund.research.adapters.filesystem import (
    FilesystemModelArtifactAdapter,
    compute_model_hash,
)

__all__ = ["FilesystemModelArtifactAdapter", "compute_model_hash"]
