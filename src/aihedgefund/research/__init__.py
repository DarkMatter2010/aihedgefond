"""Research domain; concrete adapters live in ``research.adapters``."""

from aihedgefund.research.adapters import FilesystemModelArtifactAdapter
from aihedgefund.research.model_hash import compute_model_hash

__all__ = ["FilesystemModelArtifactAdapter", "compute_model_hash"]
