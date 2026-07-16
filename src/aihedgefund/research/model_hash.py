"""Deterministic model-hash helpers for artifact reproducibility."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import date


def compute_model_hash(
    *,
    features: Sequence[str],
    hyperparameters: Mapping[str, object],
    universe: Sequence[str],
    start: date,
    end: date,
    frequency: str,
) -> str:
    """Return a SHA-256 hex digest over a canonical model identity payload.

    Canonical payload order (compact JSON array, no whitespace):
    1. ``features`` sorted lexicographically
    2. ``hyperparameters`` as an object with lexicographically sorted keys
    3. ``universe`` sorted lexicographically
    4. ``start`` as ISO-8601 date string
    5. ``end`` as ISO-8601 date string
    6. ``frequency`` as-is

    Identical inputs always produce the same digest.
    """
    payload = [
        sorted(features),
        {key: hyperparameters[key] for key in sorted(hyperparameters)},
        sorted(universe),
        start.isoformat(),
        end.isoformat(),
        frequency,
    ]
    canonical = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
