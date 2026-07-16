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
    seed: int,
) -> str:
    """Return a SHA-256 hex digest over a canonical model identity payload.

    Canonical payload order (compact JSON array, no whitespace):
    1. ``features`` in the given order (training-relevant; not sorted)
    2. ``hyperparameters`` as an object with lexicographically sorted keys
    3. ``universe`` sorted lexicographically
    4. ``start`` as ISO-8601 date string
    5. ``end`` as ISO-8601 date string
    6. ``frequency`` as-is
    7. ``seed`` as an integer

    Identical inputs always produce the same digest. Different feature order
    produces a different digest.
    """
    payload = [
        list(features),
        {key: hyperparameters[key] for key in sorted(hyperparameters)},
        sorted(universe),
        start.isoformat(),
        end.isoformat(),
        frequency,
        seed,
    ]
    canonical = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
