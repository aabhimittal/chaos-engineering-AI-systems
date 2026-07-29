"""Embedding-drift injector.

Simulates a stale/incompatible embedding model or a corrupted vector index by
perturbing dense vectors at the ``EMBEDDING`` stage. Two mechanisms, blended by
``intensity``:

* **Gaussian noise** — small random jitter added to every dimension, modelling
  numerical drift between embedding-model versions.
* **Rotation/scale drift** — a systematic bias vector that shifts embeddings in
  a consistent direction, modelling a genuine distribution shift.

Downstream, this degrades retrieval quality (the RAG target's similarity search
returns worse documents), which is exactly what embedding drift does in prod.
"""

from __future__ import annotations

import math
import random
from typing import Any, List

from chaoslab.injectors.base import Injector, Stage


def _as_vector(payload: Any) -> List[float] | None:
    if isinstance(payload, (list, tuple)) and payload and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) for x in payload
    ):
        # Coerce non-finite inputs (NaN/Inf from a broken embedding service) to 0
        # so downstream retrieval degrades gracefully instead of producing NaN.
        return [float(x) if math.isfinite(float(x)) else 0.0 for x in payload]
    return None


class EmbeddingDriftInjector(Injector):
    """Add noise + systematic bias to embedding vectors."""

    stage = Stage.EMBEDDING
    name = "embedding_drift"

    def _apply(self, payload: Any, rng: random.Random):
        vec = _as_vector(payload)
        if vec is None:
            return None

        sigma = 0.15 * self.intensity  # noise scale
        bias = 0.30 * self.intensity  # systematic drift scale
        dims = len(vec)

        # A stable-ish bias direction (seeded by rng) so drift is directional.
        bias_dir = [rng.gauss(0, 1) for _ in range(dims)]
        norm = math.sqrt(sum(b * b for b in bias_dir)) or 1.0
        bias_dir = [b / norm for b in bias_dir]

        drifted = [
            v + rng.gauss(0, sigma) + bias * bias_dir[i] for i, v in enumerate(vec)
        ]

        # Report cosine similarity between original and drifted vector.
        dot = sum(a * b for a, b in zip(vec, drifted))
        na = math.sqrt(sum(a * a for a in vec)) or 1.0
        nb = math.sqrt(sum(b * b for b in drifted)) or 1.0
        cos = dot / (na * nb)

        return (
            drifted,
            f"drifted embedding, cosine={cos:.3f}",
            round(1.0 - cos, 4),
            {"cosine_similarity": round(cos, 4), "sigma": sigma, "bias": bias},
        )
