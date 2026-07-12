"""Classic infrastructure fault injectors, for comparison and blended runs.

AI systems still run on infra, so the lab includes the traditional faults too:

* :class:`LatencyInjector` — adds simulated latency to a stage (returned as
  metadata; the engine accounts for it in the latency SLO check).
* :class:`ErrorInjector` — probabilistically raises a
  :class:`FaultInjectionError` to model a dependency (vector store, model
  endpoint) returning an error, exercising the target's fallback path.
"""

from __future__ import annotations

import random
from typing import Any

from chaoslab.injectors.base import Injector, Stage


class FaultInjectionError(RuntimeError):
    """Raised by :class:`ErrorInjector` to simulate a dependency failure."""


class LatencyInjector(Injector):
    """Attach simulated added-latency (ms) to a stage payload's context."""

    stage = Stage.LLM_OUTPUT
    name = "latency"

    def __init__(self, intensity: float = 0.5, base_ms: float = 1500.0, **params: Any):
        super().__init__(intensity=intensity, **params)
        self.base_ms = base_ms

    def _apply(self, payload: Any, rng: random.Random):
        added = self.base_ms * self.intensity * (0.5 + rng.random())
        # Payload is unchanged; latency is reported via metadata + magnitude.
        return (payload, f"added {added:.0f}ms latency", self.intensity, {"added_ms": added})


class ErrorInjector(Injector):
    """Probabilistically raise to simulate a failing dependency."""

    stage = Stage.RETRIEVAL
    name = "dependency_error"

    def _apply(self, payload: Any, rng: random.Random):
        if rng.random() < self.intensity:
            raise FaultInjectionError(
                f"injected dependency error at stage={self.stage.value}"
            )
        return None
