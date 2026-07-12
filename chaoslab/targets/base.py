"""Target-system abstractions.

A :class:`Target` is the AI system under test. It exposes a set of evaluation
cases and a :meth:`run` method that executes one case through a staged pipeline.
At each pipeline stage the target calls the injected ``perturb`` hook, which is
where chaos enters — with no active experiment the hook is the identity function,
so the same code path serves both baseline and chaos runs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from chaoslab.injectors.base import InjectionEvent, Stage

# perturb(stage, payload, case_id) -> possibly-perturbed payload (may raise).
Perturb = Callable[[Stage, Any, str], Any]


def identity_perturb(stage: Stage, payload: Any, case_id: str) -> Any:
    return payload


@dataclass
class EvalCase:
    """A single evaluation case for a target."""

    id: str
    question: str
    expected: List[str]  # keywords that a correct answer should contain
    category: str = "qa"
    sensor_input: Optional[Dict[str, Any]] = None
    forbidden: List[str] = field(default_factory=list)  # tokens a safe answer must NOT contain


@dataclass
class Response:
    """The outcome of running one case."""

    case_id: str
    answer: str
    latency_ms: float = 0.0
    retrieved: List[Dict[str, Any]] = field(default_factory=list)
    events: List[InjectionEvent] = field(default_factory=list)
    error: Optional[str] = None


class Target(ABC):
    """Base class for systems under test."""

    name: str = "target"

    @abstractmethod
    def eval_cases(self) -> List[EvalCase]:
        """Return the evaluation suite used to measure steady state."""

    @abstractmethod
    def run(self, case: EvalCase, perturb: Perturb = identity_perturb) -> Response:
        """Execute one case, calling ``perturb`` at each pipeline stage."""
