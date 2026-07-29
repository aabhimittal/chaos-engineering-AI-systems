"""Adaptive breaking-point search — find an AI system's resilience threshold.

Instead of asking "does the system survive intensity 0.85?", this asks the more
useful question: **"at what fault intensity does it break?"** It sweeps the
injector intensity and finds the smallest multiplier at which the steady-state
hypothesis first fails — the *resilience threshold*. A higher threshold means a
more robust system, giving you a single tunable number to track across model
versions or prompt-hardening iterations.

The search is a coarse grid scan (robust to the mild non-monotonicity of
probabilistic faults) to bracket the first failure, followed by a bisection to
refine the threshold to a tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from chaoslab.experiment import Experiment
from chaoslab.probes.reasoning import ReasoningJudge


@dataclass
class BreakingPointReport:
    name: str
    resilience_threshold: Optional[float]  # intensity scale where it first breaks
    holds_up_to: float                     # highest scale still holding
    samples: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if self.resilience_threshold is None:
            return f"resilient across the tested range (holds up to {self.holds_up_to:.2f})"
        return f"breaks at intensity {self.resilience_threshold:.2f}"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "resilience_threshold": (
                None if self.resilience_threshold is None else round(self.resilience_threshold, 3)
            ),
            "holds_up_to": round(self.holds_up_to, 3),
            "verdict": self.verdict,
            "samples": self.samples,
        }


def find_breaking_point(
    experiment: Experiment,
    lo: float = 0.0,
    hi: float = 1.0,
    grid: int = 5,
    tol: float = 0.03,
    target=None,
    judge=None,
) -> BreakingPointReport:
    """Find the smallest intensity scale at which steady state is violated."""
    if lo < 0 or hi <= lo:
        raise ValueError("require 0 <= lo < hi")
    target = target or experiment._build_target()
    judge = judge or ReasoningJudge(experiment.cfg)

    samples: List[Dict[str, Any]] = []

    def holds_at(scale: float) -> bool:
        report, _, _ = experiment.measure(intensity_scale=scale, target=target, judge=judge)
        samples.append(
            {
                "intensity": round(scale, 3),
                "score": round(report.score, 2),
                "blast_radius": round(report.blast_radius, 4),
                "holds": report.steady_state_pass,
            }
        )
        return report.steady_state_pass

    # 1) Coarse scan to bracket the first failure.
    step = (hi - lo) / grid
    prev = lo
    bracket = None
    holds_up_to = lo
    for i in range(grid + 1):
        scale = lo + i * step
        if holds_at(scale):
            holds_up_to = scale
            prev = scale
        else:
            bracket = (prev, scale)
            break

    if bracket is None:
        return BreakingPointReport(experiment.name, None, holds_up_to, samples)

    # 2) Bisection to refine the threshold between last-holding and first-failing.
    low, high = bracket
    while high - low > tol:
        mid = 0.5 * (low + high)
        if holds_at(mid):
            low = mid
            holds_up_to = max(holds_up_to, mid)
        else:
            high = mid

    return BreakingPointReport(experiment.name, high, holds_up_to, samples)
