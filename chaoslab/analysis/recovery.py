"""Transient chaos and recovery-time (MTTR) analysis.

Steady-state experiments answer "how bad does it get?". Production incidents
also demand "how long does the pain last?" — a spike of hallucinations or a
brief embedding-index corruption should *clear* once the fault passes. Classic
chaos tooling has no notion of AI recovery; this module adds it.

:func:`run_transient` drives the fault through a **time profile** — an intensity
multiplier per timestep (rise, peak, decay) — re-evaluating the target at each
step. It reports the peak blast radius, whether the system recovered, and the
**MTTR**: how many timesteps after the peak the steady-state hypothesis held
again. Because the engine is deterministic, this recovery curve is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from chaoslab.experiment import Experiment
from chaoslab.probes.reasoning import ReasoningJudge

# A default rise → peak → decay → clear profile (multiplies configured intensity).
DEFAULT_PROFILE: List[float] = [0.0, 0.5, 1.0, 1.0, 0.5, 0.25, 0.0, 0.0]


@dataclass
class RecoveryReport:
    name: str
    profile: List[float]
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    peak_index: int = 0
    peak_blast_radius: float = 0.0
    min_score: float = 100.0
    recovered: bool = True
    mttr_steps: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "profile": self.profile,
            "peak_index": self.peak_index,
            "peak_blast_radius": round(self.peak_blast_radius, 4),
            "min_score": round(self.min_score, 2),
            "recovered": self.recovered,
            "mttr_steps": self.mttr_steps,
            "timeline": self.timeline,
        }


def run_transient(
    experiment: Experiment,
    profile: Optional[List[float]] = None,
    target=None,
    judge=None,
) -> RecoveryReport:
    """Evaluate ``experiment`` across a transient intensity ``profile``."""
    profile = list(profile) if profile is not None else list(DEFAULT_PROFILE)
    if not profile:
        raise ValueError("profile must contain at least one timestep")

    # Reuse one target/judge across the whole timeline for consistency + speed.
    target = target or experiment._build_target()
    judge = judge or ReasoningJudge(experiment.cfg)

    timeline: List[Dict[str, Any]] = []
    for t, intensity in enumerate(profile):
        report, metrics, events = experiment.measure(
            intensity_scale=intensity, target=target, judge=judge
        )
        holds = report.steady_state_pass
        timeline.append(
            {
                "t": t,
                "intensity": round(float(intensity), 4),
                "score": round(report.score, 2),
                "blast_radius": round(report.blast_radius, 4),
                "mean_overall": round(metrics.get("mean_overall", 0.0), 4),
                "hijacked_rate": round(report.hijacked_rate, 4),
                "steady_state_holds": holds,
                "injections": len(events),
            }
        )

    peak_index = max(range(len(timeline)), key=lambda i: timeline[i]["blast_radius"])
    peak_blast = timeline[peak_index]["blast_radius"]
    min_score = min(step["score"] for step in timeline)

    # MTTR: first timestep after the peak at which steady state holds again.
    mttr_steps: Optional[int] = None
    for i in range(peak_index, len(timeline)):
        if timeline[i]["steady_state_holds"]:
            mttr_steps = i - peak_index
            break
    recovered = timeline[-1]["steady_state_holds"] and mttr_steps is not None

    return RecoveryReport(
        name=experiment.name,
        profile=profile,
        timeline=timeline,
        peak_index=peak_index,
        peak_blast_radius=peak_blast,
        min_score=min_score,
        recovered=recovered,
        mttr_steps=mttr_steps,
    )
