"""The chaos experiment engine.

Loads a declarative experiment spec, runs the target's evaluation suite twice —
once as a clean **baseline** and once with faults active — scores both with the
reasoning judge, checks the steady-state hypothesis, computes the robustness
report, and fans the results out to the metrics sinks.

Safety / "incident-free" design: chaos is applied purely as in-process
middleware around a sandboxed target, never to real production traffic, and a
configurable ``max_blast_radius`` ceiling flags runs that exceeded their agreed
blast radius so they can be caught in review.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from chaoslab.config import Config, load_config
from chaoslab.injectors.base import InjectionEvent, Stage
from chaoslab.injectors.infra import FaultInjectionError
from chaoslab.metrics.mlflow_logger import MLflowLogger
from chaoslab.metrics.prometheus import PrometheusSink
from chaoslab.probes.reasoning import ReasoningJudge, ReasoningScore
from chaoslab.probes.steady_state import SteadyState, percentile
from chaoslab.registry import build_injector, build_target
from chaoslab.scoring import RobustnessReport, compute_robustness
from chaoslab.targets.base import Response, Target, identity_perturb


class ChaosRuntime:
    """Applies stage injectors as a ``perturb`` hook and records what it did."""

    def __init__(self, injectors, base_seed: int):
        self._by_stage = defaultdict(list)
        for inj in injectors:
            self._by_stage[inj.stage].append(inj)
        self.base_seed = base_seed
        self.events: List[InjectionEvent] = []
        self.injected_latency_ms: Dict[str, float] = defaultdict(float)

    def perturb(self, stage: Stage, payload: Any, case_id: str) -> Any:
        rng = random.Random(f"{self.base_seed}:{stage.value}:{case_id}")
        for inj in self._by_stage.get(stage, []):
            payload, event = inj.apply(payload, rng, case_id)
            if event is not None:
                self.events.append(event)
                self.injected_latency_ms[case_id] += float(event.meta.get("added_ms", 0.0))
        return payload


@dataclass
class ExperimentResult:
    name: str
    report: RobustnessReport
    metrics: Dict[str, float]
    events: List[InjectionEvent] = field(default_factory=list)
    run_ref: Optional[str] = None
    aborted: bool = False
    prometheus_text: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "report": self.report.as_dict(),
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "injections": len(self.events),
            "run_ref": self.run_ref,
            "aborted": self.aborted,
        }


class Experiment:
    """A single chaos experiment defined declaratively."""

    def __init__(
        self,
        name: str,
        target_spec: Dict[str, Any],
        injector_specs: List[Dict[str, Any]],
        steady_state: SteadyState,
        description: str = "",
        runs: int = 1,
        seed: Optional[int] = None,
        config: Optional[Config] = None,
        recovery_profile: Optional[List[float]] = None,
    ):
        if not isinstance(injector_specs, list):
            raise ValueError("'injectors' must be a list of injector specs")
        for spec in injector_specs:
            if "name" not in spec:
                raise ValueError(f"injector spec missing required 'name': {spec!r}")
        self.name = name
        self.description = description
        self.target_spec = target_spec
        self.injector_specs = injector_specs
        self.steady_state = steady_state
        self.runs = max(1, int(runs))
        self.cfg = config or load_config()
        self.seed = self.cfg.seed if seed is None else int(seed)
        # Optional transient intensity profile for recovery/MTTR analysis.
        self.recovery_profile = [float(x) for x in recovery_profile] if recovery_profile else None

    # -- loading ------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Dict[str, Any], config: Optional[Config] = None) -> "Experiment":
        if "name" not in data:
            raise ValueError("experiment spec missing required field 'name'")
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            target_spec=data.get("target", {"name": "rag_agent"}),
            injector_specs=data.get("injectors", []),
            steady_state=SteadyState.from_dict(data.get("steady_state")),
            runs=data.get("runs", 1),
            seed=data.get("seed"),
            config=config,
            recovery_profile=data.get("recovery_profile"),
        )

    @classmethod
    def from_yaml(cls, path: str, config: Optional[Config] = None) -> "Experiment":
        with open(path) as fh:
            data = yaml.safe_load(fh)
        return cls.from_dict(data, config=config)

    # -- helpers ------------------------------------------------------------
    def _build_target(self) -> Target:
        spec = self.target_spec or {"name": "rag_agent"}
        return build_target(spec.get("name", "rag_agent"), config=self.cfg, **spec.get("params", {}))

    def _build_injectors(self, scale: float = 1.0):
        """Instantiate the spec's injectors, scaling every intensity by ``scale``.

        ``scale`` (used by the recovery and breaking-point analyses) multiplies
        each injector's configured intensity and clamps to ``[0, 1]``.
        """
        injectors = []
        for spec in self.injector_specs:
            params = dict(spec.get("params", {}))
            base = float(spec.get("intensity", 0.5))
            params["intensity"] = max(0.0, min(1.0, base * scale))
            injectors.append(build_injector(spec["name"], **params))
        return injectors

    def _score_all(self, target: Target, judge: ReasoningJudge, perturb, runtime, run_idx: int):
        scores: List[ReasoningScore] = []
        latencies: List[float] = []
        for case in target.eval_cases():
            try:
                resp = target.run(case, perturb)
            except FaultInjectionError as exc:
                resp = Response(case_id=case.id, answer="", error=str(exc))
            # account for injected latency, if any
            if runtime is not None:
                resp.latency_ms += runtime.injected_latency_ms.get(case.id, 0.0)
            latencies.append(resp.latency_ms)
            score = judge.score(case, resp)
            # namespace case id per run so repeated runs align independently
            score.case_id = f"{case.id}#r{run_idx}"
            scores.append(score)
        return scores, latencies

    def _collect(self, target, judge, intensity_scale: float = 1.0, runs: Optional[int] = None):
        """Run baseline + chaos (at ``intensity_scale``) over ``runs`` repetitions."""
        runs = self.runs if runs is None else runs
        baseline: List[ReasoningScore] = []
        chaos: List[ReasoningScore] = []
        latencies: List[float] = []
        events: List[InjectionEvent] = []
        for r in range(runs):
            b_scores, _ = self._score_all(target, judge, identity_perturb, None, r)
            baseline.extend(b_scores)
            if intensity_scale <= 0.0:
                # Recovery / baseline phase: chaos removed, identical code path.
                c_scores, c_lat = self._score_all(target, judge, identity_perturb, None, r)
            else:
                runtime = ChaosRuntime(self._build_injectors(intensity_scale), base_seed=self.seed + r)
                c_scores, c_lat = self._score_all(target, judge, runtime.perturb, runtime, r)
                events.extend(runtime.events)
            chaos.extend(c_scores)
            latencies.extend(c_lat)
        return baseline, chaos, latencies, events

    def measure(
        self,
        intensity_scale: float = 1.0,
        target: Optional[Target] = None,
        judge: Optional[ReasoningJudge] = None,
        runs: Optional[int] = None,
    ):
        """Evaluate the target at a given fault intensity vs a clean baseline.

        Returns ``(report, metrics, events)``. Reusable building block for the
        recovery and breaking-point analyses; ``run`` is this plus the sinks.
        """
        target = target or self._build_target()
        judge = judge or ReasoningJudge(self.cfg)
        baseline, chaos, latencies, events = self._collect(target, judge, intensity_scale, runs)
        metrics = self._aggregate_metrics(chaos, latencies)
        violations = self.steady_state.evaluate(metrics)
        report = compute_robustness(baseline, chaos, violations)
        return report, metrics, events

    # -- execution ----------------------------------------------------------
    def run(self, sinks: bool = True) -> ExperimentResult:
        report, metrics, events = self.measure()

        aborted = self.cfg.strict_incident_free and report.blast_radius > self.cfg.max_blast_radius

        prom = PrometheusSink(self.cfg)
        run_ref = None
        if sinks:
            prom.record_experiment(self.name, report)
            run_ref = MLflowLogger(self.cfg).log_experiment(
                self.name,
                params={
                    "target": self.target_spec.get("name", "rag_agent"),
                    "injectors": ",".join(s["name"] for s in self.injector_specs),
                    "seed": self.seed,
                    "runs": self.runs,
                    "llm_provider": self.cfg.llm_provider,
                },
                report=report,
                extra_metrics={"injections": float(len(events))},
            )

        return ExperimentResult(
            name=self.name,
            report=report,
            metrics=metrics,
            events=events,
            run_ref=run_ref,
            aborted=aborted,
            prometheus_text=prom.render_text(),
        )

    @staticmethod
    def _aggregate_metrics(scores: List[ReasoningScore], latencies: List[float]) -> Dict[str, float]:
        n = len(scores) or 1
        return {
            "mean_correctness": sum(s.correctness for s in scores) / n,
            "mean_overall": sum(s.overall for s in scores) / n,
            "mean_groundedness": sum(s.groundedness for s in scores) / n,
            "hijacked_rate": sum(1.0 for s in scores if s.hijacked) / n,
            "p95_latency_ms": percentile(latencies, 0.95),
            "cases": float(n),
        }
