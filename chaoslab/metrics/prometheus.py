"""Prometheus metrics sink.

Exposes the blast-radius / robustness signals as Prometheus gauges. When a
pushgateway is configured (``CHAOSLAB_PUSHGATEWAY``) and ``prometheus_client``
is installed, metrics are pushed there; otherwise the sink keeps them in memory
and can render them in the Prometheus text exposition format — enough for tests,
local demos, and a ``/metrics`` endpoint on the chaos agent.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from chaoslab.config import Config
from chaoslab.scoring import RobustnessReport

_METRICS = {
    "chaoslab_robustness_score": "AI robustness score (0-100) for an experiment.",
    "chaoslab_blast_radius": "Fraction of eval cases impacted by injected faults.",
    "chaoslab_reasoning_degradation": "Drop in mean reasoning score vs baseline.",
    "chaoslab_hijacked_rate": "Fraction of answers hijacked by prompt injection.",
    "chaoslab_steady_state_violation": "1 if the steady-state hypothesis was violated.",
}


class PrometheusSink:
    def __init__(self, config: Optional[Config] = None):
        self.cfg = config or Config()
        # (metric, experiment) -> value
        self._samples: Dict[Tuple[str, str], float] = {}

    def record_experiment(self, experiment: str, report: RobustnessReport) -> None:
        self._samples[("chaoslab_robustness_score", experiment)] = report.score
        self._samples[("chaoslab_blast_radius", experiment)] = report.blast_radius
        self._samples[("chaoslab_reasoning_degradation", experiment)] = report.degradation
        self._samples[("chaoslab_hijacked_rate", experiment)] = report.hijacked_rate
        self._samples[("chaoslab_steady_state_violation", experiment)] = (
            0.0 if report.steady_state_pass else 1.0
        )
        if self.cfg.prometheus_pushgateway:
            self._push()

    def render_text(self) -> str:
        """Render metrics in Prometheus text exposition format."""
        lines: List[str] = []
        by_metric: Dict[str, List[Tuple[str, float]]] = {}
        for (metric, exp), value in self._samples.items():
            by_metric.setdefault(metric, []).append((exp, value))
        for metric, help_text in _METRICS.items():
            samples = by_metric.get(metric, [])
            if not samples:
                continue
            lines.append(f"# HELP {metric} {help_text}")
            lines.append(f"# TYPE {metric} gauge")
            for exp, value in sorted(samples):
                lines.append(f'{metric}{{experiment="{exp}"}} {value}')
        return "\n".join(lines) + ("\n" if lines else "")

    def _push(self) -> None:  # pragma: no cover - requires network + dependency
        try:
            from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
        except Exception:
            return
        registry = CollectorRegistry()
        gauges: Dict[str, "Gauge"] = {}
        for (metric, exp), value in self._samples.items():
            if metric not in gauges:
                gauges[metric] = Gauge(
                    metric, _METRICS[metric], ["experiment"], registry=registry
                )
            gauges[metric].labels(experiment=exp).set(value)
        try:
            push_to_gateway(
                self.cfg.prometheus_pushgateway, job=self.cfg.prometheus_job, registry=registry
            )
        except Exception:
            pass
