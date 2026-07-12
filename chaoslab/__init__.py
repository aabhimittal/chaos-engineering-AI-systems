"""chaoslab — Chaos Engineering framework for AI/LLM systems.

An AI-native chaos lab: inject *AI-specific* failure modes (hallucination
spikes, embedding drift, prompt injection, robotics sensor noise) into a target
AI system, measure the blast radius, and produce a quantified AI Robustness
Score.

The framework is deliberately dependency-light and runs fully offline with a
deterministic mock LLM, so experiments are reproducible and incident-free.
Real backends (Claude API, MLflow server, Prometheus pushgateway) are enabled
transparently via environment/config.
"""

from chaoslab.experiment import Experiment, ExperimentResult
from chaoslab.scoring import RobustnessReport, compute_robustness

__version__ = "0.1.0"

__all__ = [
    "Experiment",
    "ExperimentResult",
    "RobustnessReport",
    "compute_robustness",
    "__version__",
]
