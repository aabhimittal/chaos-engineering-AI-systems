"""Plugin registry mapping string names -> injector / target classes.

Experiment YAML files reference injectors and targets by name; the registry
resolves those names to classes so the framework is extensible: register a new
injector and it becomes usable from a spec with no engine changes.
"""

from __future__ import annotations

from typing import Dict, Type

from chaoslab.injectors.base import Injector
from chaoslab.injectors.context_window import (
    ContextTruncationInjector,
    OutputTruncationInjector,
)
from chaoslab.injectors.embedding_drift import EmbeddingDriftInjector
from chaoslab.injectors.hallucination import HallucinationInjector
from chaoslab.injectors.infra import ErrorInjector, LatencyInjector
from chaoslab.injectors.prompt_injection import PromptInjectionInjector
from chaoslab.injectors.sensor_noise import SensorNoiseInjector
from chaoslab.injectors.unicode_noise import UnicodePerturbationInjector

INJECTORS: Dict[str, Type[Injector]] = {}


def register_injector(cls: Type[Injector]) -> Type[Injector]:
    INJECTORS[cls.name] = cls
    return cls


for _cls in (
    HallucinationInjector,
    EmbeddingDriftInjector,
    PromptInjectionInjector,
    SensorNoiseInjector,
    ContextTruncationInjector,
    OutputTruncationInjector,
    UnicodePerturbationInjector,
    LatencyInjector,
    ErrorInjector,
):
    register_injector(_cls)


def build_injector(name: str, **params) -> Injector:
    if name not in INJECTORS:
        raise KeyError(
            f"unknown injector '{name}'. Registered: {sorted(INJECTORS)}"
        )
    return INJECTORS[name](**params)


# Targets are registered lazily to avoid import cycles.
def build_target(name: str, **params):
    from chaoslab.targets.rag_agent import RagAgentTarget

    targets = {"rag_agent": RagAgentTarget}
    if name not in targets:
        raise KeyError(f"unknown target '{name}'. Registered: {sorted(targets)}")
    return targets[name](**params)
