import random

import pytest

from chaoslab.injectors import (
    CANARY,
    EmbeddingDriftInjector,
    HallucinationInjector,
    PromptInjectionInjector,
    SensorNoiseInjector,
    Stage,
)
from chaoslab.injectors.infra import ErrorInjector, FaultInjectionError, LatencyInjector


def rng(seed="t"):
    return random.Random(seed)


def test_intensity_bounds():
    with pytest.raises(ValueError):
        HallucinationInjector(intensity=1.5)
    with pytest.raises(ValueError):
        HallucinationInjector(intensity=-0.1)


def test_injectors_are_deterministic():
    inj = EmbeddingDriftInjector(intensity=0.8)
    vec = [0.1, 0.2, 0.3, 0.4]
    out1, ev1 = inj.apply(list(vec), rng("same"))
    out2, ev2 = inj.apply(list(vec), rng("same"))
    assert out1 == out2
    assert ev1.meta["cosine_similarity"] == ev2.meta["cosine_similarity"]


def test_hallucination_appends_fabrication():
    inj = HallucinationInjector(intensity=1.0)
    out, ev = inj.apply("The answer is four.", rng("h"))
    assert ev is not None
    assert ev.stage == Stage.LLM_OUTPUT.value
    assert len(out) > len("The answer is four.")
    assert ev.meta["fabricated_claims"]


def test_hallucination_ignores_empty():
    inj = HallucinationInjector(intensity=1.0)
    out, ev = inj.apply("", rng("h"))
    assert ev is None and out == ""


def test_embedding_drift_reduces_cosine():
    inj = EmbeddingDriftInjector(intensity=1.0)
    vec = [1.0, 0.0, 0.0, 0.0, 0.0]
    out, ev = inj.apply(vec, rng("d"))
    assert ev.meta["cosine_similarity"] < 1.0
    assert len(out) == len(vec)


def test_embedding_drift_ignores_non_vector():
    inj = EmbeddingDriftInjector(intensity=0.5)
    out, ev = inj.apply("not a vector", rng("d"))
    assert ev is None


def test_prompt_injection_prompt_stage_appends_attack():
    inj = PromptInjectionInjector(intensity=1.0, target="prompt")
    assert inj.stage == Stage.PROMPT
    out, ev = inj.apply("What is 2+2?", rng("p"))
    assert CANARY in ev.meta["canary"]
    assert CANARY in out or ev.meta["attack"]


def test_prompt_injection_retrieval_stage_poisons_doc():
    inj = PromptInjectionInjector(intensity=1.0, target="retrieval")
    assert inj.stage == Stage.RETRIEVAL
    docs = [{"id": "a", "text": "hello"}, {"id": "b", "text": "world"}]
    out, ev = inj.apply(docs, rng("p"))
    assert any(d.get("poisoned") for d in out)


def test_sensor_noise_corrupts_channels():
    inj = SensorNoiseInjector(intensity=1.0)
    readings = {"front_distance_m": 2.0, "lidar": [1.0, 2.0, 3.0]}
    out, ev = inj.apply(readings, rng("s"))
    assert ev.meta["values_affected"] >= 4
    assert out != readings


def test_latency_injector_reports_added_ms():
    inj = LatencyInjector(intensity=1.0, base_ms=1000)
    out, ev = inj.apply("payload", rng("l"))
    assert out == "payload"
    assert ev.meta["added_ms"] > 0


def test_error_injector_raises():
    inj = ErrorInjector(intensity=1.0)
    with pytest.raises(FaultInjectionError):
        inj.apply(["doc"], rng("e"))
