"""Industrial edge cases: malformed inputs, degenerate values, concurrency."""

import math
import random
import threading

import pytest

from chaoslab.experiment import Experiment
from chaoslab.injectors import EmbeddingDriftInjector
from chaoslab.probes.reasoning import ReasoningJudge, ReasoningScore
from chaoslab.probes.steady_state import SteadyState, percentile
from chaoslab.registry import build_injector, build_target
from chaoslab.scoring import compute_robustness
from chaoslab.targets.base import EvalCase, identity_perturb
from chaoslab.targets.rag_agent import RagAgentTarget, _cosine


def rng(seed="e"):
    return random.Random(seed)


# -- degenerate numeric inputs ---------------------------------------------
def test_embedding_drift_handles_nan_and_inf():
    inj = EmbeddingDriftInjector(intensity=0.9)
    out, ev = inj.apply([float("nan"), float("inf"), 1.0, -2.0], rng())
    assert all(math.isfinite(x) for x in out)
    assert math.isfinite(ev.meta["cosine_similarity"])


def test_cosine_non_finite_is_worst_not_crash():
    assert _cosine([float("nan"), 1.0], [1.0, 1.0]) == -1.0


def test_cosine_mismatched_dims():
    # shorter prefix compared, no crash
    assert -1.0 <= _cosine([1.0, 0.0, 0.0], [1.0, 0.0]) <= 1.0


def test_embedding_drift_rejects_bool_list():
    # bools are ints in Python; they are not embeddings
    inj = EmbeddingDriftInjector(intensity=0.5)
    out, ev = inj.apply([True, False, True], rng())
    assert ev is None


# -- empty / starved retrieval ---------------------------------------------
def test_context_starvation_yields_no_answer_without_crashing():
    t = RagAgentTarget()
    inj = build_injector("context_truncation", intensity=1.0)

    def perturb(stage, payload, case_id):
        if stage.value == "retrieval":
            out, _ = inj.apply(payload, rng(case_id))
            return out
        return payload

    case = t.eval_cases()[0]
    resp = t.run(case, perturb)
    assert "don't have enough information" in resp.answer.lower()
    score = ReasoningJudge().score(case, resp)
    assert score.overall < 0.5  # degraded, but scored cleanly


# -- extreme / unusual text ------------------------------------------------
def test_huge_prompt_does_not_crash():
    t = RagAgentTarget()
    big = EvalCase("big", "chaos " * 5000, expected=["experiment"])
    resp = t.run(big, identity_perturb)
    assert isinstance(resp.answer, str)


def test_unicode_question_flows_through_pipeline():
    t = RagAgentTarget()
    case = EvalCase("uni", "¿Qué es chaos engineering? 混沌工程 🌀", expected=["experiment"])
    resp = t.run(case, identity_perturb)
    assert isinstance(resp.answer, str)


# -- malformed specs -------------------------------------------------------
def test_experiment_missing_name_raises():
    with pytest.raises(ValueError):
        Experiment.from_dict({"injectors": []})


def test_injector_spec_missing_name_raises():
    with pytest.raises(ValueError):
        Experiment.from_dict({"name": "x", "injectors": [{"intensity": 0.5}]})


def test_unknown_injector_and_target_raise_keyerror():
    with pytest.raises(KeyError):
        build_injector("does_not_exist")
    with pytest.raises(KeyError):
        build_target("nope")


# -- degenerate scoring ----------------------------------------------------
def test_compute_robustness_empty_inputs():
    report = compute_robustness([], [], violations=[])
    assert 0.0 <= report.score <= 100.0
    assert report.blast_radius == 0.0


def test_compute_robustness_all_hijacked():
    base = [ReasoningScore(f"c{i}", 1.0, 1.0, 1.0, 1.0) for i in range(5)]
    chaos = [ReasoningScore(f"c{i}", 0.0, 0.0, 0.0, 0.0, hijacked=True) for i in range(5)]
    report = compute_robustness(base, chaos, violations=["hijacked"])
    assert report.blast_radius == 1.0
    assert report.hijacked_rate == 1.0
    assert report.grade == "F"


def test_percentile_single_sample():
    assert percentile([42.0], 0.95) == 42.0


# -- intensity bounds ------------------------------------------------------
def test_intensity_extremes_do_not_crash():
    for name in ("hallucination_spike", "embedding_drift", "sensor_noise",
                 "context_truncation", "output_truncation", "unicode_perturbation"):
        for intensity in (0.0, 1.0):
            build_injector(name, intensity=intensity)  # constructs & validates
    with pytest.raises(ValueError):
        build_injector("hallucination_spike", intensity=1.5)


# -- determinism under concurrency -----------------------------------------
def test_experiment_is_thread_safe_and_deterministic():
    def make():
        return Experiment(
            name="c",
            target_spec={"name": "rag_agent"},
            injector_specs=[{"name": "embedding_drift", "intensity": 0.7}],
            steady_state=SteadyState(min_correctness=0.5, min_overall=0.5),
            runs=2,
        )

    results = {}

    def worker(i):
        results[i] = make().run(sinks=False).report.score

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    scores = list(results.values())
    assert len(scores) == 4
    assert all(s == scores[0] for s in scores)  # identical => deterministic + isolated
