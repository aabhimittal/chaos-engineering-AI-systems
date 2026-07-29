from chaoslab.analysis import find_breaking_point, run_transient
from chaoslab.experiment import Experiment
from chaoslab.probes.steady_state import SteadyState


def _exp(name, injectors, ss, runs=3):
    return Experiment(
        name=name,
        target_spec={"name": "rag_agent"},
        injector_specs=injectors,
        steady_state=ss,
        runs=runs,
    )


def test_run_transient_produces_recovery_curve():
    exp = _exp(
        "transient",
        [{"name": "hallucination_spike", "intensity": 1.0}],
        SteadyState(min_correctness=0.7, min_overall=0.93, max_hijacked_rate=0.0),
    )
    report = run_transient(exp, profile=[0.0, 0.4, 0.8, 1.0, 0.6, 0.3, 0.0])
    assert len(report.timeline) == 7
    # blast radius peaks in the middle (high intensity), not at the clean ends
    assert report.timeline[0]["blast_radius"] == 0.0
    assert report.peak_blast_radius > report.timeline[0]["blast_radius"]
    # violated at the peak, then recovered by the end
    assert report.mttr_steps is not None
    assert report.recovered is True


def test_run_transient_is_deterministic():
    exp = _exp(
        "t", [{"name": "embedding_drift", "intensity": 0.8}],
        SteadyState(min_correctness=0.55, min_overall=0.55),
    )
    a = run_transient(exp, profile=[0.0, 0.5, 1.0, 0.5, 0.0])
    b = run_transient(exp, profile=[0.0, 0.5, 1.0, 0.5, 0.0])
    assert [s["blast_radius"] for s in a.timeline] == [s["blast_radius"] for s in b.timeline]


def test_run_transient_rejects_empty_profile():
    exp = _exp("t", [{"name": "hallucination_spike", "intensity": 1.0}], SteadyState())
    try:
        run_transient(exp, profile=[])
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty profile")


def test_breaking_point_finds_low_threshold_for_prompt_injection():
    exp = _exp(
        "inject",
        [{"name": "prompt_injection", "intensity": 1.0, "params": {"target": "retrieval"}}],
        SteadyState(min_correctness=0.5, min_overall=0.5, max_hijacked_rate=0.0),
    )
    report = find_breaking_point(exp)
    # prompt injection is catastrophic — it should break at a low intensity
    assert report.resilience_threshold is not None
    assert report.resilience_threshold < 0.5


def test_breaking_point_resilient_experiment_has_no_threshold():
    # A permissive steady state that even full-intensity drift satisfies.
    exp = _exp(
        "resilient",
        [{"name": "embedding_drift", "intensity": 1.0}],
        SteadyState(min_correctness=0.0, min_overall=0.0, max_hijacked_rate=1.0, max_p95_latency_ms=1e9),
    )
    report = find_breaking_point(exp)
    assert report.resilience_threshold is None
    assert report.holds_up_to >= 1.0
