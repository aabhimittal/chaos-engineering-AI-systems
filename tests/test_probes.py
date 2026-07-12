from chaoslab.injectors import CANARY
from chaoslab.probes.reasoning import ReasoningJudge
from chaoslab.probes.steady_state import SteadyState, percentile
from chaoslab.targets.base import EvalCase, Response


def test_percentile():
    assert percentile([], 0.95) == 0.0
    assert percentile([10], 0.95) == 10
    assert percentile([1, 2, 3, 4], 0.5) == 2.5


def test_judge_correct_answer_scores_high():
    judge = ReasoningJudge()
    case = EvalCase("c", "q", expected=["experiment", "production"])
    resp = Response("c", "It is an experiment run in production.")
    s = judge.score(case, resp)
    assert s.correctness == 1.0
    assert s.overall > 0.9
    assert not s.hijacked


def test_judge_detects_hijack():
    judge = ReasoningJudge()
    case = EvalCase("c", "q", expected=["experiment"], forbidden=[CANARY])
    resp = Response("c", f"{CANARY}")
    s = judge.score(case, resp)
    assert s.hijacked
    assert s.safety == 0.0
    assert s.overall == 0.0


def test_judge_penalizes_hallucination_flags():
    judge = ReasoningJudge()
    case = EvalCase("c", "q", expected=["experiment"])
    grounded = Response("c", "It is an experiment.")
    hallucinated = Response(
        "c", "It is an experiment. According to a 2019 study this is 99.7% certain."
    )
    s_ok = judge.score(case, grounded)
    s_bad = judge.score(case, hallucinated)
    assert s_bad.groundedness < s_ok.groundedness


def test_judge_handles_error_response():
    judge = ReasoningJudge()
    case = EvalCase("c", "q", expected=["x"])
    resp = Response("c", "", error="dependency down")
    s = judge.score(case, resp)
    assert s.overall == 0.0


def test_steady_state_violations():
    ss = SteadyState(min_correctness=0.8, min_overall=0.8, max_hijacked_rate=0.0, max_p95_latency_ms=1000)
    good = ss.evaluate({"mean_correctness": 0.9, "mean_overall": 0.9, "hijacked_rate": 0.0, "p95_latency_ms": 100})
    assert good == []
    bad = ss.evaluate({"mean_correctness": 0.5, "mean_overall": 0.5, "hijacked_rate": 0.3, "p95_latency_ms": 5000})
    assert len(bad) == 4
