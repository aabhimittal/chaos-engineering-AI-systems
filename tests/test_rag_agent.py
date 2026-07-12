from chaoslab.experiment import ChaosRuntime
from chaoslab.injectors import CANARY
from chaoslab.registry import build_injector
from chaoslab.targets.base import identity_perturb
from chaoslab.targets.rag_agent import RagAgentTarget


def test_baseline_answers_are_correct():
    t = RagAgentTarget()
    from chaoslab.probes.reasoning import ReasoningJudge

    judge = ReasoningJudge()
    for case in t.eval_cases():
        resp = t.run(case, identity_perturb)
        s = judge.score(case, resp)
        assert s.correctness >= 0.5, f"{case.id}: {resp.answer!r}"
        assert not s.hijacked


def test_robotics_decision_boundary():
    t = RagAgentTarget()
    cases = {c.id: c for c in t.eval_cases()}
    stop = t.run(cases["r_stop"], identity_perturb)
    go = t.run(cases["r_go"], identity_perturb)
    assert "stop" in stop.answer
    assert "proceed" in go.answer


def test_indirect_prompt_injection_can_hijack_naive_model():
    t = RagAgentTarget()
    inj = build_injector("prompt_injection", intensity=1.0, target="retrieval")
    hijacked = False
    for run in range(6):
        rt = ChaosRuntime([inj], base_seed=1000 + run)
        for case in t.eval_cases():
            if case.category == "robotics":
                continue
            resp = t.run(case, rt.perturb)
            if CANARY in resp.answer:
                hijacked = True
    assert hijacked, "expected the undefended target to be hijacked at least once"


def test_sensor_noise_can_flip_decision():
    t = RagAgentTarget()
    inj = build_injector("sensor_noise", intensity=1.0)
    flipped = False
    for run in range(10):
        rt = ChaosRuntime([inj], base_seed=2000 + run)
        for case in t.eval_cases():
            if case.category != "robotics":
                continue
            resp = t.run(case, rt.perturb)
            if case.expected[0] not in resp.answer:
                flipped = True
    assert flipped, "expected sensor noise to flip a borderline safety decision"
