from chaoslab.probes.reasoning import ReasoningScore
from chaoslab.scoring import compute_robustness


def score(cid, overall, hijacked=False, correctness=None):
    c = overall if correctness is None else correctness
    return ReasoningScore(cid, c, 1.0, 0.0 if hijacked else 1.0, overall, hijacked)


def test_perfect_resilience_scores_100():
    baseline = [score(f"c{i}", 1.0) for i in range(5)]
    chaos = [score(f"c{i}", 1.0) for i in range(5)]
    report = compute_robustness(baseline, chaos, violations=[])
    assert report.score == 100.0
    assert report.grade == "A"
    assert report.blast_radius == 0.0


def test_degradation_lowers_score_and_blast_radius():
    baseline = [score(f"c{i}", 1.0) for i in range(4)]
    # two of four cases collapse
    chaos = [score("c0", 0.2), score("c1", 0.1), score("c2", 1.0), score("c3", 1.0)]
    report = compute_robustness(baseline, chaos, violations=[])
    assert report.blast_radius == 0.5
    assert report.degradation > 0
    assert report.score < 100


def test_hijack_tanks_safety_and_score():
    baseline = [score(f"c{i}", 1.0) for i in range(4)]
    chaos = [score(f"c{i}", 0.0, hijacked=True) for i in range(4)]
    report = compute_robustness(baseline, chaos, violations=["hijacked_rate 1.00 > max 0.00"])
    assert report.hijacked_rate == 1.0
    assert report.grade == "F"
    assert not report.steady_state_pass


def test_steady_state_violation_applies_penalty():
    baseline = [score(f"c{i}", 1.0) for i in range(4)]
    chaos = [score(f"c{i}", 1.0) for i in range(4)]
    clean = compute_robustness(baseline, chaos, violations=[])
    penalized = compute_robustness(baseline, chaos, violations=["some violation"])
    assert penalized.score < clean.score


def test_report_serialization_roundtrip():
    report = compute_robustness([score("c0", 1.0)], [score("c0", 0.5)], violations=[])
    d = report.as_dict()
    assert set(["score", "grade", "blast_radius", "hijacked_rate"]).issubset(d)
