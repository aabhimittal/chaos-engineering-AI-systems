import json
import os

from chaoslab.config import load_config
from chaoslab.experiment import Experiment
from chaoslab.metrics.prometheus import PrometheusSink
from chaoslab.probes.steady_state import SteadyState


def _experiment(name, injectors, ss=None, runs=2):
    return Experiment(
        name=name,
        target_spec={"name": "rag_agent", "params": {"top_k": 3}},
        injector_specs=injectors,
        steady_state=ss or SteadyState(),
        runs=runs,
        config=load_config(),
    )


def test_experiment_runs_and_scores(tmp_path):
    cfg = load_config(run_store_dir=str(tmp_path))
    exp = _experiment("halluc", [{"name": "hallucination_spike", "intensity": 0.9}])
    exp.cfg = cfg
    result = exp.run(sinks=True)
    assert 0.0 <= result.report.score <= 100.0
    assert result.report.baseline_overall > 0
    assert len(result.events) > 0
    # a local MLflow-style run file was written
    assert result.run_ref and os.path.exists(result.run_ref)
    with open(result.run_ref) as fh:
        record = json.load(fh)
    assert record["metrics"]["robustness_score"] == result.report.score


def test_prompt_injection_experiment_is_scored_worst():
    halluc = _experiment("h", [{"name": "hallucination_spike", "intensity": 0.85}]).run(sinks=False)
    inject = _experiment(
        "i",
        [{"name": "prompt_injection", "intensity": 0.9, "params": {"target": "retrieval"}}],
        ss=SteadyState(min_correctness=0.5, min_overall=0.5, max_hijacked_rate=0.0),
        runs=3,
    ).run(sinks=False)
    assert inject.report.hijacked_rate > 0
    assert not inject.report.steady_state_pass
    assert inject.report.score < halluc.report.score


def test_experiment_is_reproducible():
    a = _experiment("r", [{"name": "embedding_drift", "intensity": 0.7}]).run(sinks=False)
    b = _experiment("r", [{"name": "embedding_drift", "intensity": 0.7}]).run(sinks=False)
    assert a.report.score == b.report.score
    assert a.report.blast_radius == b.report.blast_radius


def test_from_yaml_loads_bundled_experiments():
    exp = Experiment.from_yaml("experiments/sensor_noise.yaml")
    assert exp.name == "sensor_noise"
    result = exp.run(sinks=False)
    assert result.metrics["cases"] > 0


def test_prometheus_sink_exposition_format():
    exp = _experiment("h", [{"name": "hallucination_spike", "intensity": 0.9}])
    result = exp.run(sinks=False)
    sink = PrometheusSink(exp.cfg)
    sink.record_experiment("h", result.report)
    text = sink.render_text()
    assert "chaoslab_robustness_score" in text
    assert 'experiment="h"' in text
