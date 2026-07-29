# Quickstart

## Install (offline core, no keys required)

```bash
pip install -e .
# or: pip install -e ".[dev]"   # + pytest/ruff
```

## Run experiments

```bash
# list registered injectors and bundled experiments
chaoslab list

# run a single experiment
chaoslab run experiments/hallucination_spike.yaml

# run every experiment in a directory
chaoslab run experiments/

# machine-readable output
chaoslab run experiments/ --json

# skip Prometheus/MLflow sinks (pure compute)
chaoslab run experiments/ --no-sinks

# gate CI: exit 1 if the steady-state hypothesis is violated
chaoslab run experiments/prompt_injection.yaml --fail-on-violation
```

## Advanced analyses

```bash
# recovery / MTTR: transient fault (rise -> peak -> decay), measure recovery time
chaoslab recover experiments/transient_hallucination.yaml
chaoslab recover experiments/hallucination_spike.yaml --profile 0,0.5,1,1,0.5,0

# breaking-point search: find the resilience threshold (min intensity that breaks)
chaoslab search experiments/prompt_injection.yaml
chaoslab search experiments/embedding_drift.yaml --grid 6 --tol 0.02
```

See [advanced analyses](advanced-analyses.md) for details.

## Run the agent locally

```bash
chaoslab serve --dir experiments --interval 60 --port 8000
curl localhost:8000/metrics    # Prometheus exposition
curl localhost:8000/healthz    # liveness probe
```

## Programmatic API

```python
from chaoslab.experiment import Experiment
from chaoslab.probes.steady_state import SteadyState

exp = Experiment(
    name="my_test",
    target_spec={"name": "rag_agent", "params": {"top_k": 3}},
    injector_specs=[{"name": "embedding_drift", "intensity": 0.7}],
    steady_state=SteadyState(min_correctness=0.6),
    runs=3,
)
result = exp.run(sinks=False)
print(result.report.score, result.report.grade)
```

See [`examples/run_local_experiment.py`](../examples/run_local_experiment.py).

## Writing an experiment

```yaml
name: my_experiment
description: what this experiment tests
target:
  name: rag_agent
  params: { top_k: 3 }
injectors:
  - name: embedding_drift      # any registered injector (see `chaoslab list`)
    intensity: 0.7             # 0..1 severity
  - name: hallucination_spike  # injectors compose — blend failure modes
    intensity: 0.5
steady_state:                  # the hypothesis that must hold under chaos
  min_correctness: 0.6
  min_overall: 0.6
  max_hijacked_rate: 0.0
  max_p95_latency_ms: 3000
runs: 3                        # repetitions (averaged) for stable numbers
seed: 1337                     # reproducibility
```

## Configuration (environment variables)

| Variable | Default | Purpose |
| --- | --- | --- |
| `CHAOSLAB_SEED` | `1337` | master RNG seed |
| `CHAOSLAB_LLM_PROVIDER` | `mock` | `mock` or `claude` (auto → `claude` if key present) |
| `ANTHROPIC_API_KEY` | — | enable real Claude generation + judge |
| `CHAOSLAB_JUDGE_PROVIDER` | `auto` | `auto` / `claude` / `heuristic` |
| `MLFLOW_TRACKING_URI` | — | log to a real MLflow server (else local JSON store) |
| `CHAOSLAB_PUSHGATEWAY` | — | push metrics to a Prometheus pushgateway |
| `CHAOSLAB_MAX_BLAST_RADIUS` | `0.85` | incident-free ceiling; runs above it are flagged |
