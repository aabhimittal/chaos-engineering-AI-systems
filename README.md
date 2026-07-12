# 🌀 AI-native Chaos Lab

**Chaos engineering for AI/LLM systems** — inject *AI-specific* failure modes,
not just node failures, measure the blast radius on a real target system, and
get a quantified **AI Robustness Score** you can gate rollouts on.

> Almost nobody tests AI failure modes properly. This framework does — and it
> runs **fully offline with zero API keys**, so every experiment is reproducible
> and incident-free.

[![CI](https://github.com/aabhimittal/chaos-engineering-ai-systems/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

---

## Why this exists

Traditional chaos engineering kills pods and adds latency. But AI systems fail
in ways infra chaos never surfaces:

| AI failure mode | What it models | Injected at stage |
| --- | --- | --- |
| 🤯 **Hallucination spike** | Model fabricates confident, unsupported claims | `llm_output` |
| 🧭 **Embedding drift** | Vector distribution shifts (e.g. after a model upgrade), degrading retrieval | `embedding` |
| 💉 **Prompt injection** | Adversarial text (direct or via a poisoned document) hijacks the model | `prompt` / `retrieval` |
| 🤖 **Robotics sensor noise** | Perception degrades — noise, dropout, miscalibration — flipping safety decisions | `sensor` |
| 🧱 **Infra faults** | Latency + dependency errors, for blended experiments | any stage |

Each injector is deterministic (seeded), so runs are **reproducible**. The lab
never touches production traffic — chaos is applied as in-process middleware
around a sandboxed target, with a configurable blast-radius ceiling.

## Architecture

```
        experiments/*.yaml
               │  declarative spec (target, injectors, steady-state hypothesis)
               ▼
     ┌────────────────────┐       perturb(stage, payload)        ┌───────────────────┐
     │  Experiment engine  │ ──────────────────────────────────▶ │  Target AI system  │
     │  baseline vs chaos  │ ◀────────────────────────────────── │ (sample RAG+agent) │
     └────────────────────┘         responses + events           └───────────────────┘
               │
   ┌───────────┼─────────────────────────────┐
   ▼           ▼                              ▼
 Reasoning   Robustness scoring          Metrics sinks
 judge       (blast radius, score,       ├─ Prometheus  → Grafana dashboards
 (LLM /      steady-state gate)          └─ MLflow      → resilience metrics per run
 heuristic)
```

* **Chaos agents** deploy to **Kubernetes** (Deployment + ConfigMap + RBAC + ServiceMonitor).
* **Airflow** schedules experiments on a cron and fails the DAG on a robustness regression.
* **Prometheus/Grafana** track blast radius, hijack rate, and reasoning degradation live.
* An **LLM agent** (Claude, or an offline heuristic) evaluates *reasoning degradation*.
* **MLflow** logs resilience metrics for every run.

See [`docs/architecture.md`](docs/architecture.md) for the full design.

## Quickstart (offline, no keys)

```bash
pip install -e .

chaoslab list                               # show injectors + experiments
chaoslab run experiments/                    # run the whole suite
chaoslab run experiments/prompt_injection.yaml
python examples/run_local_experiment.py      # programmatic API
```

Sample output:

```
  Experiment : prompt_injection
  Robustness :  38.3/100  [█████████░░░░░░░░░░░░░░░]  grade F
  Baseline   : 0.942   Under chaos: 0.408
  Degradation: 0.567   Retained: 0.433
  Blast radius: 67%   Hijacked: 33%   Injections: 28
  Steady state: VIOLATED
      - hijacked_rate 0.33 > max 0.00
      - overall 0.41 < min 0.50
```

## What the results look like

Running the bundled suite against the sample RAG agent (offline mock model):

| Experiment | Robustness | Grade | Blast radius | Hijacked | Steady state |
| --- | --- | --- | --- | --- | --- |
| embedding_drift | 96.9 | A | 5% | 0% | ✅ holds |
| hallucination_spike | 83.6 | B | 40% | 0% | ✅ holds |
| **prompt_injection** | **38.3** | **F** | **67%** | **33%** | ❌ **violated** |
| sensor_noise | 97.2 | A | 4% | 0% | ✅ holds |

The spectrum is the point: the undefended target degrades *gracefully* under
drift and sensor noise, but is **catastrophically vulnerable to prompt
injection** — exactly the kind of finding that should block an LLM rollout.
Swap in the real Claude backend (with a hardened system prompt) and re-run to
see defenses move the score.

## Use it as a rollout gate

`chaoslab run ... --fail-on-violation` exits non-zero when the steady-state
hypothesis breaks, so it drops straight into CI/CD:

```bash
chaoslab run experiments/prompt_injection.yaml --fail-on-violation || exit 1
```

## Real backends (optional)

Everything runs offline by default. Opt into real infrastructure via env vars:

```bash
export ANTHROPIC_API_KEY=sk-...        # real Claude generation + LLM judge
export MLFLOW_TRACKING_URI=http://localhost:5000
export CHAOSLAB_PUSHGATEWAY=localhost:9091
pip install -e ".[all]"
```

## Deploy the full stack

```bash
make compose-up     # chaos-agent + Prometheus + Grafana + MLflow
#   Grafana    → http://localhost:3000   (AI Chaos Lab dashboard, auto-provisioned)
#   Prometheus → http://localhost:9090
#   MLflow     → http://localhost:5000

# Kubernetes
kubectl apply -f deploy/kubernetes/

# Airflow — copy deploy/airflow/dags/chaos_experiments_dag.py into your dags/
```

## Repository layout

```
chaoslab/            core framework (injectors, targets, probes, scoring, engine, agent, CLI)
experiments/         declarative experiment specs (YAML)
deploy/kubernetes/   chaos-agent Deployment, ConfigMap, RBAC, ServiceMonitor
deploy/airflow/      scheduling DAG with a robustness gate
deploy/prometheus/   scrape config + resilience alert rules
deploy/grafana/      auto-provisioned datasource + dashboard
deploy/docker-compose.yml   full local stack
tests/               pytest suite (offline, deterministic)
docs/                architecture, failure modes, robustness score, quickstart
```

## Documentation

* [Quickstart](docs/quickstart.md)
* [Architecture](docs/architecture.md)
* [AI failure modes](docs/failure-modes.md)
* [The AI Robustness Score](docs/robustness-score.md)

## Development

```bash
make dev     # install with dev + all backends
make test    # pytest
make lint    # ruff
make demo    # run every experiment
```

## License

MIT © Abhishek Mittal
