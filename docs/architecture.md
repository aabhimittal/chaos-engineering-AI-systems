# Architecture

## Core idea: chaos as pipeline-stage middleware

An AI system is modelled as a pipeline of **stages**. Injectors are middleware
that perturb the payload flowing through one stage:

| Stage | Payload | Example injector |
| --- | --- | --- |
| `PROMPT` | user/system prompt string | prompt injection (direct) |
| `EMBEDDING` | dense query vector | embedding drift |
| `RETRIEVAL` | retrieved documents | prompt injection (indirect), dependency error |
| `LLM_OUTPUT` | generated text | hallucination spike, latency |
| `SENSOR` | robotics sensor readings | sensor noise |

The target calls a `perturb(stage, payload, case_id)` hook at each stage. With
no experiment running, `perturb` is the identity function, so **baseline and
chaos share the exact same code path** — the only difference is whether
injectors are attached.

```
question ─(PROMPT)─▶ embed ─(EMBEDDING)─▶ retrieve ─(RETRIEVAL)─▶ generate ─(LLM_OUTPUT)─▶ answer
                                                                   sensors ─(SENSOR)─▶ decision
```

## Components

```
chaoslab/
├── injectors/        fault injectors, one per failure mode (Injector ABC + Stage)
├── targets/          systems under test
│   ├── llm_client.py offline deterministic MockLLM + optional ClaudeLLM
│   └── rag_agent.py  sample RAG + robotics agent with a built-in KB & eval suite
├── probes/
│   ├── reasoning.py  reasoning-degradation judge (heuristic or Claude)
│   └── steady_state.py  steady-state hypothesis + violation checks
├── scoring.py        AI Robustness Score
├── experiment.py     engine: baseline vs chaos, ChaosRuntime, orchestration
├── metrics/          Prometheus exposition + MLflow logging (offline fallbacks)
├── agent/            deployable agent loop with /metrics + /healthz
├── registry.py       name → class plugin registry
├── config.py         env/config resolution (offline-first)
└── cli.py            chaoslab run | list | serve
```

## Experiment lifecycle

1. **Load** the YAML spec → target, injectors, steady-state hypothesis.
2. **Baseline** — run the eval suite with `identity_perturb`; score every
   response with the reasoning judge.
3. **Chaos** — build a `ChaosRuntime` that attaches injectors to their stages,
   each driven by a deterministic per-(stage, case) RNG; run the eval suite
   again through `runtime.perturb`; score again.
4. **Aggregate** metrics (mean correctness/overall, hijack rate, p95 latency).
5. **Check** the steady-state hypothesis → list of violations.
6. **Score** — `compute_robustness(baseline, chaos, violations)`.
7. **Sink** — record to Prometheus + MLflow.

## Determinism & safety

* Every injector's randomness comes from a seed derived from
  `seed:stage:case_id`, so an experiment is byte-for-byte reproducible.
* Chaos is in-process middleware around a **sandboxed** target, never real
  traffic. A `max_blast_radius` ceiling flags runs that exceeded their agreed
  blast radius, keeping experiments *incident-free*.

## Offline-first backends

| Concern | Offline default | Real backend |
| --- | --- | --- |
| Generation & judging | deterministic `MockLLM` / heuristic | Claude API (`ANTHROPIC_API_KEY`) |
| Experiment tracking | local JSON run store | MLflow server (`MLFLOW_TRACKING_URI`) |
| Metrics | in-memory + text exposition | Prometheus pushgateway/scrape |

Backends are selected by config, so the same experiment runs on a laptop with no
keys or against a full production observability stack.

## Deployment topology

```
        ┌──────────────┐    scrape /metrics    ┌────────────┐   ┌─────────┐
        │  Airflow DAG │                        │ Prometheus │──▶│ Grafana │
        │  (schedule)  │                        └─────┬──────┘   └─────────┘
        └──────┬───────┘                              │ alerts
               │ triggers                             ▼
               ▼                                  Alertmanager
   ┌─────────────────────────┐   logs runs    ┌──────────┐
   │  chaos-agent (K8s pod)   │ ─────────────▶ │  MLflow  │
   │  experiments via ConfigMap                └──────────┘
   └─────────────────────────┘
```
