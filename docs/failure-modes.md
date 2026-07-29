# AI failure modes

Each injector is a subclass of `chaoslab.injectors.base.Injector`, bound to one
pipeline `Stage`, with an `intensity` in `[0, 1]` scaling severity. Add your own
by subclassing and registering it in `chaoslab/registry.py`.

## 🤯 Hallucination spike — `hallucination_spike`

**Stage:** `llm_output`

Simulates a model fabricating confident but unsupported content. It injects
fabricated statistics, invented citations (fake studies, RFCs, ISO standards),
and rewrites hedges ("might" → "definitely") into unwarranted certainty. Firing
probability and the number of fabricated claims scale with intensity.

The reasoning judge detects these as **groundedness** red-flags (citation
patterns / percentages / over-confident phrasing not present in the retrieved
context), so hallucinated answers score lower even when they still contain the
right keywords.

## 🧭 Embedding drift — `embedding_drift`

**Stage:** `embedding`

Simulates a stale or upgraded embedding model / corrupted index by perturbing
query vectors with **Gaussian noise** plus a **systematic directional bias**.
The event records the cosine similarity between the original and drifted vector.
Downstream, retrieval returns worse documents, so the RAG target grounds its
answer on the wrong context — a realistic, quiet failure that classic infra
chaos never surfaces.

## 💉 Prompt injection — `prompt_injection`

**Stage:** `prompt` (direct) or `retrieval` (indirect, `params: {target: retrieval}`)

Appends an adversarial instruction — either to the user prompt or smuggled into
a retrieved document — that tries to override the system instructions and make
the model emit an attacker **canary token** (`PWNED_BY_CHAOSLAB`). The sample
target's mock model is deliberately *undefended*, so indirect injection can
hijack it; the safety gate in the judge scores any hijacked answer at ~0 overall
and the steady-state hypothesis (`max_hijacked_rate: 0.0`) is violated.

This is the failure mode most teams under-test — and the one this lab scores
harshest.

## 🤖 Robotics sensor noise — `sensor_noise`

**Stage:** `sensor`

Corrupts raw sensor readings (scalars or lists — lidar, IMU, joint angles) with
**Gaussian noise**, **dropout** (channels go to zero, modelling lost packets /
dead pixels), and **bias** (a miscalibrated sensor). The sample agent makes a
safety decision (stop vs proceed) from a distance reading; near the decision
boundary, noise can flip the call — quantifying perception fragility for embodied
AI.

## 📉 Context truncation — `context_truncation`

**Stage:** `retrieval`

Simulates a context-window overflow / aggressive truncation policy: trailing
retrieved documents are dropped and the survivors are clipped to a shrinking
character budget. At high intensity the retrieval is *starved* entirely (zero
docs), and the answer-bearing passage silently vanishes — a very common,
under-tested RAG failure as prompts grow.

## ✂️ Output truncation — `output_truncation`

**Stage:** `llm_output`

Models generation hitting `max_tokens` or a dropped streaming connection. The
answer keeps a leading fraction and loses its tail — where the payload often
lives — so it stays fluent and plausible while correctness quietly drops.

## 🔤 Unicode perturbation — `unicode_perturbation`

**Stage:** `prompt`

Corrupts the query with homoglyph confusables (Latin `a` → Cyrillic `а`) and
invisible zero-width characters. Visually identical to a human, but a different
byte sequence, so exact retrieval/matching degrades. Models copy-paste
encoding bugs and adversarial filter-evasion — the kind of silent failure
unicode-naive systems ship to production.

## 🧱 Infrastructure faults — `latency`, `dependency_error`

**Stages:** `llm_output` (latency), `retrieval` (error)

Classic chaos, included so AI and infra faults can be **blended** in one
experiment. `latency` reports added milliseconds (checked against the latency
SLO in the steady-state hypothesis); `dependency_error` probabilistically raises
a `FaultInjectionError` to exercise the target's fallback path.

## Composing failure modes

Injectors compose — list several under `injectors:` in one experiment to model
correlated failures (e.g. embedding drift *and* a hallucination spike during an
incident):

```yaml
injectors:
  - { name: embedding_drift, intensity: 0.6 }
  - { name: hallucination_spike, intensity: 0.5 }
```
