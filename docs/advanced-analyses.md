# Advanced analyses

Beyond a single baseline-vs-chaos run, the lab ships two higher-order analyses
that answer questions steady-state experiments can't.

## Recovery / MTTR — how long does the pain last?

A production incident isn't just "how bad" but "for how long". A hallucination
spike or a brief embedding-index corruption should *clear* once the fault
passes. `chaoslab recover` drives the fault through a **time profile** (an
intensity multiplier per timestep — rise, peak, decay) and measures the
recovery curve.

```bash
chaoslab recover experiments/transient_hallucination.yaml
```

```
  Recovery analysis: transient_hallucination
  Blast radius   : ▁▁▃▄▂▁▁▁  (peak 43% @ t=3)
  Min score      : 70.1/100
  MTTR           : 1 timestep(s) after peak
  Recovered      : yes
  Timeline:
    t=0  intensity=0.00  blast=0.00  score=100  [ok ]
    t=2  intensity=0.80  blast=0.38  score=72  [VIO]
    t=3  intensity=1.00  blast=0.43  score=70  [VIO]
    t=4  intensity=0.60  blast=0.17  score=93  [ok ]
    ...
```

* **MTTR** = timesteps after the peak until the steady-state hypothesis holds
  again (a recovery-time analogue for AI behavior).
* **Recovered** = did steady state hold by the end of the profile?

Define the profile inline in the experiment YAML:

```yaml
recovery_profile: [0.0, 0.4, 0.8, 1.0, 0.6, 0.3, 0.0, 0.0]
```

or override on the CLI with `--profile 0,0.5,1,1,0.5,0`. Because the engine is
deterministic, the recovery curve is fully reproducible.

## Breaking-point search — what is the resilience threshold?

Instead of "does it survive intensity 0.85?", ask "**at what intensity does it
break?**". `chaoslab search` sweeps the fault intensity and finds the smallest
multiplier at which the steady-state hypothesis first fails — the **resilience
threshold**. A higher threshold is a more robust system, giving you one tunable
number to track across model versions or prompt-hardening iterations.

```bash
chaoslab search experiments/prompt_injection.yaml
```

```
  Breaking-point search: prompt_injection
  Verdict            : breaks at intensity 0.03
  Resilience threshold: 0.03  (breaks at this intensity scale)
  Holds up to        : 0.00
```

The search is a coarse grid scan (robust to the mild non-monotonicity of
probabilistic faults) that brackets the first failure, followed by a bisection
to refine the threshold to `--tol`. Prompt injection breaking at `0.03` — while
embedding drift and sensor noise never break across the full range — is exactly
the kind of comparative, actionable finding that prioritizes hardening work.

## Programmatic API

```python
from chaoslab.experiment import Experiment
from chaoslab.analysis import run_transient, find_breaking_point

exp = Experiment.from_yaml("experiments/transient_hallucination.yaml")

recovery = run_transient(exp)                 # -> RecoveryReport
print(recovery.mttr_steps, recovery.recovered)

threshold = find_breaking_point(exp)          # -> BreakingPointReport
print(threshold.resilience_threshold)
```
