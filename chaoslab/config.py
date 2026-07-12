"""Runtime configuration for chaoslab.

Configuration is resolved with the following precedence (highest first):

1. Explicit keyword arguments passed to :func:`load_config`.
2. Environment variables (``CHAOSLAB_*``, plus well-known backend vars such as
   ``ANTHROPIC_API_KEY`` and ``MLFLOW_TRACKING_URI``).
3. Built-in defaults, chosen so the whole framework runs offline with no keys.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Config:
    """Global configuration.

    Attributes:
        seed: Master seed for deterministic, reproducible chaos.
        llm_provider: ``"mock"`` (offline, default) or ``"claude"``.
        llm_model: Model id used when ``llm_provider == "claude"``.
        anthropic_api_key: Claude API key; when unset we fall back to the mock.
        judge_provider: LLM provider used by the reasoning-degradation judge.
        mlflow_tracking_uri: If set, log to a real MLflow server; else local store.
        mlflow_experiment: MLflow experiment name.
        prometheus_pushgateway: If set, push metrics here; else metrics are mocked.
        prometheus_job: Job label for pushed metrics.
        run_store_dir: Directory for the offline run/metric store.
        strict_incident_free: Abort a chaos run if it would exceed safety bounds.
        max_blast_radius: Safety ceiling (0..1). Runs stop before exceeding it.
    """

    seed: int = 1337
    llm_provider: str = "mock"
    llm_model: str = "claude-opus-4-8"
    anthropic_api_key: Optional[str] = None

    judge_provider: str = "auto"  # auto -> claude if key present else heuristic

    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment: str = "ai-chaos-lab"

    prometheus_pushgateway: Optional[str] = None
    prometheus_job: str = "chaoslab"

    run_store_dir: str = ".chaoslab/runs"

    strict_incident_free: bool = True
    max_blast_radius: float = 0.85

    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def claude_available(self) -> bool:
        return bool(self.anthropic_api_key)

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Never surface secrets in serialized config.
        if d.get("anthropic_api_key"):
            d["anthropic_api_key"] = "***redacted***"
        return d


def load_config(**overrides: Any) -> Config:
    """Build a :class:`Config` from env + overrides."""
    cfg = Config(
        seed=int(os.environ.get("CHAOSLAB_SEED", 1337)),
        llm_provider=os.environ.get("CHAOSLAB_LLM_PROVIDER", "mock"),
        llm_model=os.environ.get("CHAOSLAB_LLM_MODEL", "claude-opus-4-8"),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        judge_provider=os.environ.get("CHAOSLAB_JUDGE_PROVIDER", "auto"),
        mlflow_tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"),
        mlflow_experiment=os.environ.get("CHAOSLAB_MLFLOW_EXPERIMENT", "ai-chaos-lab"),
        prometheus_pushgateway=os.environ.get("CHAOSLAB_PUSHGATEWAY"),
        prometheus_job=os.environ.get("CHAOSLAB_PROM_JOB", "chaoslab"),
        run_store_dir=os.environ.get("CHAOSLAB_RUN_STORE", ".chaoslab/runs"),
        strict_incident_free=_env_bool("CHAOSLAB_STRICT", True),
        max_blast_radius=float(os.environ.get("CHAOSLAB_MAX_BLAST_RADIUS", 0.85)),
    )

    # If a Claude key is present and the user did not pin the mock explicitly,
    # opt into the real provider automatically.
    if cfg.anthropic_api_key and "CHAOSLAB_LLM_PROVIDER" not in os.environ:
        cfg.llm_provider = "claude"

    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            cfg.extra[key] = value
    return cfg
