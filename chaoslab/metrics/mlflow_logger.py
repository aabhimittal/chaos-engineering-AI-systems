"""MLflow resilience-metrics logger.

Logs each experiment's parameters, robustness metrics, and the full JSON report
as an artifact. When MLflow is installed and a tracking URI is configured it
uses the real MLflow client; otherwise it writes an MLflow-shaped run to a local
JSON store under ``run_store_dir`` so the demo works with zero infrastructure.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, Optional

from chaoslab.config import Config
from chaoslab.scoring import RobustnessReport


class MLflowLogger:
    def __init__(self, config: Optional[Config] = None):
        self.cfg = config or Config()
        self._client = None
        if self.cfg.mlflow_tracking_uri:
            try:  # pragma: no cover - requires mlflow + server
                import mlflow

                mlflow.set_tracking_uri(self.cfg.mlflow_tracking_uri)
                mlflow.set_experiment(self.cfg.mlflow_experiment)
                self._client = mlflow
            except Exception:
                self._client = None

    def log_experiment(
        self,
        experiment: str,
        params: Dict[str, Any],
        report: RobustnessReport,
        extra_metrics: Optional[Dict[str, float]] = None,
    ) -> str:
        """Log a run; returns the run id (real) or local run path."""
        metrics = {
            "robustness_score": report.score,
            "blast_radius": report.blast_radius,
            "reasoning_degradation": report.degradation,
            "hijacked_rate": report.hijacked_rate,
            "retained_quality": report.retained_quality,
            "baseline_overall": report.baseline_overall,
            "chaos_overall": report.chaos_overall,
            "steady_state_pass": 1.0 if report.steady_state_pass else 0.0,
        }
        if extra_metrics:
            metrics.update(extra_metrics)

        if self._client is not None:  # pragma: no cover - requires mlflow
            with self._client.start_run(run_name=f"chaos-{experiment}") as run:
                self._client.log_params(params)
                self._client.log_metrics(metrics)
                self._client.log_dict(report.as_dict(), "robustness_report.json")
                return run.info.run_id

        return self._log_local(experiment, params, metrics, report)

    def _log_local(
        self,
        experiment: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        report: RobustnessReport,
    ) -> str:
        os.makedirs(self.cfg.run_store_dir, exist_ok=True)
        run_id = uuid.uuid4().hex[:12]
        record = {
            "run_id": run_id,
            "experiment": experiment,
            "mlflow_experiment": self.cfg.mlflow_experiment,
            "timestamp": time.time(),
            "params": params,
            "metrics": metrics,
            "report": report.as_dict(),
        }
        path = os.path.join(self.cfg.run_store_dir, f"{experiment}-{run_id}.json")
        with open(path, "w") as fh:
            json.dump(record, fh, indent=2)
        return path
