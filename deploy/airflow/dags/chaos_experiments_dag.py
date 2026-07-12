"""Airflow DAG that schedules AI chaos experiments.

Each bundled experiment becomes a task; tasks run in parallel on a daily
schedule (adjust ``schedule`` as needed). A final gate task asserts that no
experiment regressed below its robustness threshold, so a scheduled chaos run
can fail the DAG — and page an on-call — when AI resilience degrades.

Drop this file into your Airflow ``dags/`` folder. It imports ``chaoslab``,
so install the package in the Airflow image (``pip install chaoslab``).
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

EXPERIMENTS_DIR = os.environ.get("CHAOSLAB_EXPERIMENTS_DIR", "/opt/airflow/experiments")
ROBUSTNESS_THRESHOLD = float(os.environ.get("CHAOSLAB_MIN_ROBUSTNESS", "60"))

EXPERIMENTS = [
    "hallucination_spike",
    "embedding_drift",
    "prompt_injection",
    "sensor_noise",
]

default_args = {
    "owner": "ai-reliability",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def run_experiment(experiment: str, **context) -> dict:
    """Run one chaos experiment and push its report to XCom."""
    from chaoslab.experiment import Experiment

    path = os.path.join(EXPERIMENTS_DIR, f"{experiment}.yaml")
    result = Experiment.from_yaml(path).run(sinks=True)
    report = result.report.as_dict()
    context["ti"].xcom_push(key="report", value=report)
    print(f"{experiment}: score={report['score']} grade={report['grade']}")
    return report


def gate(**context) -> None:
    """Fail the DAG if any experiment fell below the robustness threshold."""
    ti = context["ti"]
    failures = []
    for experiment in EXPERIMENTS:
        report = ti.xcom_pull(task_ids=f"run_{experiment}", key="report") or {}
        score = report.get("score", 0.0)
        if score < ROBUSTNESS_THRESHOLD or not report.get("steady_state_pass", True):
            failures.append(f"{experiment} (score={score})")
    if failures:
        raise ValueError(
            f"AI robustness gate failed for: {', '.join(failures)} "
            f"(threshold={ROBUSTNESS_THRESHOLD})"
        )
    print("All experiments passed the robustness gate.")


with DAG(
    dag_id="ai_chaos_experiments",
    description="Scheduled AI-native chaos experiments with a robustness gate",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule="0 3 * * *",  # daily at 03:00
    catchup=False,
    tags=["chaos", "ai", "resilience", "mlops"],
) as dag:
    experiment_tasks = [
        PythonOperator(
            task_id=f"run_{name}",
            python_callable=run_experiment,
            op_kwargs={"experiment": name},
        )
        for name in EXPERIMENTS
    ]

    robustness_gate = PythonOperator(task_id="robustness_gate", python_callable=gate)

    experiment_tasks >> robustness_gate
