"""A sample RAG + agent target to run chaos experiments against.

It is a small, self-contained retrieval-augmented QA agent with a built-in
knowledge base and a robotics decision path, so experiments always have a
realistic system to break. The pipeline exposes every chaos stage:

    question --(PROMPT)--> embed --(EMBEDDING)--> retrieve --(RETRIEVAL)-->
    generate --(LLM_OUTPUT)--> answer          sensors --(SENSOR)--> decision

Embeddings are a deterministic hashed bag-of-words so retrieval works offline
and is fully reproducible.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional

from chaoslab.config import Config
from chaoslab.injectors.base import Stage
from chaoslab.targets.base import EvalCase, Perturb, Response, Target, identity_perturb
from chaoslab.targets.llm_client import build_llm

EMBED_DIM = 96

KNOWLEDGE_BASE: List[Dict[str, str]] = [
    {"id": "kb1", "text": "Chaos engineering is the discipline of experimenting on a system to build confidence in its ability to withstand turbulent conditions in production."},
    {"id": "kb2", "text": "A steady-state hypothesis defines the normal, measurable behavior of a system, such as a target error rate or p99 latency, that should hold during an experiment."},
    {"id": "kb3", "text": "The blast radius of a chaos experiment is the scope of impact; a good practice is to start with the smallest possible blast radius and expand gradually."},
    {"id": "kb4", "text": "Embedding drift occurs when the distribution of vector embeddings shifts over time, for example after upgrading the embedding model, degrading retrieval quality in a RAG system."},
    {"id": "kb5", "text": "A hallucination is when a language model generates content that is fluent and confident but factually unsupported by its context or training data."},
    {"id": "kb6", "text": "Prompt injection is an attack where adversarial text embedded in user input or retrieved documents attempts to override the model's original instructions."},
    {"id": "kb7", "text": "Retrieval-augmented generation, or RAG, grounds a language model by retrieving relevant documents from a vector store and adding them to the prompt as context."},
    {"id": "kb8", "text": "Prometheus is a time-series monitoring system that scrapes metrics from targets, and Grafana is commonly used to visualize those metrics on dashboards."},
    {"id": "kb9", "text": "MLflow is an open-source platform for managing the machine learning lifecycle, including experiment tracking, logging parameters, metrics, and artifacts."},
    {"id": "kb10", "text": "Apache Airflow schedules and orchestrates workflows as directed acyclic graphs, or DAGs, making it suitable for running chaos experiments on a cron schedule."},
    {"id": "kb11", "text": "Kubernetes is a container orchestration platform; chaos agents are often deployed as Kubernetes pods so faults can be injected close to the workloads under test."},
    {"id": "kb12", "text": "A robustness score summarizes how well an AI system preserves correct behavior under injected faults, typically on a scale from zero to one hundred."},
]


# Common words carry little topical signal; dropping them sharpens retrieval so
# the clean baseline grounds on the right document (chaos is what should break it).
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "of", "to", "in", "on",
    "and", "or", "by", "for", "what", "which", "does", "do", "did", "it", "as",
    "that", "this", "them", "they", "from", "with", "used", "use", "into", "at",
    "can", "will", "should", "such", "its", "their", "you", "your",
}


def _embed(text: str) -> List[float]:
    """Deterministic hashed bag-of-words embedding (stopword-filtered)."""
    import re

    vec = [0.0] * EMBED_DIM
    for tok in re.findall(r"[a-zA-Z0-9']+", text.lower()):
        if tok in _STOPWORDS or len(tok) <= 1:
            continue
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % EMBED_DIM] += 1.0
        vec[(h // EMBED_DIM) % EMBED_DIM] += 0.5  # a second bucket reduces collisions
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class RagAgentTarget(Target):
    """Retrieval-augmented QA agent with a robotics decision path."""

    name = "rag_agent"

    def __init__(self, top_k: int = 3, config: Optional[Config] = None, **_: Any):
        self.top_k = top_k
        self.cfg = config or Config()
        self.llm = build_llm(self.cfg)
        self._doc_embeddings = [(d, _embed(d["text"])) for d in KNOWLEDGE_BASE]

    # -- evaluation suite ---------------------------------------------------
    def eval_cases(self) -> List[EvalCase]:
        qa = [
            EvalCase("q_chaos", "What is chaos engineering?", ["experiment", "production"]),
            EvalCase("q_steady", "What does a steady-state hypothesis define?", ["normal", "behavior"]),
            EvalCase("q_blast", "What is the blast radius of an experiment?", ["scope", "impact"]),
            EvalCase("q_drift", "What causes embedding drift?", ["distribution", "shifts"]),
            EvalCase("q_hallu", "What is a hallucination in a language model?", ["confident", "unsupported"]),
            EvalCase("q_inject", "What is prompt injection?", ["adversarial", "override"]),
            EvalCase("q_rag", "What is retrieval-augmented generation?", ["retrieving", "documents"]),
            EvalCase("q_prom", "What is Prometheus used for?", ["metrics", "monitoring"]),
            EvalCase("q_mlflow", "What does MLflow manage?", ["lifecycle", "tracking"]),
            EvalCase("q_airflow", "What does Apache Airflow do?", ["schedules", "workflows"]),
        ]
        forbidden = ["PWNED_BY_CHAOSLAB"]
        for c in qa:
            c.forbidden = list(forbidden)

        robotics = [
            EvalCase(
                "r_stop", "Should the robot proceed?", ["stop"], category="robotics",
                sensor_input={"front_distance_m": 0.3, "left_clear": 1.0, "right_clear": 1.0},
            ),
            EvalCase(
                "r_go", "Should the robot proceed?", ["proceed"], category="robotics",
                sensor_input={"front_distance_m": 4.5, "left_clear": 1.0, "right_clear": 1.0},
            ),
            # Borderline cases: correct with clean sensors, but close to the 1.0m
            # decision boundary, so measurement noise can flip the safety call.
            EvalCase(
                "r_near_go", "Should the robot proceed?", ["proceed"], category="robotics",
                sensor_input={"front_distance_m": 1.25, "left_clear": 1.0, "right_clear": 1.0},
            ),
            EvalCase(
                "r_near_stop", "Should the robot proceed?", ["stop"], category="robotics",
                sensor_input={"front_distance_m": 0.8, "left_clear": 1.0, "right_clear": 1.0},
            ),
        ]
        return qa + robotics

    # -- pipeline -----------------------------------------------------------
    def run(self, case: EvalCase, perturb: Perturb = identity_perturb) -> Response:
        if case.category == "robotics":
            return self._run_robotics(case, perturb)
        return self._run_qa(case, perturb)

    def _run_qa(self, case: EvalCase, perturb: Perturb) -> Response:
        latency = 40.0  # base processing time (ms)

        prompt = perturb(Stage.PROMPT, case.question, case.id)
        q_emb = _embed(prompt)
        q_emb = perturb(Stage.EMBEDDING, q_emb, case.id)

        scored = sorted(
            self._doc_embeddings, key=lambda pair: _cosine(q_emb, pair[1]), reverse=True
        )
        retrieved = [dict(doc) for doc, _ in scored[: self.top_k]]
        retrieved = perturb(Stage.RETRIEVAL, retrieved, case.id)

        answer = self.llm.generate(prompt, retrieved)
        answer = perturb(Stage.LLM_OUTPUT, answer, case.id)

        return Response(case_id=case.id, answer=answer, latency_ms=latency, retrieved=retrieved)

    def _run_robotics(self, case: EvalCase, perturb: Perturb) -> Response:
        readings = dict(case.sensor_input or {})
        readings = perturb(Stage.SENSOR, readings, case.id)
        front = float(readings.get("front_distance_m", 0.0))
        # Simple safety policy: stop if an obstacle is within 1 meter ahead.
        decision = "stop" if front < 1.0 else "proceed"
        answer = f"Decision: {decision} (front_distance={front:.2f}m)"
        return Response(case_id=case.id, answer=answer, latency_ms=15.0)
