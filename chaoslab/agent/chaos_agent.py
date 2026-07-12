"""Long-running chaos agent, deployable as a Kubernetes pod.

The agent discovers experiment specs from a directory (typically mounted from a
ConfigMap at ``/etc/chaoslab/experiments``), runs them on an interval, keeps the
latest robustness metrics in a shared Prometheus sink, and serves them at
``/metrics`` so Prometheus can scrape the pod. A ``/healthz`` endpoint supports
Kubernetes liveness/readiness probes.
"""

from __future__ import annotations

import glob
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional

from chaoslab.config import Config, load_config
from chaoslab.experiment import Experiment, ExperimentResult
from chaoslab.metrics.prometheus import PrometheusSink


class ChaosAgent:
    def __init__(self, experiments_dir: str = "experiments", config: Optional[Config] = None):
        self.experiments_dir = experiments_dir
        self.cfg = config or load_config()
        self.sink = PrometheusSink(self.cfg)
        self._last_results: List[ExperimentResult] = []

    def discover(self) -> List[str]:
        patterns = ("*.yaml", "*.yml")
        files: List[str] = []
        for pat in patterns:
            files.extend(glob.glob(os.path.join(self.experiments_dir, pat)))
        return sorted(files)

    def run_once(self) -> List[ExperimentResult]:
        results: List[ExperimentResult] = []
        for path in self.discover():
            try:
                exp = Experiment.from_yaml(path, config=self.cfg)
                result = exp.run(sinks=True)
                self.sink.record_experiment(result.name, result.report)
                results.append(result)
                print(
                    f"[chaos-agent] {result.name}: score={result.report.score:.1f} "
                    f"grade={result.report.grade} blast_radius={result.report.blast_radius:.2f}",
                    flush=True,
                )
            except Exception as exc:  # keep the agent alive on a bad spec
                print(f"[chaos-agent] error running {path}: {exc}", flush=True)
        self._last_results = results
        return results

    # -- HTTP / observability ----------------------------------------------
    def _make_handler(self):
        agent = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence default logging
                pass

            def do_GET(self):  # noqa: N802
                if self.path.startswith("/metrics"):
                    body = agent.sink.render_text().encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; version=0.0.4")
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path.startswith("/healthz"):
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"ok")
                else:
                    self.send_response(404)
                    self.end_headers()

        return Handler

    def serve(self, interval: int = 300, port: int = 8000, once: bool = False) -> None:
        server = ThreadingHTTPServer(("0.0.0.0", port), self._make_handler())
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print(f"[chaos-agent] serving metrics on :{port}/metrics", flush=True)
        try:
            while True:
                self.run_once()
                if once:
                    break
                time.sleep(interval)
        finally:
            server.shutdown()
