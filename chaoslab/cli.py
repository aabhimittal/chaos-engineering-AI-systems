"""Command-line interface: ``chaoslab run | list | serve | version``."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import List

from chaoslab import __version__
from chaoslab.config import load_config
from chaoslab.experiment import Experiment, ExperimentResult
from chaoslab.registry import INJECTORS

_BAR_WIDTH = 24


def _bar(value: float, lo: float = 0.0, hi: float = 100.0) -> str:
    frac = 0.0 if hi == lo else max(0.0, min(1.0, (value - lo) / (hi - lo)))
    filled = int(round(frac * _BAR_WIDTH))
    return "█" * filled + "░" * (_BAR_WIDTH - filled)


def _print_result(result: ExperimentResult) -> None:
    r = result.report
    print()
    print(f"  Experiment : {result.name}")
    print(f"  Robustness : {r.score:5.1f}/100  [{_bar(r.score)}]  grade {r.grade}")
    print(f"  Baseline   : {r.baseline_overall:.3f}   Under chaos: {r.chaos_overall:.3f}")
    print(f"  Degradation: {r.degradation:.3f}   Retained: {r.retained_quality:.3f}")
    print(f"  Blast radius: {r.blast_radius:.0%}   Hijacked: {r.hijacked_rate:.0%}   Injections: {len(result.events)}")
    status = "HOLDS" if r.steady_state_pass else "VIOLATED"
    print(f"  Steady state: {status}")
    for v in r.violations:
        print(f"      - {v}")
    if result.aborted:
        print("  ⚠️  blast radius exceeded the incident-free ceiling (flagged for review)")
    if result.run_ref:
        print(f"  Run logged : {result.run_ref}")
    print()


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config()
    paths: List[str] = []
    for target in args.experiments:
        if os.path.isdir(target):
            paths.extend(sorted(glob.glob(os.path.join(target, "*.y*ml"))))
        else:
            paths.append(target)
    if not paths:
        print("no experiment files found", file=sys.stderr)
        return 2

    results = []
    for path in paths:
        exp = Experiment.from_yaml(path, config=cfg)
        result = exp.run(sinks=not args.no_sinks)
        results.append(result)
        if not args.json:
            _print_result(result)

    if args.json:
        print(json.dumps([r.as_dict() for r in results], indent=2))

    # Non-zero exit if any experiment violated its steady-state hypothesis, so
    # this can gate a CI/CD rollout.
    failed = [r for r in results if not r.report.steady_state_pass]
    if failed and args.fail_on_violation:
        return 1
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    print("Registered injectors:")
    for name, cls in sorted(INJECTORS.items()):
        print(f"  - {name:20s} stage={cls.stage.value}")
    exp_dir = args.dir
    files = sorted(glob.glob(os.path.join(exp_dir, "*.y*ml"))) if os.path.isdir(exp_dir) else []
    print(f"\nExperiments in ./{exp_dir}:")
    for f in files:
        print(f"  - {f}")
    if not files:
        print("  (none)")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from chaoslab.agent.chaos_agent import ChaosAgent

    agent = ChaosAgent(experiments_dir=args.dir, config=load_config())
    agent.serve(interval=args.interval, port=args.port, once=args.once)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="chaoslab", description="AI-native chaos engineering lab")
    p.add_argument("--version", action="version", version=f"chaoslab {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run one or more experiments")
    run.add_argument("experiments", nargs="+", help="experiment YAML file(s) or a directory")
    run.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    run.add_argument("--no-sinks", action="store_true", help="skip Prometheus/MLflow logging")
    run.add_argument("--fail-on-violation", action="store_true", help="exit 1 if steady state is violated")
    run.set_defaults(func=_cmd_run)

    lst = sub.add_parser("list", help="list injectors and experiments")
    lst.add_argument("--dir", default="experiments", help="experiments directory")
    lst.set_defaults(func=_cmd_list)

    serve = sub.add_parser("serve", help="run the chaos agent with a /metrics endpoint")
    serve.add_argument("--dir", default="experiments", help="experiments directory")
    serve.add_argument("--interval", type=int, default=300, help="seconds between runs")
    serve.add_argument("--port", type=int, default=8000, help="metrics port")
    serve.add_argument("--once", action="store_true", help="run a single cycle then exit")
    serve.set_defaults(func=_cmd_serve)

    return p


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
