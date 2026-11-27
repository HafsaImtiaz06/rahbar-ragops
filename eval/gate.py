#!/usr/bin/env python3
"""Release gate (C2). Exit 0 = pass, 1 = gate failed, 2 = could not decide.

    python eval/gate.py                          # newest eval/results/*-eval.json
    python eval/gate.py --results path.json --config eval/gate.yaml
    python eval/gate.py promote path.json        # make a complete run the new baseline

The gate fails closed: an incomplete run, a smoke run, a missing metric or a missing baseline is
never a pass. Results marked "fixture" (eval/fixtures/gate/*) are only accepted with
--allow-fixture, so test data cannot pass a real gate.

Rules (thresholds in eval/gate.yaml):
  recall@5 not more than N percentage points below the baseline
  faithfulness at least a minimum
  p95 latency at most a maximum
  abstention correctness no worse than the baseline
  citation validity at least a minimum, and optionally a cap on over-abstention
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from metrics import flatten

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_CONFIG = HERE / "gate.yaml"
RESULTS_DIR = HERE / "results"

# The older run_eval.py results use flat names; map them onto the suite's dotted names.
LEGACY = {
    "recall_at_5": "retrieval_api.recall_at_5",
    "abstention_correctness": "answer.abstention_correctness",
    "citation_validity": "answer.citation_validity",
    "latency_seconds_p95": "answer.latency_seconds_p95",
}


class GateError(Exception):
    """The gate cannot decide (bad input), as opposed to deciding 'fail'."""


@dataclass
class Thresholds:
    baseline: str = "eval/baseline.m2.json"
    recall_metric: str = "retrieval_api.recall_at_5"
    max_recall_at_5_drop_pp: float = 2.0
    min_faithfulness: float = 0.80
    max_p95_latency_seconds: float = 240.0
    abstention_tolerance: float = 0.0
    min_citation_validity: float = 0.95
    max_over_abstention_rate: float | None = None


def load_thresholds(path: Path) -> Thresholds:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    unknown = set(data) - set(Thresholds.__dataclass_fields__)
    if unknown:
        raise GateError(f"{path}: unknown setting(s) {sorted(unknown)}")
    try:
        return Thresholds(**data)
    except TypeError as exc:
        raise GateError(f"{path}: {exc}") from exc


def metrics_of(doc: dict) -> dict[str, float]:
    """Flat {dotted name: number} for a results or baseline file, new or legacy format."""
    if "metrics" in doc and isinstance(doc["metrics"], dict):
        return flatten(doc["metrics"])
    flat = flatten({k: v for k, v in doc.items() if k != "per_question"})
    return {LEGACY[k]: v for k, v in flat.items() if k in LEGACY}


def evaluate(results: dict, baseline: dict | None, th: Thresholds, *,
             allow_fixture: bool = False, allow_smoke: bool = False) -> tuple[list[str], list[str]]:
    """Return (failures, checks). Raises GateError when the inputs cannot be judged."""
    if results.get("fixture") and not allow_fixture:
        raise GateError("results are marked as a test fixture; pass --allow-fixture to use them")
    if results.get("profile", "full") != "full" and not allow_smoke:
        raise GateError(f"results profile is '{results.get('profile')}', not a real evaluation")
    if "complete" in results and results["complete"] is not True:
        raise GateError("the evaluation run is not complete; an incomplete run cannot pass")

    got = metrics_of(results)
    failures: list[str] = []
    checks: list[str] = []

    def need(name: str) -> float | None:
        if name not in got:
            failures.append(f"missing metric {name}: a gate cannot pass on a metric it cannot see")
            return None
        return got[name]

    # recall@5 against the baseline
    recall = need(th.recall_metric)
    if baseline is None:
        raise GateError(f"no baseline at {th.baseline}; create one with 'gate.py promote'")
    base = metrics_of(baseline)
    if recall is not None:
        if th.recall_metric not in base:
            raise GateError(f"the baseline has no {th.recall_metric}")
        drop_pp = round((base[th.recall_metric] - recall) * 100, 6)   # avoid float noise at the limit
        line = (f"recall@5 {recall:.4f} vs baseline {base[th.recall_metric]:.4f} "
                f"(drop {drop_pp:.2f} pp, allowed {th.max_recall_at_5_drop_pp:g} pp)")
        (failures if drop_pp > th.max_recall_at_5_drop_pp else checks).append(
            line if drop_pp <= th.max_recall_at_5_drop_pp else "FAIL " + line)

    # abstention no worse than baseline
    abst = need("answer.abstention_correctness")
    if abst is not None:
        if "answer.abstention_correctness" not in base:
            raise GateError("the baseline has no answer.abstention_correctness")
        floor = base["answer.abstention_correctness"] - th.abstention_tolerance
        line = (f"abstention correctness {abst:.4f} vs baseline "
                f"{base['answer.abstention_correctness']:.4f} (minimum {floor:.4f})")
        (checks if abst >= floor else failures).append(line if abst >= floor else "FAIL " + line)

    faith = need("answer.faithfulness")
    if faith is not None:
        line = f"faithfulness {faith:.4f} (minimum {th.min_faithfulness:g})"
        (checks if faith >= th.min_faithfulness else failures).append(
            line if faith >= th.min_faithfulness else "FAIL " + line)

    p95 = need("answer.latency_seconds_p95")
    if p95 is not None:
        line = f"p95 latency {p95:.2f} s (maximum {th.max_p95_latency_seconds:g} s)"
        (checks if p95 <= th.max_p95_latency_seconds else failures).append(
            line if p95 <= th.max_p95_latency_seconds else "FAIL " + line)

    cite = need("answer.citation_validity")
    if cite is not None:
        line = f"citation validity {cite:.4f} (minimum {th.min_citation_validity:g})"
        (checks if cite >= th.min_citation_validity else failures).append(
            line if cite >= th.min_citation_validity else "FAIL " + line)

    if th.max_over_abstention_rate is not None:
        over = need("answer.over_abstention_rate")
        if over is not None:
            line = (f"over-abstention {over:.4f} (maximum {th.max_over_abstention_rate:g})")
            ok = over <= th.max_over_abstention_rate
            (checks if ok else failures).append(line if ok else "FAIL " + line)
    return failures, checks


def newest_results(directory: Path) -> Path:
    candidates = sorted(directory.glob("*-eval.json"))
    if not candidates:
        raise GateError(f"no *-eval.json files in {directory}; run eval/suite.py first")
    return candidates[-1]


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def promote(results_path: Path, destination: Path, *, allow_fixture: bool = False) -> None:
    doc = json.loads(results_path.read_text(encoding="utf-8"))
    if doc.get("fixture") and not allow_fixture:
        raise GateError("refusing to promote fixture data to a baseline")
    if doc.get("profile", "full") != "full" or doc.get("complete") is not True:
        raise GateError("only a complete, full-profile run can become the baseline")
    keep = {k: doc[k] for k in ("schema_version", "run_id", "profile", "complete", "config",
                                "metrics") if k in doc}
    keep["promoted_from"] = results_path.name
    destination.write_text(json.dumps(keep, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Release gate: exit 0 pass, 1 fail, 2 undecided.")
    parser.add_argument("action", nargs="?", choices=["check", "promote"], default="check")
    parser.add_argument("target", nargs="?", type=Path, help="promote: the results file")
    parser.add_argument("--results", type=Path)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--config", type=Path, help="defaults to $GATE_CONFIG, then gate.yaml")
    parser.add_argument("--baseline", type=Path, help="overrides the baseline in the config")
    parser.add_argument("--allow-fixture", action="store_true")
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args(argv)

    import os
    try:
        config_path = args.config or Path(os.environ.get("GATE_CONFIG", DEFAULT_CONFIG))
        th = load_thresholds(config_path)
        baseline_path = args.baseline or _resolve(th.baseline)

        if args.action == "promote":
            if not args.target:
                raise GateError("promote needs the results file")
            promote(args.target, baseline_path, allow_fixture=args.allow_fixture)
            print(f"baseline written: {baseline_path}")
            return 0

        results_path = args.results or newest_results(args.results_dir)
        results = json.loads(results_path.read_text(encoding="utf-8"))
        baseline = (json.loads(baseline_path.read_text(encoding="utf-8"))
                    if baseline_path.exists() else None)
        print(f"results:  {results_path}\nbaseline: {baseline_path}\nconfig:   {config_path}")
        if results.get("fixture"):
            print("NOTE: these results are TEST FIXTURE DATA, not a real evaluation")
        failures, checks = evaluate(results, baseline, th, allow_fixture=args.allow_fixture,
                                    allow_smoke=args.allow_smoke)
    except GateError as exc:
        print(f"\nGATE UNDECIDED: {exc}")
        return 2

    print()
    for line in checks:
        print(f"  ok    {line}")
    for line in failures:
        print(f"  {line}" if line.startswith(("FAIL", "missing")) else f"  FAIL  {line}")
    if failures:
        print("\nGATE FAILED")
        return 1
    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
