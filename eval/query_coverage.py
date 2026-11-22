#!/usr/bin/env python3
"""Query coverage check (RAGOps part E2 and E4).

For every live query, find the most similar golden question (cosine similarity of their
embeddings). A query is covered when that score reaches the threshold; the share of covered
queries is the coverage. The command exits 1 when coverage is below the minimum, so it can gate
a release or a CI step, and exits 2 when embeddings cannot be computed.

    python eval/query_coverage.py --label before
    python eval/query_coverage.py --label after            # after adding golden questions (E4)
    python eval/query_coverage.py --calibrate              # how to choose the threshold (E2)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import urllib.request
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
Embedder = Callable[[list[str]], list[list[float]]]


def load_golden(path: Path) -> list[dict]:
    items = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("questions") or []
    golden = [{"id": str(q["id"]), "question": str(q["question"]), "type": q.get("type", "")}
              for q in items if q.get("question")]
    if not golden:
        raise SystemExit(f"no golden questions in {path}")
    return golden


def load_live(path: Path) -> list[dict]:
    items = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("queries") or []
    live = [{"id": str(q["id"]), "text": str(q["text"]), "kind": q.get("kind", ""),
             "topic": q.get("topic", "other")} for q in items if q.get("text")]
    if not live:
        raise SystemExit(f"no live queries in {path}")
    return live


def ollama_embedder(url: str, model: str, batch: int = 16) -> Embedder:
    def embed(texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch):
            body = json.dumps({"model": model, "input": texts[start:start + batch]}).encode()
            request = urllib.request.Request(
                url.rstrip("/") + "/api/embed", data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=120) as response:
                vectors.extend(json.loads(response.read())["embeddings"])
        return vectors
    return embed


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def match(live: list[dict], golden: list[dict], embed: Embedder, threshold: float) -> list[dict]:
    vectors = embed([q["text"] for q in live] + [g["question"] for g in golden])
    live_vecs, golden_vecs = vectors[:len(live)], vectors[len(live):]
    rows = []
    for query, vec in zip(live, live_vecs, strict=True):
        scores = [cosine(vec, g) for g in golden_vecs]
        best = max(range(len(golden)), key=scores.__getitem__)
        rows.append({**query, "best_score": round(scores[best], 4),
                     "nearest_golden": golden[best]["id"],
                     "nearest_question": golden[best]["question"],
                     "covered": scores[best] >= threshold})
    return rows


def summarise(rows: list[dict], threshold: float, min_coverage: float) -> dict:
    covered = sum(1 for r in rows if r["covered"])
    by_kind: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_kind[r["kind"] or "unlabelled"].append(r["best_score"])
    gaps: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if not r["covered"]:
            gaps[r["topic"]].append(r)
    share = covered / len(rows)
    return {
        "queries": len(rows), "covered": covered, "coverage": round(share, 4),
        "similarity_threshold": threshold, "min_coverage": min_coverage,
        "passes": share >= min_coverage,
        "mean_best_score_by_kind": {k: round(statistics.mean(v), 4) for k, v in by_kind.items()},
        "uncovered_by_topic": {t: [{"id": r["id"], "text": r["text"],
                                    "best_score": r["best_score"],
                                    "nearest_golden": r["nearest_golden"]} for r in rs]
                               for t, rs in sorted(gaps.items())},
    }


def render(summary: dict, rows: list[dict], golden_count: int) -> str:
    out = [f"Query coverage: {summary['covered']} of {summary['queries']} live queries reach "
           f"similarity {summary['similarity_threshold']} against {golden_count} golden questions "
           f"= {summary['coverage']:.1%} (minimum {summary['min_coverage']:.0%}): "
           + ("PASS" if summary["passes"] else "FAIL"), ""]
    out.append("Mean best similarity by kind: " + ", ".join(
        f"{k} {v:.3f}" for k, v in summary["mean_best_score_by_kind"].items()))
    out.append("")
    out.append("| Id | Kind | Best score | Nearest golden | Covered | Query |")
    out.append("| --- | --- | ---: | --- | --- | --- |")
    for r in rows:
        out.append(f"| {r['id']} | {r['kind']} | {r['best_score']:.3f} | {r['nearest_golden']} | "
                   f"{'yes' if r['covered'] else 'NO'} | {r['text']} |")
    out.append("")
    if summary["uncovered_by_topic"]:
        out.append("Uncovered queries, grouped by topic:")
        for topic, items in summary["uncovered_by_topic"].items():
            out.append(f"  {topic} ({len(items)}): " + "; ".join(
                f"{i['id']} {i['best_score']:.2f}" for i in items))
    else:
        out.append("Uncovered queries: none")
    return "\n".join(out)


def calibration(rows: list[dict]) -> str:
    """Where do designed-to-be-covered and designed-to-be-uncovered queries land?"""
    scores: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        scores[r["kind"] or "unlabelled"].append(r["best_score"])
    out = ["Calibration (best similarity to a golden question, by how the query was designed):"]
    for kind, values in scores.items():
        out.append(f"  {kind:<11} n={len(values):<3} min {min(values):.3f}  "
                   f"median {statistics.median(values):.3f}  max {max(values):.3f}")
    if "paraphrase" in scores and "offtopic" in scores:
        midpoint = (statistics.median(scores["paraphrase"])
                    + statistics.median(scores["offtopic"])) / 2
        out.append(f"  suggested threshold (midpoint of the paraphrase and offtopic medians): "
                   f"{midpoint:.3f}")
    return "\n".join(out)


def main(argv: list[str] | None = None, embed: Embedder | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--live", type=Path, default=HERE / "live_queries.yaml")
    parser.add_argument("--golden", type=Path, default=HERE / "golden.yaml")
    parser.add_argument("--config", type=Path, default=HERE / "coverage.yaml")
    parser.add_argument("--ollama", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    parser.add_argument("--threshold", type=float, help="override similarity_threshold")
    parser.add_argument("--min-coverage", type=float, help="override min_coverage")
    parser.add_argument("--label", default="run", help="names the result file")
    parser.add_argument("--out-dir", type=Path, default=HERE / "results")
    parser.add_argument("--calibrate", action="store_true", help="print the calibration only")
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    threshold = args.threshold if args.threshold is not None else float(
        config.get("similarity_threshold", 0.75))
    min_coverage = args.min_coverage if args.min_coverage is not None else float(
        config.get("min_coverage", 0.85))
    golden, live = load_golden(args.golden), load_live(args.live)
    if embed is None:
        embed = ollama_embedder(args.ollama, config.get("embed_model", "nomic-embed-text"))
    try:
        rows = match(live, golden, embed, threshold)
    except OSError as exc:
        print(f"cannot compute embeddings ({exc}); is Ollama running at {args.ollama}?",
              file=sys.stderr)
        return 2

    if args.calibrate:
        print(calibration(rows))
        return 0
    summary = summarise(rows, threshold, min_coverage)
    print(render(summary, rows, len(golden)))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = args.out_dir / f"coverage-{args.label}.json"
    result.write_text(json.dumps({"summary": summary, "golden_questions": len(golden),
                                  "queries": rows}, indent=2), encoding="utf-8")
    print(f"\nresult written to {result}")
    return 0 if summary["passes"] else 1


if __name__ == "__main__":
    sys.exit(main())
