#!/usr/bin/env python3
"""Ask the running API the chosen live queries and show what it retrieves (RAGOps E4).

Use it to find which uncovered topics the corpus can answer, and the source and section to put in
`relevant:` for a new golden question.

    python eval/probe_live.py q11 q14 q15 --label e4
Needs the API on localhost:8000 (kubectl port-forward svc/api 8000:8000 -n rahbar).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def ask(api: str, question: str, timeout: float = 300.0) -> dict:
    request = urllib.request.Request(
        api.rstrip("/") + "/api/ask", data=json.dumps({"question": question}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def summarise(query: dict, reply: dict) -> dict:
    return {"id": query["id"], "question": query["text"], "answer": reply.get("answer"),
            "answered_from_context": reply.get("answered_from_context"),
            "confidence": reply.get("confidence"),
            "retrieved": [{"source": r["source"], "section": r["section"], "page": r["page"],
                           "score": r["score"]} for r in reply.get("retrieved", [])[:3]]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("ids", nargs="+", help="live query ids, for example q11 q14")
    parser.add_argument("--live", type=Path, default=HERE / "live_queries.yaml")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--label", default="probe")
    parser.add_argument("--out-dir", type=Path, default=HERE / "results")
    args = parser.parse_args(argv)
    queries = {q["id"]: q for q in
               (yaml.safe_load(args.live.read_text(encoding="utf-8")) or {})["queries"]}
    unknown = [i for i in args.ids if i not in queries]
    if unknown:
        print(f"unknown query ids: {', '.join(unknown)}", file=sys.stderr)
        return 2
    results = []
    for query_id in args.ids:
        query = queries[query_id]
        try:
            row = summarise(query, ask(args.api, query["text"]))
        except OSError as exc:
            print(f"{query_id}: the API did not answer ({exc})", file=sys.stderr)
            return 2
        results.append(row)
        print(f"\n{row['id']}: {row['question']}")
        print(f"  answer: {row['answer']}  (from context: {row['answered_from_context']}, "
              f"confidence: {row['confidence']})")
        for item in row["retrieved"]:
            print(f"  {item['score']:.3f}  {item['source']}  p.{item['page']}  "
                  f"{item['section']}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"probe-{args.label}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nresult written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
