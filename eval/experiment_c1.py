#!/usr/bin/env python3
"""C1 experiment: one baseline configuration against one candidate configuration.

The variable is the chunking configuration (size:overlap in tokens). Baseline defaults to the one
in k8s/base/configmap.yaml (400:50); the candidate defaults to 200:30, a hypothesis to test, not a
result: smaller chunks may help exact-term questions and may split long definitions.

    python eval/experiment_c1.py --corpus corpus/raw --ollama-url http://localhost:11434 \\
        --golden eval/golden.jsonl --baseline 400:50 --candidate 200:30

Three metric groups are compared:
  embedding   pair similarity (the model is the same, so it cannot change), chunks per corpus,
              token amplification and embedding time per chunk, per configuration
  retrieval   recall@3/@5, MRR, nDCG@5 (eval/chunk_sweep.py's evaluate_config, offline)
  end-to-end  the answer stage through an API whose index was built with that configuration

End-to-end numbers need a deployed API per configuration, so they are only produced for sides
given with --api-baseline / --api-candidate, and the tool refuses an API whose index has a
different chunk count than the configuration would produce. Anything not measured is reported
as pending, with the command to measure it. Nothing is estimated.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx

import chunk_sweep
import offline
import suite
from judge import make_judge
from run_eval import load_golden

HERE = Path(__file__).resolve().parent
RETRIEVAL_KEYS = ("recall_at_3", "recall_at_5", "mrr", "ndcg_at_5")
E2E_KEYS = ("faithfulness", "abstention_correctness", "citation_validity", "latency_seconds_p50",
            "latency_seconds_p95", "over_abstention_rate")


def parse_config(text: str) -> tuple[int, int]:
    size, overlap = (int(x) for x in text.split(":"))
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError(f"bad chunk configuration {text!r}")
    return size, overlap


def delta(base: float | None, cand: float | None) -> float | None:
    return None if base is None or cand is None else round(cand - base, 4)


def compare(baseline: dict, candidate: dict, keys: tuple[str, ...]) -> dict:
    return {k: {"baseline": baseline.get(k), "candidate": candidate.get(k),
                "delta": delta(baseline.get(k), candidate.get(k))} for k in keys}


def api_index_chunks(api: str, host: str | None) -> int:
    headers = {"Host": host} if host else {}
    r = httpx.get(f"{api.rstrip('/')}/readyz", headers=headers, timeout=30)
    r.raise_for_status()
    return int(r.json()["checks"]["index"]["chunks"])


def run_e2e(args, side: str, api: str, size: int, overlap: int, questions: list[dict],
            chunks: list) -> dict:
    """The answer stage for one side. Refuses an API that does not serve this configuration."""
    served = api_index_chunks(api, args.host)
    if served != len(chunks):
        return {"status": "failed", "error": f"{side}: the API index has {served} chunks but "
                f"{size}:{overlap} on this corpus gives {len(chunks)}; the API is not serving "
                "this configuration (re-index, or point at the right API)"}
    cfg = suite.Config(api=api, host=args.host, judge=args.judge, judge_url=args.ollama_url,
                       judge_model=args.judge_model, chunk_size=size, chunk_overlap=overlap)
    judge = make_judge(args.judge, url=args.ollama_url, model=args.judge_model, seed=args.seed)
    stage, _ = suite.stage_answer(cfg, suite.ApiAsker(api, args.host, False), questions, chunks,
                                  judge)
    stage.pop("api_retrieval", None)
    return {"status": stage["status"], "error": stage.get("error"), "metrics": stage["metrics"]}


def pending_commands(args, side: str, size: int, overlap: int) -> list[str]:
    return [
        f"1. Make the API serve {size}:{overlap}: set CHUNK_SIZE={size} and CHUNK_OVERLAP={overlap} "
        "in the rahbar-config ConfigMap, then `make index` and wait for the Job to finish.",
        f"2. python eval/experiment_c1.py --corpus {args.corpus} --ollama-url {args.ollama_url} "
        f"--golden {args.golden} --baseline {args.baseline} --candidate {args.candidate} "
        f"--api-{side} http://<api> --merge <previous c1 json>"]


def render(result: dict) -> str:
    out = [f"# C1 experiment {result['run_id']}", "",
           f"- Status: **{result['status'].upper()}**",
           f"- Baseline {result['baseline']['config']} vs candidate {result['candidate']['config']}"
           f"; model {result['embed_model']}; golden {result['golden']} "
           f"({result['questions_scored']} scored questions)", ""]
    for note in result["pending"]:
        out.append(f"- PENDING: {note}")
    out.append("" if result["pending"] else "")
    emb = result["embedding"]
    out += ["## Embedding", "",
            f"- Pair similarity (same model, so identical for both): paraphrase "
            f"{emb['pairs'].get('paraphrase_mean_cosine')}, unrelated "
            f"{emb['pairs'].get('unrelated_mean_cosine')}, gap {emb['pairs'].get('similarity_gap')}",
            "", "| Metric | Baseline | Candidate | Change |", "| --- | --- | --- | --- |"]
    for k, v in emb["per_config"].items():
        out.append(f"| {k} | {v['baseline']} | {v['candidate']} | {v['delta']} |")
    out += ["", "## Retrieval (offline cosine index)", "",
            "| Metric | Baseline | Candidate | Change |", "| --- | --- | --- | --- |"]
    for k, v in result["retrieval"].items():
        out.append(f"| {k} | {v['baseline']} | {v['candidate']} | {v['delta']} |")
    out += ["", "## End to end (through the API)", ""]
    if result["end_to_end"]:
        out += ["| Metric | Baseline | Candidate | Change |", "| --- | --- | --- | --- |"]
        for k, v in result["end_to_end"].items():
            out.append(f"| {k} | {v['baseline']} | {v['candidate']} | {v['delta']} |")
    else:
        out.append("Not measured: see PENDING above.")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=str(HERE.parent / "corpus" / "raw"))
    ap.add_argument("--golden", default=str(HERE / "golden.jsonl"))
    ap.add_argument("--pairs", default=str(HERE / "embedding_pairs.jsonl"))
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--embed-model", default="nomic-embed-text")
    ap.add_argument("--baseline", default="400:50")
    ap.add_argument("--candidate", default="200:30")
    ap.add_argument("--api-baseline")
    ap.add_argument("--api-candidate")
    ap.add_argument("--host")
    ap.add_argument("--merge", type=Path, help="an earlier C1 result: reuse its end-to-end side(s)")
    ap.add_argument("--judge", choices=["ollama", "lexical"], default="ollama")
    ap.add_argument("--judge-model", default="llama3.2:3b")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out-dir", type=Path, default=HERE / "results")
    args = ap.parse_args(argv)

    sizes = {"baseline": parse_config(args.baseline), "candidate": parse_config(args.candidate)}
    if sizes["baseline"] == sizes["candidate"]:
        print("baseline and candidate are the same configuration")
        return 2

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    questions = [q for q in load_golden(Path(args.golden)) if q.get("relevant")]
    pending: list[str] = []
    try:
        settings = chunk_sweep.make_settings(args.corpus, args.embed_model, args.ollama_url)
        docs = offline.load_documents(args.corpus)
        if not docs:
            raise RuntimeError(f"no documents in {args.corpus}")
        original = sum(chunk_sweep.count(pg.text) for d in docs for pg in d.pages)
        qvecs = chunk_sweep.embed_queries(questions, args.ollama_url, args.embed_model)
        rows = {side: chunk_sweep.evaluate_config(docs, questions, qvecs, *sizes[side], settings,
                                                  original) for side in sizes}
        cfg = suite.Config(ollama_url=args.ollama_url, embed_model=args.embed_model)
        pairs_stage = suite.stage_embedding(
            cfg, offline.Embedder(args.ollama_url, args.embed_model),
            suite.load_pairs(Path(args.pairs)), None)
        pair_metrics = {k: pairs_stage["metrics"][k] for k in (
            "paraphrase_mean_cosine", "unrelated_mean_cosine", "similarity_gap")}
    except Exception as exc:
        # Nothing measured: report the experiment as pending, with the commands, and no numbers.
        result = {"run_id": run_id, "status": "pending", "baseline": {"config": args.baseline},
                  "candidate": {"config": args.candidate}, "embed_model": args.embed_model,
                  "golden": args.golden, "questions_scored": len(questions),
                  "pending": [f"nothing was measured: {type(exc).__name__}: {exc}",
                              "needs Ollama with the embedding model, and the corpus PDFs in "
                              f"{args.corpus}; then run: python eval/experiment_c1.py "
                              f"--corpus {args.corpus} --ollama-url {args.ollama_url} "
                              f"--baseline {args.baseline} --candidate {args.candidate}"],
                  "embedding": {"pairs": {}, "per_config": {}}, "retrieval": {},
                  "end_to_end": {}}
        return _write(args, result)

    e2e: dict[str, dict] = {}
    previous = json.loads(args.merge.read_text(encoding="utf-8")) if args.merge else {}
    for side, api in (("baseline", args.api_baseline), ("candidate", args.api_candidate)):
        size, overlap = sizes[side]
        if api:
            chunks = offline.chunk_documents(docs, size, overlap)
            e2e[side] = run_e2e(args, side, api, size, overlap, questions, chunks)
        elif previous.get("sides", {}).get(side, {}).get("end_to_end", {}).get("metrics") and \
                previous["sides"][side]["config"] == f"{size}:{overlap}":
            e2e[side] = previous["sides"][side]["end_to_end"]
        else:
            e2e[side] = {"status": "pending"}
            pending.append(f"end-to-end for {side} ({size}:{overlap}) not measured. "
                           + " ".join(pending_commands(args, side, size, overlap)))
    for side, item in e2e.items():
        if item["status"] not in ("complete", "pending"):
            pending.append(f"end-to-end for {side} is {item['status']}: {item.get('error')}")

    both = all(e2e[s].get("status") == "complete" for s in e2e)
    result = {
        "run_id": run_id, "status": "complete" if both and not pending else "pending",
        "baseline": {"config": args.baseline}, "candidate": {"config": args.candidate},
        "embed_model": args.embed_model, "golden": args.golden, "questions_scored": len(questions),
        "pending": pending,
        "sides": {s: {"config": f"{sizes[s][0]}:{sizes[s][1]}", "retrieval": rows[s],
                      "end_to_end": e2e[s]} for s in sizes},
        "embedding": {"pairs": pair_metrics, "per_config": compare(
            rows["baseline"], rows["candidate"], ("chunks", "amplification",
                                                  "embed_seconds_per_chunk"))},
        "retrieval": compare(rows["baseline"], rows["candidate"], RETRIEVAL_KEYS),
        "end_to_end": compare(e2e["baseline"].get("metrics", {}), e2e["candidate"].get(
            "metrics", {}), E2E_KEYS) if both else {},
    }
    return _write(args, result)


def _write(args, result: dict) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{result['run_id']}-c1-experiment"
    (args.out_dir / f"{stem}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.out_dir / f"{stem}.md").write_text(render(result), encoding="utf-8")
    print(render(result))
    print(f"written: {args.out_dir / stem}.json and .md")
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
