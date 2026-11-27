#!/usr/bin/env python3
"""Offline chunk-size experiment (section 3.1 asks you to try at least three).

    python eval/chunk_sweep.py --corpus corpus/raw --golden eval/golden.jsonl \\
        --configs 200:30,400:50,800:100 --ollama-url http://localhost:11434

For each size:overlap it chunks the corpus with the REAL indexer code, embeds the chunks and the
golden questions with the pinned embedding model, ranks by cosine similarity in memory (no
Chroma, no cluster) and reports recall@3 / recall@5 / MRR / nDCG@5 overall and by question type,
the token amplification and the embedding time per chunk. Needs only Ollama running with the
embedding model pulled. Paste the table into docs/PIPELINE.md.

evaluate_config() is also what eval/experiment_c1.py calls, so the sweep and the C1 experiment
cannot disagree.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "indexer"))
sys.path.insert(0, str(ROOT / "eval"))

import httpx  # noqa: E402
from pipeline.embed import embed, prepare  # noqa: E402
from pipeline.config import IndexerSettings  # noqa: E402
from pipeline.tokens import count  # noqa: E402

import offline  # noqa: E402
from metrics import cosine as cos  # noqa: E402,F401  (kept: other tools import cos from here)
from metrics import retrieval_metrics  # noqa: E402
from run_eval import load_golden, recall_at_k  # noqa: E402


def make_settings(corpus: str, embed_model: str, ollama_url: str) -> IndexerSettings:
    return IndexerSettings(embed_model=embed_model, chroma_host="", chroma_port=0, collection="",
                           corpus_dir=corpus, chunk_size=0, chunk_overlap=0,
                           ollama_url=ollama_url)


def embed_queries(questions: list[dict], ollama_url: str, embed_model: str) -> list[list[float]]:
    vectors = []
    with httpx.Client(base_url=ollama_url, timeout=300) as client:
        for q in questions:
            r = client.post("/api/embed", json={"model": embed_model, "input": [
                prepare(embed_model, q["question"], "query")]})
            r.raise_for_status()
            vectors.append(r.json()["embeddings"][0])
    return vectors


def evaluate_config(docs, questions: list[dict], qvecs: list[list[float]], size: int,
                    overlap: int, settings: IndexerSettings, original_tokens: int) -> dict:
    """Chunk, embed and score one size:overlap. Returns one row of the sweep table."""
    chunks = offline.chunk_documents(docs, size, overlap)
    started = time.perf_counter()
    vecs = embed([c.embed_text() for c in chunks], settings)
    embed_seconds = time.perf_counter() - started
    index = offline.Index(chunks, vecs)

    rows = []
    for q, qv in zip(questions, qvecs, strict=True):
        rows.append({"type": q.get("type", "ordinary"), "relevant": q["relevant"],
                     "retrieved": index.search(qv, q["question"], 5)})
    scored = retrieval_metrics(rows)
    by_type: dict[str, list[float]] = {}
    for r in rows:
        by_type.setdefault(r["type"], []).append(recall_at_k(r["relevant"], r["retrieved"], 5)
                                                 or 0.0)
    return {
        "size": size, "overlap": overlap, "chunks": len(chunks),
        "amplification": round(sum(count(c.text) for c in chunks) / max(original_tokens, 1), 3),
        "recall_at_3": round(scored["recall_at_3"], 3), "recall_at_5": round(scored["recall_at_5"], 3),
        "mrr": round(scored["mrr"], 3), "ndcg_at_5": round(scored["ndcg_at_5"], 3),
        "embed_seconds_per_chunk": round(embed_seconds / max(len(chunks), 1), 5),
        "by_type": {k: round(sum(v) / len(v), 3) for k, v in sorted(by_type.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(ROOT / "corpus" / "raw"))
    ap.add_argument("--golden", default=str(ROOT / "eval" / "golden.yaml"))
    ap.add_argument("--configs", default="200:30,400:50,800:100")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--embed-model", default="nomic-embed-text")
    ap.add_argument("--out", default=str(ROOT / "eval" / "chunk_sweep.json"))
    a = ap.parse_args()

    questions = [q for q in load_golden(Path(a.golden)) if q.get("relevant")]
    settings = make_settings(a.corpus, a.embed_model, a.ollama_url)
    docs = offline.load_documents(a.corpus)
    original = sum(count(pg.text) for d in docs for pg in d.pages)
    qvecs = embed_queries(questions, a.ollama_url, a.embed_model)

    rows = []
    for cfg in a.configs.split(","):
        size, overlap = (int(x) for x in cfg.split(":"))
        rows.append(evaluate_config(docs, questions, qvecs, size, overlap, settings, original))
        print(json.dumps(rows[-1]), flush=True)
    Path(a.out).write_text(json.dumps(rows, indent=2))
    print("\n| size:overlap | chunks | amplification | recall@3 | recall@5 | MRR | nDCG@5 "
          "| s/chunk | by type (recall@5) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['size']}:{r['overlap']} | {r['chunks']} | {r['amplification']} | "
              f"{r['recall_at_3']} | {r['recall_at_5']} | {r['mrr']} | {r['ndcg_at_5']} | "
              f"{r['embed_seconds_per_chunk']} | {r['by_type']} |")


if __name__ == "__main__":
    main()
