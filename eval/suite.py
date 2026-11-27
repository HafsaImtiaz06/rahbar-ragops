#!/usr/bin/env python3
"""The complete M2 evaluation suite in one command.

    python eval/suite.py --api http://<ingress-or-localhost:8000> [--host rahbar.local]

Needs: the API (end-to-end stage), Ollama with the pinned embedding model (embedding, retrieval
and noise stages), and the corpus PDFs in corpus/raw (offline index and faithfulness evidence).

Stages, run once per repetition (default 2, with the same seed):
  embedding   paraphrase vs unrelated pairs, embedding time per chunk, deterministic re-embedding
  retrieval   recall@3, recall@5, MRR, nDCG@5 on an in-memory index of the corpus
  noise       the same metrics with 4 irrelevant documents added, with typos in the queries, and both
  answer      the golden questions through the API: faithfulness, abstention correctness,
              citation validity, p50/p95 latency, and retrieval metrics as the API returned them

Results go to eval/results/<UTC timestamp>-eval.json plus a readable -summary.md. A run is only
marked "complete": true when every stage finished in every repetition. Exit codes: 0 complete,
2 incomplete or invalid input, 130 interrupted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx

import golden_set
import offline
from judge import LABELS, make_judge
from metrics import (
    compare_runs,
    cosine,
    determinism_report,
    embedding_report,
    mean,
    percentile,
    retrieval_metrics,
)
from run_eval import API_TIMEOUT_SECONDS, citation_counts, load_golden, recall_at_k

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STAGES = ("embedding", "retrieval", "noise", "answer")
MIN_PAIRS = 15
MIN_DISTRACTORS = 3


@dataclass
class Config:
    golden: Path = HERE / "golden.jsonl"
    pairs: Path = HERE / "embedding_pairs.jsonl"
    noise_dir: Path = HERE / "noise_docs"
    corpus: str = str(ROOT / "corpus" / "raw")
    ollama_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    api: str | None = None
    host: str | None = None
    chunk_size: int = 400
    chunk_overlap: int = 50
    seed: int = 1337
    repeat: int = 2
    judge: str = "ollama"
    judge_url: str = ""
    judge_model: str = "llama3.2:3b"
    judge_threshold: float = 0.6
    out_dir: Path = HERE / "results"
    profile: str = "full"            # "smoke" marks a plumbing check that the gate refuses
    use_cache: bool = False          # False: send follow_up=true so the API cache is bypassed
    typo_rate: float = 0.15
    determinism_tolerance: float = 1e-6
    stages: tuple[str, ...] = field(default_factory=lambda: STAGES)


class ApiAsker:
    def __init__(self, base: str, host: str | None, use_cache: bool,
                 client: httpx.Client | None = None) -> None:
        self.base = base.rstrip("/")
        self.use_cache = use_cache
        self.client = client or httpx.Client(timeout=API_TIMEOUT_SECONDS,
                                             headers={"Host": host} if host else {})

    def ask(self, question: str) -> tuple[dict, float]:
        started = time.perf_counter()
        response = self.client.post(f"{self.base}/api/ask",
                                    json={"question": question, "follow_up": not self.use_cache})
        response.raise_for_status()
        return response.json(), time.perf_counter() - started


def _stage(status: str, metrics: dict | None = None, error: str | None = None, **extra) -> dict:
    out = {"status": status, "metrics": metrics or {}}
    if error:
        out["error"] = error
    out.update(extra)
    return out


def _describe(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:300]


def load_pairs(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    kinds = [r["kind"] for r in rows]
    if kinds.count("paraphrase") < MIN_PAIRS or kinds.count("unrelated") < MIN_PAIRS:
        raise ValueError(f"{path}: need at least {MIN_PAIRS} paraphrase and {MIN_PAIRS} "
                         "unrelated pairs")
    return rows


def _by_type(rows: list[dict]) -> dict:
    groups: dict[str, list[float]] = {}
    for r in rows:
        if r["relevant"]:
            groups.setdefault(r["type"], []).append(
                recall_at_k(r["relevant"], r["retrieved"], 5) or 0.0)
    return {k: round(sum(v) / len(v), 4) for k, v in sorted(groups.items())}


# ---- stages -------------------------------------------------------------------------------

def stage_embedding(cfg: Config, embedder, pairs: list[dict], corpus_timing: dict | None) -> dict:
    texts = sorted({t for p in pairs for t in (p["a"], p["b"])})
    position = {t: i for i, t in enumerate(texts)}
    vectors, seconds = embedder.timed(texts, "document")
    sims = {"paraphrase": [], "unrelated": []}
    for p in pairs:
        sims[p["kind"]].append(cosine(vectors[position[p["a"]]], vectors[position[p["b"]]]))
    metrics = embedding_report(sims["paraphrase"], sims["unrelated"])
    metrics["embed_seconds_per_text"] = round(seconds / len(texts), 5)

    # Deterministic repeated-text comparison: embed the same texts again, in reverse order and
    # in a separate request, then compare vector by vector.
    probe = texts[:8]
    again = list(reversed(embedder.embed(list(reversed(probe)), "document")))
    metrics["determinism"] = determinism_report(vectors[:8], again, cfg.determinism_tolerance)

    if corpus_timing:
        metrics["embed_seconds_per_chunk"] = round(
            corpus_timing["seconds"] / corpus_timing["chunks"], 5)
        metrics["chunks_embedded"] = corpus_timing["chunks"]
        return _stage("complete", metrics)
    return _stage("partial", metrics,
                  "embedding time per chunk needs the corpus index (see the retrieval stage)")


def _scored(questions: list[dict]) -> list[dict]:
    return [q for q in questions if q.get("relevant")]


def stage_retrieval(index: offline.Index, questions: list[dict], qvecs: dict) -> dict:
    rows = [{"id": q["id"], "type": q.get("type", "ordinary"), "relevant": q["relevant"],
             "retrieved": index.search(qvecs[q["id"]], q["question"], 5)}
            for q in _scored(questions)]
    metrics = retrieval_metrics(rows)
    metrics["recall_at_5_by_type"] = _by_type(rows)
    return _stage("complete", metrics)


def stage_noise(cfg: Config, index: offline.Index, embedder, questions: list[dict],
                qvecs: dict, clean_metrics: dict) -> dict:
    d_chunks, d_names = offline.load_distractors(cfg.noise_dir, cfg.chunk_size, cfg.chunk_overlap)
    if len(d_names) < MIN_DISTRACTORS:
        return _stage("failed", error=f"need at least {MIN_DISTRACTORS} distractor documents in "
                                      f"{cfg.noise_dir}, found {len(d_names)}")
    d_vectors = embedder.embed([c.embed_text() for c in d_chunks], "document")
    noisy_index = index.with_extra(d_chunks, d_vectors)

    scored = _scored(questions)
    rng = random.Random(cfg.seed)
    typo_text = {q["id"]: offline.add_typos(q["question"], rng, cfg.typo_rate) for q in scored}
    ids = [q["id"] for q in scored]
    typo_vecs = dict(zip(ids, embedder.embed([typo_text[i] for i in ids], "query"), strict=True))

    def run(idx: offline.Index, typos: bool) -> dict:
        rows = []
        for q in scored:
            text = typo_text[q["id"]] if typos else q["question"]
            vec = typo_vecs[q["id"]] if typos else qvecs[q["id"]]
            rows.append({"relevant": q["relevant"], "retrieved": idx.search(vec, text, 5)})
        return retrieval_metrics(rows)

    conditions = {"clean": clean_metrics, "distractors": run(noisy_index, False),
                  "typos": run(index, True), "distractors_and_typos": run(noisy_index, True)}
    keys = ("recall_at_3", "recall_at_5", "mrr", "ndcg_at_5")
    deltas = {name: {k: round((m[k] or 0) - (clean_metrics[k] or 0), 4) for k in keys}
              for name, m in conditions.items() if name != "clean"}
    return _stage("complete", {
        "conditions": {k: {m: v[m] for m in keys} for k, v in conditions.items()},
        "delta_vs_clean": deltas,
        "distractor_documents": d_names,
        "distractor_chunks": len(d_chunks),
        "typo_rate": cfg.typo_rate,
    }, typo_examples=[{"original": q["question"], "typo": typo_text[q["id"]]}
                      for q in scored[:3]])


def judge_answers(rows: list[dict], chunks: list, judge) -> tuple[list[dict], int]:
    """Judge every answered row against the text of the chunks it cites. Returns the items and
    the number of answered rows that could not be judged (no evidence, or the judge failed)."""
    items: list[dict] = []
    unjudged = 0
    for row in rows:
        resp = row["response"]
        if not resp.get("answered_from_context"):
            continue
        refs = resp.get("citations") or resp.get("retrieved") or []
        evidence = offline.evidence_text(refs, chunks)
        item = {"id": row["id"], "question": row["question"], "answer": resp.get("answer", ""),
                "evidence": evidence[:2500], "judge_label": None, "judge_score": None}
        if evidence:
            try:
                verdict = judge(row["question"], item["answer"], evidence)
                if verdict["label"] in LABELS:
                    item["judge_label"] = verdict["label"]
                    item["judge_score"] = verdict.get("score")
            except Exception as exc:  # a judge failure must not look like a faithful answer
                item["judge_error"] = _describe(exc)
        if item["judge_label"] is None:
            unjudged += 1
        items.append(item)
    return items, unjudged


def stage_answer(cfg: Config, asker, questions: list[dict], chunks: list, judge) -> tuple[dict, list]:
    rows: list[dict] = []
    errors: list[dict] = []
    for q in questions:
        try:
            response, seconds = asker.ask(q["question"])
        except Exception as exc:
            errors.append({"id": q["id"], "error": _describe(exc)})
            continue
        rows.append({"id": q["id"], "question": q["question"], "type": q.get("type", "ordinary"),
                     "must_abstain": bool(q.get("must_abstain")), "relevant": q.get("relevant") or [],
                     "expect": q.get("expect"), "response": response,
                     "seconds": round(seconds, 3)})
        print(f"  {q['id']:<5}{seconds:>8.2f}s  {q['question'][:60]}", flush=True)

    if not rows:
        return _stage("failed", error="no question was answered", question_errors=errors), []

    retrieval_rows = [{"relevant": r["relevant"], "type": r["type"],
                       "retrieved": r["response"].get("retrieved", [])} for r in rows]
    api_retrieval = retrieval_metrics(retrieval_rows)
    api_retrieval["recall_at_5_by_type"] = _by_type(retrieval_rows)

    valid = total = 0
    for r in rows:
        v, t = citation_counts(r["response"])
        valid += v
        total += t
    must = [r for r in rows if r["must_abstain"]]
    normal = [r for r in rows if not r["must_abstain"]]
    expect_rows = [r for r in normal if r["expect"] and r["response"].get("answered_from_context")]
    seconds_all = [r["seconds"] for r in rows]
    answered = [r for r in rows if r["response"].get("answered_from_context")]

    items, unjudged = judge_answers(rows, chunks, judge)
    judged = [i for i in items if i["judge_label"] in LABELS]
    metrics = {
        "questions": len(rows),
        "citation_validity": round(valid / total, 4) if total else 1.0,
        "citations_checked": total,
        "abstention_correctness": round(
            sum(1 for r in must if not r["response"].get("answered_from_context")) / len(must), 4
        ) if must else None,
        "must_abstain_questions": len(must),
        "over_abstention_rate": round(
            sum(1 for r in normal if not r["response"].get("answered_from_context"))
            / len(normal), 4) if normal else None,
        "faithfulness": round(sum(1 for i in judged if i["judge_label"] == "faithful")
                              / len(judged), 4) if judged else None,
        "faithfulness_answers_judged": len(judged),
        "faithfulness_coverage": round(len(judged) / len(answered), 4) if answered else None,
        "expect_hit_rate": round(mean(
            1.0 if str(r["expect"]).lower() in r["response"].get("answer", "").lower() else 0.0
            for r in expect_rows) or 0.0, 4) if expect_rows else None,
        "latency_seconds_p50": round(percentile(seconds_all, 50) or 0.0, 3),
        "latency_seconds_p95": round(percentile(seconds_all, 95) or 0.0, 3),
        "cache_hits": sum(1 for r in rows if r["response"].get("cached")),
    }
    problems = []
    if errors:
        problems.append(f"{len(errors)} question(s) failed: "
                        + "; ".join(f"{e['id']} {e['error']}" for e in errors[:3]))
    if not answered:
        problems.append("no question was answered from context, so faithfulness is undefined")
    elif unjudged:
        problems.append(f"{unjudged} answered question(s) could not be judged (missing evidence "
                        "in the corpus index, or the judge failed)")
    stage = _stage("partial" if problems else "complete", metrics, "; ".join(problems) or None,
                   question_errors=errors, providers=sorted({r["response"].get("provider", "?")
                                                              for r in rows}))
    stage["api_retrieval"] = api_retrieval
    stage["per_question"] = [{
        "id": r["id"], "type": r["type"], "must_abstain": r["must_abstain"],
        "answered_from_context": r["response"].get("answered_from_context"),
        "recall_at_5": recall_at_k(r["relevant"], r["response"].get("retrieved", []), 5),
        "citations_valid": "{}/{}".format(*citation_counts(r["response"])),
        "cached": bool(r["response"].get("cached")), "seconds": r["seconds"]} for r in rows]
    return stage, items


# ---- one repetition -----------------------------------------------------------------------

def run_once(cfg: Config, questions: list[dict], pairs: list[dict], chunks: list, embedder,
             asker, judge) -> dict:
    wanted = set(cfg.stages)
    stages: dict[str, dict] = {}
    index = None
    timing = None
    index_error = None
    qvecs: dict = {}

    if wanted & {"embedding", "retrieval", "noise"}:
        try:
            if not chunks:
                raise RuntimeError(f"no chunks: the corpus at {cfg.corpus} is missing or empty")
            vectors, seconds = embedder.timed([c.embed_text() for c in chunks], "document")
            index = offline.Index(chunks, vectors)
            timing = {"seconds": seconds, "chunks": len(chunks)}
            scored = _scored(questions)
            vecs = embedder.embed([q["question"] for q in scored], "query")
            qvecs = {q["id"]: v for q, v in zip(scored, vecs, strict=True)}
        except Exception as exc:
            index_error = _describe(exc)

    def guarded(name: str, fn: Callable[[], dict]) -> None:
        if name not in wanted:
            stages[name] = _stage("skipped", error="stage not requested")
            return
        try:
            stages[name] = fn()
        except Exception as exc:
            stages[name] = _stage("failed", error=_describe(exc))

    guarded("embedding", lambda: stage_embedding(cfg, embedder, pairs, timing))
    retrieval_stage = None
    if index is not None:
        guarded("retrieval", lambda: stage_retrieval(index, questions, qvecs))
        retrieval_stage = stages["retrieval"]
    elif "retrieval" in wanted:
        stages["retrieval"] = _stage("failed", error=index_error or "no corpus index")
    else:
        stages["retrieval"] = _stage("skipped", error="stage not requested")

    if index is not None and retrieval_stage and retrieval_stage["status"] == "complete":
        guarded("noise", lambda: stage_noise(cfg, index, embedder, questions, qvecs,
                                             {k: retrieval_stage["metrics"][k] for k in
                                              ("questions", "recall_at_3", "recall_at_5", "mrr",
                                               "ndcg_at_5")}))
    else:
        stages["noise"] = _stage("failed" if "noise" in wanted else "skipped",
                                 error=index_error or "needs a completed retrieval stage")

    answer_items: list[dict] = []
    if "answer" in wanted:
        if asker is None:
            stages["answer"] = _stage("failed", error="no --api given")
        else:
            try:
                stages["answer"], answer_items = stage_answer(cfg, asker, questions, chunks, judge)
            except Exception as exc:
                stages["answer"] = _stage("failed", error=_describe(exc))
    else:
        stages["answer"] = _stage("skipped", error="stage not requested")

    metrics: dict = {}
    for name in ("embedding", "retrieval", "noise", "answer"):
        if stages[name]["metrics"]:
            key = "retrieval_offline" if name == "retrieval" else name
            metrics[key] = stages[name]["metrics"]
    if "api_retrieval" in stages["answer"]:
        metrics["retrieval_api"] = stages["answer"].pop("api_retrieval")

    return {"complete": all(s["status"] == "complete" for s in stages.values()),
            "stages": stages, "metrics": metrics, "answer_items": answer_items}


# ---- aggregation and reporting ------------------------------------------------------------

def average_runs(runs: list[dict]) -> dict:
    """Mean of numeric leaves across runs; booleans are ANDed; anything else comes from run 1."""
    first = runs[0]
    out: dict = {}
    for key, value in first.items():
        values = [r[key] for r in runs if key in r]
        if isinstance(value, dict):
            out[key] = average_runs(values)
        elif isinstance(value, bool):
            out[key] = all(values)
        elif isinstance(value, (int, float)):
            numeric = [v for v in values if isinstance(v, (int, float))]
            if all(isinstance(v, int) for v in numeric) and len(set(numeric)) == 1:
                out[key] = numeric[0]          # counts that never changed stay integers
            else:
                out[key] = round(sum(numeric) / len(numeric), 6)
        else:
            out[key] = value
    return out


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, timeout=5, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def build_results(cfg: Config, run_id: str, started: str, runs: list[dict], golden_rows: list[dict],
                  interrupted: bool = False) -> dict:
    stage_set_ok = set(cfg.stages) == set(STAGES)
    complete = (not interrupted and stage_set_ok and len(runs) == cfg.repeat
                and all(r["complete"] for r in runs))
    reasons = []
    if interrupted:
        reasons.append("the run was interrupted")
    if not stage_set_ok:
        reasons.append("only these stages were requested: " + ", ".join(cfg.stages))
    if len(runs) < cfg.repeat:
        reasons.append(f"{len(runs)} of {cfg.repeat} repetitions finished")
    for number, run in enumerate(runs, 1):
        for name, stage in run["stages"].items():
            if stage["status"] != "complete":
                reasons.append(f"run {number} stage {name}: {stage['status']}"
                               + (f" ({stage['error']})" if stage.get("error") else ""))

    comparable = [r["metrics"] for r in runs if r["metrics"]]
    results = {
        "schema_version": 1, "run_id": run_id, "profile": cfg.profile, "complete": complete,
        "incomplete_reasons": reasons, "started_at": started,
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": {
            "seed": cfg.seed, "repetitions_requested": cfg.repeat, "stages": list(cfg.stages),
            "golden": _rel(cfg.golden), "golden_sha256": hashlib.sha256(
                cfg.golden.read_bytes()).hexdigest(), "golden_questions": len(golden_rows),
            "corpus": cfg.corpus, "embed_model": cfg.embed_model, "ollama_url": cfg.ollama_url,
            "chunk_size": cfg.chunk_size, "chunk_overlap": cfg.chunk_overlap, "api": cfg.api,
            "cache_bypassed": not cfg.use_cache, "judge": f"{cfg.judge}:{cfg.judge_model}"
            if cfg.judge == "ollama" else f"lexical:{cfg.judge_threshold}",
            "git_commit": _git_commit(), "python": platform.python_version(),
            "platform": platform.platform()},
        "metrics": average_runs(comparable) if comparable else {},
        "metrics_note": "numeric metrics are the mean over the repetitions; see 'variation'",
        "variation": compare_runs(comparable) if len(comparable) > 1 else {},
        "answer_items": runs[0]["answer_items"] if runs else [],
        "runs": [{"index": i, "complete": r["complete"], "stages": r["stages"],
                  "metrics": r["metrics"]} for i, r in enumerate(runs, 1)],
        "manual_review": {
            "golden_needing_review": sum(1 for q in golden_rows if q.get("needs_manual_review")),
            "faithfulness_agreement": "pending: run eval/judge.py export, label, then agree"},
    }
    return results


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _fmt(value, digits: int = 3) -> str:
    return "n/a" if value is None else (f"{value:.{digits}f}" if isinstance(value, float)
                                        else str(value))


def render_summary(results: dict) -> str:
    m = results["metrics"]
    cfg = results["config"]
    lines = [f"# Evaluation summary {results['run_id']}", ""]
    status = "COMPLETE" if results["complete"] else "INCOMPLETE: not a valid evaluation result"
    lines += [f"- Status: **{status}**", f"- Profile: {results['profile']}"
              + ("  (plumbing check only; the release gate refuses it)"
                 if results["profile"] != "full" else ""),
              f"- Golden set: {cfg['golden']} ({cfg['golden_questions']} questions), seed "
              f"{cfg['seed']}, {len(results['runs'])} of {cfg['repetitions_requested']} "
              "repetitions",
              f"- Embedding model {cfg['embed_model']}, chunks {cfg['chunk_size']}:"
              f"{cfg['chunk_overlap']}, judge {cfg['judge']}, API cache bypassed: "
              f"{cfg['cache_bypassed']}", ""]
    for reason in results["incomplete_reasons"]:
        lines.append(f"- NOT COMPLETE: {reason}")
    if results["incomplete_reasons"]:
        lines.append("")

    e = m.get("embedding", {})
    d = e.get("determinism", {})
    lines += ["## Embedding", "",
              f"- Paraphrase pairs ({_fmt(e.get('paraphrase_pairs'), 0)}): mean cosine "
              f"{_fmt(e.get('paraphrase_mean_cosine'))}",
              f"- Unrelated pairs ({_fmt(e.get('unrelated_pairs'), 0)}): mean cosine "
              f"{_fmt(e.get('unrelated_mean_cosine'))}",
              f"- Similarity gap {_fmt(e.get('similarity_gap'))}; overlap fraction "
              f"{_fmt(e.get('overlap_fraction'))} (0 means fully separable)",
              f"- Embedding time per chunk {_fmt(e.get('embed_seconds_per_chunk'), 4)} s "
              f"({_fmt(e.get('chunks_embedded'), 0)} chunks)",
              f"- Same text embedded twice: deterministic = {d.get('deterministic', 'n/a')} "
              f"(min cosine {_fmt(d.get('min_cosine'), 8)})", ""]

    lines += ["## Retrieval", "", "| Source | recall@3 | recall@5 | MRR | nDCG@5 | questions |",
              "| --- | --- | --- | --- | --- | --- |"]
    for label, key in (("API as deployed", "retrieval_api"), ("Offline cosine index",
                                                              "retrieval_offline")):
        r = m.get(key)
        if r:
            lines.append(f"| {label} | {_fmt(r['recall_at_3'])} | {_fmt(r['recall_at_5'])} | "
                         f"{_fmt(r['mrr'])} | {_fmt(r['ndcg_at_5'])} | {_fmt(r['questions'], 0)} |")
    lines.append("")

    n = m.get("noise")
    if n:
        lines += ["## Noise robustness (offline index)", "",
                  f"Distractor documents: {', '.join(n['distractor_documents'])}; typo rate "
                  f"{n['typo_rate']}.", "",
                  "| Condition | recall@3 | recall@5 | MRR | nDCG@5 | change in recall@5 |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for name, c in n["conditions"].items():
            change = "" if name == "clean" else f"{n['delta_vs_clean'][name]['recall_at_5']:+.4f}"
            lines.append(f"| {name} | {_fmt(c['recall_at_3'])} | {_fmt(c['recall_at_5'])} | "
                         f"{_fmt(c['mrr'])} | {_fmt(c['ndcg_at_5'])} | {change} |")
        lines.append("")

    a = m.get("answer")
    if a:
        lines += ["## Answer quality (through the API)", "",
                  f"- Faithfulness {_fmt(a.get('faithfulness'))} over "
                  f"{_fmt(a.get('faithfulness_answers_judged'), 0)} answers, judge {cfg['judge']}, "
                  f"coverage {_fmt(a.get('faithfulness_coverage'))}",
                  f"- Abstention correctness {_fmt(a.get('abstention_correctness'))} on "
                  f"{_fmt(a.get('must_abstain_questions'), 0)} unanswerable questions; "
                  f"over-abstention on answerable questions {_fmt(a.get('over_abstention_rate'))}",
                  f"- Citation validity {_fmt(a.get('citation_validity'))} over "
                  f"{_fmt(a.get('citations_checked'), 0)} citations",
                  f"- Latency p50 {_fmt(a.get('latency_seconds_p50'), 2)} s, p95 "
                  f"{_fmt(a.get('latency_seconds_p95'), 2)} s; cache hits "
                  f"{_fmt(a.get('cache_hits'), 0)}", ""]

    var = results.get("variation") or {}
    if var:
        moved = {k: v for k, v in var.items() if not v["identical"]}
        lines += ["## Variation between repetitions", "",
                  f"{len(var) - len(moved)} of {len(var)} numeric metrics were identical.", ""]
        for key, v in sorted(moved.items(), key=lambda kv: -kv[1]["range"])[:12]:
            lines.append(f"- {key}: {v['values']} (range {v['range']})")
        lines.append("")

    lines += ["## Manual review", "",
              f"- {results['manual_review']['golden_needing_review']} synthetic golden question(s) "
              "still need a person to confirm their ground truth against the PDFs.",
              f"- Faithfulness judge vs people: {results['manual_review']['faithfulness_agreement']}"]
    return "\n".join(lines) + "\n"


def write_results(cfg: Config, results: dict) -> tuple[Path, Path]:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{results['run_id']}-eval"
    json_path = cfg.out_dir / f"{stem}.json"
    md_path = cfg.out_dir / f"{stem}-summary.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    md_path.write_text(render_summary(results), encoding="utf-8")
    return json_path, md_path


def parse_args(argv: list[str] | None = None) -> Config:
    p = argparse.ArgumentParser(description="Run the complete M2 evaluation suite.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    d = Config()
    p.add_argument("--api", help="API base URL for the answer stage, e.g. http://localhost:8000")
    p.add_argument("--host", help="Host header when going through the Ingress (rahbar.local)")
    p.add_argument("--golden", type=Path, default=d.golden)
    p.add_argument("--pairs", type=Path, default=d.pairs)
    p.add_argument("--corpus", default=d.corpus)
    p.add_argument("--ollama-url", default=d.ollama_url)
    p.add_argument("--embed-model", default=d.embed_model)
    p.add_argument("--chunk-size", type=int, default=d.chunk_size)
    p.add_argument("--chunk-overlap", type=int, default=d.chunk_overlap)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--repeat", type=int, default=d.repeat)
    p.add_argument("--judge", choices=["ollama", "lexical"], default=d.judge)
    p.add_argument("--judge-url", default="", help="defaults to --ollama-url")
    p.add_argument("--judge-model", default=d.judge_model)
    p.add_argument("--judge-threshold", type=float, default=d.judge_threshold)
    p.add_argument("--out-dir", type=Path, default=d.out_dir)
    p.add_argument("--profile", choices=["full", "smoke"], default=d.profile)
    p.add_argument("--use-cache", action="store_true",
                   help="let the API cache answer (default: bypass it with follow_up=true)")
    p.add_argument("--stages", default=",".join(STAGES),
                   help="comma-separated subset; anything less than all four is never 'complete'")
    a = p.parse_args(argv)
    return Config(golden=a.golden, pairs=a.pairs, corpus=a.corpus, ollama_url=a.ollama_url,
                  embed_model=a.embed_model, api=a.api, host=a.host, chunk_size=a.chunk_size,
                  chunk_overlap=a.chunk_overlap, seed=a.seed, repeat=a.repeat, judge=a.judge,
                  judge_url=a.judge_url or a.ollama_url, judge_model=a.judge_model,
                  judge_threshold=a.judge_threshold, out_dir=a.out_dir, profile=a.profile,
                  use_cache=a.use_cache,
                  stages=tuple(s.strip() for s in a.stages.split(",") if s.strip()))


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv)
    random.seed(cfg.seed)
    unknown = set(cfg.stages) - set(STAGES)
    if unknown or cfg.repeat < 1:
        print(f"invalid options: unknown stages {sorted(unknown)} or repeat < 1")
        return 2

    questions = load_golden(cfg.golden)
    if cfg.profile == "full":
        problems = (golden_set.validate(questions) if cfg.golden.suffix == ".jsonl"
                    else ["the full profile needs a .jsonl golden set"])
        if problems:
            print("golden set invalid:\n  - " + "\n  - ".join(problems))
            return 2
    pairs = load_pairs(cfg.pairs)

    embedder = offline.Embedder(cfg.ollama_url, cfg.embed_model)
    asker = ApiAsker(cfg.api, cfg.host, cfg.use_cache) if cfg.api else None
    judge = make_judge(cfg.judge, url=cfg.judge_url, model=cfg.judge_model, seed=cfg.seed,
                       threshold=cfg.judge_threshold)
    try:
        chunks = offline.build_chunks(cfg.corpus, cfg.chunk_size, cfg.chunk_overlap)
    except Exception as exc:   # missing corpus must surface as incomplete stages, not a crash
        print(f"corpus not usable: {_describe(exc)}")
        chunks = []

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    started = datetime.now(UTC).isoformat(timespec="seconds")
    runs: list[dict] = []
    interrupted = False
    try:
        for number in range(1, cfg.repeat + 1):
            print(f"\n=== repetition {number} of {cfg.repeat} (seed {cfg.seed})", flush=True)
            runs.append(run_once(cfg, questions, pairs, chunks, embedder, asker, judge))
    except KeyboardInterrupt:
        interrupted = True
        print("\ninterrupted: saving what exists, marked incomplete")

    results = build_results(cfg, run_id, started, runs, questions, interrupted)
    json_path, md_path = write_results(cfg, results)
    print(f"\nresults: {json_path}\nsummary: {md_path}")
    if interrupted:
        return 130
    if not results["complete"]:
        print("\nINCOMPLETE RUN:\n  - " + "\n  - ".join(results["incomplete_reasons"]))
        return 2
    print("\ncomplete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
