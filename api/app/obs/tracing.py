"""D1 - Trace record: one JSON line per /api/ask request.

* trace_id == the request_id the API already generates (X-Request-ID), so every AskResponse already
  carries the id you give to the trace tool.
* Written to stdout (kubectl logs, F4) AND to $TRACE_DIR/traces-<pod>.jsonl on a PVC (report scripts).
  The API runs with a read-only root filesystem, so TRACE_DIR must be a mounted volume (see k8s/patches).
* Tracing must never break a user request: file errors are swallowed and reported on stderr.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_TOP1: Any = None
_INJ: Any = None
try:  # extra Prometheus series on the EXISTING /metrics endpoint (D3 / F4); optional
    from prometheus_client import Counter, Histogram

    _TOP1 = Histogram("rahbar_trace_top1_similarity", "Top-1 retrieval similarity per query",
                      buckets=[i / 10 for i in range(1, 11)])
    _INJ = Counter("rahbar_trace_injection_flagged", "Queries with at least one injection guardrail action")
except Exception:  # not installed, or already registered
    _TOP1 = _INJ = None

_LOCK = threading.Lock()
_CURRENT: ContextVar["Trace | None"] = ContextVar("rahbar_current_trace", default=None)


def current_trace() -> "Trace | None":
    """The trace of the request being handled in this context (None outside a traced request)."""
    return _CURRENT.get()


def record_prompt(prompt: str, model: str | None = None) -> None:
    """Call this where the prompt string is built/sent (services/prompt.py or the provider).
    Retries overwrite it, so the trace keeps the prompt of the LAST generation attempt."""
    t = _CURRENT.get()
    if t is not None:
        t.set_prompt(prompt, model=model)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _plain(x):
    """pydantic models / dataclasses -> dict, everything else unchanged."""
    if hasattr(x, "model_dump"):
        return x.model_dump()
    if hasattr(x, "__dict__") and not isinstance(x, dict):
        return dict(vars(x))
    return x


def is_injection_action(a: dict) -> bool:
    return "injection" in str(a.get("rail", "")).lower() and a.get("action") != "pass"


class Trace:
    def __init__(self, user_query: str, trace_id: str | None = None, trace_dir: str | None = None):
        self._dir: str = trace_dir or os.environ.get("TRACE_DIR") or "/data/traces"
        self._t0 = time.perf_counter()
        self.rec: dict = {
            "trace_id": trace_id or uuid.uuid4().hex,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "user_query": user_query,
            "rewritten_query": None,
            "embedding": {"model": os.getenv("EMBED_MODEL", "unknown"), "version": None},
            "collection": os.getenv("CHROMA_COLLECTION", "unknown"),
            "retrieved": [],            # rank, chunk_id, doc_id(source), section, page, doc_version, superseded, score, text
            "filters": {},
            "reranked_order": [],       # chunk ids as they appear in the prompt
            "prompt": None,             # full prompt so the trace tool can rebuild it exactly
            "prompt_hash": None,
            "llm": {"provider": os.getenv("LLM_PROVIDER", "unknown"), "model": os.getenv("LLM_MODEL", "unknown"),
                    "params": {"temperature": os.getenv("TEMPERATURE"), "top_k": os.getenv("TOP_K"),
                               "relevance_floor": os.getenv("RELEVANCE_FLOOR")}},
            "response": None,
            "citations": [],
            "guardrail_actions": [],
            "latency_ms": {},
            "cache": "miss",            # "hit" | "miss"
            "abstained": False,
            "outcome": None,            # answered_from_context-ok | below_relevance_floor | model_declined | fallback | extractive_policy_rule
            "release": os.getenv("RELEASE_TRACK", "live"),
            "pod": os.getenv("HOSTNAME", socket.gethostname()),
        }

    @contextmanager
    def activate(self):
        token = _CURRENT.set(self)
        try:
            yield self
        finally:
            _CURRENT.reset(token)

    def set_outcome(self, reason: str | None) -> None:
        self.rec["outcome"] = reason or "answered"

    # ---- builders ---------------------------------------------------------
    @contextmanager
    def stage(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.rec["latency_ms"][name] = round(
                self.rec["latency_ms"].get(name, 0) + (time.perf_counter() - t) * 1000, 2)

    def set_collection(self, collection: str, embed_model: str | None = None, embed_version: str | None = None):
        """Call with the collection the retriever ACTUALLY resolved (e.g. rahbar-20261120T1012 behind the
        rahbar-active pointer), not just the env var. Version defaults to the collection name."""
        self.rec["collection"] = collection
        if embed_model:
            self.rec["embedding"]["model"] = embed_model
        self.rec["embedding"]["version"] = embed_version or collection

    def set_cache(self, hit: bool):
        self.rec["cache"] = "hit" if hit else "miss"

    def set_retrieved(self, hits, filters: dict | None = None) -> None:
        out = []
        for i, h in enumerate(map(_plain, hits)):
            out.append({
                "rank": i + 1,
                "chunk_id": h.get("chunk_id"),
                "doc_id": h.get("doc_id") or h.get("source"),
                "section": h.get("section"),
                "page": h.get("page"),
                "doc_version": h.get("doc_version"),
                "superseded": h.get("superseded"),
                "score": round(float(h.get("score", 0.0)), 5),
                "text": h.get("text", ""),
            })
        self.rec["retrieved"] = out
        if filters is not None:
            self.rec["filters"] = filters

    def set_reranked(self, chunk_ids: list) -> None:
        self.rec["reranked_order"] = list(chunk_ids)

    def set_rewritten(self, q: str) -> None:
        self.rec["rewritten_query"] = q

    def set_prompt(self, prompt: str, model: str | None = None, params: dict | None = None) -> None:
        self.rec["prompt"] = prompt
        self.rec["prompt_hash"] = sha256_text(prompt)
        if model:
            self.rec["llm"]["model"] = model
        if params:
            self.rec["llm"]["params"].update(params)

    def add_guardrail(self, **action) -> None:
        self.rec["guardrail_actions"].append(action)

    # ---- finish -----------------------------------------------------------
    def finish(self, response: str, citations=None, abstained: bool = False) -> str:
        self.rec["response"] = response
        self.rec["citations"] = [_plain(c) for c in (citations or [])]
        self.rec["abstained"] = abstained
        self.rec["latency_ms"]["total"] = round((time.perf_counter() - self._t0) * 1000, 2)
        if _TOP1 is not None:
            try:
                if self.rec["retrieved"]:
                    _TOP1.observe(max(h["score"] for h in self.rec["retrieved"]))
                if any(is_injection_action(a) for a in self.rec["guardrail_actions"]):
                    _INJ.inc()
            except Exception:
                pass
        self._emit()
        return self.rec["trace_id"]

    def _emit(self) -> None:
        line = json.dumps(self.rec, ensure_ascii=False, separators=(",", ":"), default=str)
        with _LOCK:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
            try:
                d = Path(self._dir)
                d.mkdir(parents=True, exist_ok=True)
                with open(d / f"traces-{self.rec['pod']}.jsonl", "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError as e:
                print(f"trace file write failed: {e}", file=sys.stderr)
