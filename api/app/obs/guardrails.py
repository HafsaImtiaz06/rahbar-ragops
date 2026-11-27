"""D4 - Guardrail logging.

RAHBAR's existing defence is structural (passage text and question are HTML-escaped inside <evidence>/<question>
and the rules say everything inside is data). That is a *neutraliser*, so it never says when it mattered.
This module adds a DETECTOR that runs on the retrieved chunks and returns actions to write into the trace:

    kept, actions = scan_chunks(hits)               # mode="flag": keep the chunk (escaping neutralises it), log it
    for a in actions: trace.add_guardrail(**a)
    trace.add_guardrail(**escape_action(len(kept)))  # records that the delimiter escaping was applied

Use mode="drop" if you want the rail to also remove flagged chunks (then say so in the report).
"""
from __future__ import annotations

import re

PATTERNS = [
    r"ignore\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?)",
    r"disregard\s+.{0,40}(instructions?|rules?|guidelines?)",
    r"you\s+are\s+now\s+(a|an|in)\b",
    r"(reveal|print|show|output)\s+.{0,30}(system\s+prompt|hidden\s+instructions?|api\s+key|secret)",
    r"do\s+not\s+(tell|mention|inform)\s+the\s+user",
    r"</?\s*(system|instructions?|evidence|passage)\s*>",
    r"new\s+instructions?\s*:",
]
_COMPILED = [(p, re.compile(p, re.I | re.S)) for p in PATTERNS]


def _get(h, k, default=None):
    if isinstance(h, dict):
        return h.get(k, default)
    return getattr(h, k, default)


def scan_chunks(hits, mode: str = "flag"):
    """Return (kept_hits, actions). Works on dicts or objects with .text/.chunk_id/.source/.score."""
    kept, actions = [], []
    for h in hits:
        text = _get(h, "text", "") or ""
        match = next((p for p, rx in _COMPILED if rx.search(text)), None)
        if match is None:
            kept.append(h)
            continue
        actions.append({
            "rail": "injection_filter",
            "action": "dropped_chunk" if mode == "drop" else "flagged_chunk",
            "chunk_id": _get(h, "chunk_id"),
            "doc_id": _get(h, "doc_id") or _get(h, "source"),
            "score": round(float(_get(h, "score", 0.0)), 5),
            "pattern": match,
        })
        if mode != "drop":
            kept.append(h)
    return kept, actions


def escape_action(n_chunks: int) -> dict:
    """Name deliberately has no 'injection' in it, so it does not inflate the injection counts."""
    return {"rail": "prompt_delimiter_escape", "action": "applied", "chunks": n_chunks}
