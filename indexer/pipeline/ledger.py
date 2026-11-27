"""Per-document ingestion counts (RAGOps A2, paper 4.2.2): what a run embedded and what it reused.

An unchanged corpus must embed zero chunks; a changed document must replace only its own chunks.
The table makes both visible: `reused` chunks kept their vectors, `embedded` chunks are new."""

from __future__ import annotations

from typing import Any

from .models import Chunk


def summarise(chunks: list[Chunk], reused_ids: set[str]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for c in chunks:
        row = rows.setdefault(c.source, {"document": c.source, "doc_version": c.doc_version,
                                         "superseded": c.superseded, "chunks": 0,
                                         "reused": 0, "embedded": 0})
        row["chunks"] += 1
        row["reused" if c.id in reused_ids else "embedded"] += 1
    return sorted(rows.values(), key=lambda r: str(r["document"]))


def totals(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {k: sum(int(r[k]) for r in rows) for k in ("chunks", "reused", "embedded")} | {
        "documents": len(rows)}


def render(rows: list[dict[str, Any]], removed: dict[str, int] | None = None) -> str:
    t = totals(rows)
    out = ["| Document | Version | Status | Chunks | Reused | Embedded |",
           "| --- | --- | --- | ---: | ---: | ---: |"]
    for r in rows:
        status = "superseded" if r["superseded"] else "current"
        out.append(f"| {r['document']} | {r['doc_version']} | {status} | {r['chunks']} | "
                   f"{r['reused']} | {r['embedded']} |")
    out.append(f"| **total ({t['documents']} documents)** | | | {t['chunks']} | "
               f"{t['reused']} | {t['embedded']} |")
    out.append("")
    out.append(f"Chunks embedded in this run: {t['embedded']} of {t['chunks']}")
    for source, n in sorted((removed or {}).items()):
        out.append(f"Removed from the index: {source} ({n} chunks)")
    return "\n".join(out)
