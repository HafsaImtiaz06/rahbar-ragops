"""What is in the vector store, counted by document (RAGOps A2).

    python -m pipeline.inventory                      # chunks per document in the served index
    python -m pipeline.inventory --all-generations    # also the previous one kept for rollback
    python -m pipeline.inventory --source handbook.pdf   # the count for one document id
    python -m pipeline.inventory --check-orphans      # exit 1 if the store holds a document
                                                      # that is no longer in the corpus
"""

from __future__ import annotations

import argparse
from typing import Any

from .config import IndexerSettings, get_settings
from .store import POINTER_SUFFIX, _active_collection_name, get_client

PAGE = 1000


def counts_by_source(collection: Any) -> dict[str, dict[str, Any]]:
    """{document: {chunks, doc_version, superseded}} for one collection."""
    out: dict[str, dict[str, Any]] = {}
    offset = 0
    while True:
        rows = collection.get(include=["metadatas"], limit=PAGE, offset=offset)
        metas = rows.get("metadatas") or []
        for meta in metas:
            entry = out.setdefault(str(meta.get("source", "?")), {
                "chunks": 0, "doc_version": str(meta.get("doc_version", "")),
                "superseded": bool(meta.get("superseded", False))})
            entry["chunks"] += 1
        if len(metas) < PAGE:
            return out
        offset += PAGE


def generations(client: Any, settings: IndexerSettings) -> list[tuple[str, str]]:
    """[('active', name), ('previous', name)]: the served collection and the rollback one."""
    found: list[tuple[str, str]] = []
    active = _active_collection_name(client, settings)
    if active:
        found.append(("active", active))
    names = {c if isinstance(c, str) else c.name for c in client.list_collections()}
    pointer = settings.collection + POINTER_SUFFIX
    if pointer in names:
        meta = (client.get_collection(pointer).get(ids=["active"])["metadatas"] or [{}])[0]
        previous = (meta or {}).get("previous", "")
        if previous and previous in names:
            found.append(("previous", str(previous)))
    return found


def active_counts(settings: IndexerSettings) -> dict[str, int]:
    """Chunks per document in the served index, or {} when there is none yet."""
    client = get_client(settings)
    name = _active_collection_name(client, settings)
    if name is None:
        return {}
    return {s: int(v["chunks"]) for s, v in
            counts_by_source(client.get_collection(name)).items()}


def render(label: str, name: str, counts: dict[str, dict[str, Any]]) -> str:
    out = [f"{label} generation: {name}", "",
           "| Document | Version | Status | Chunks |", "| --- | --- | --- | ---: |"]
    for source, v in sorted(counts.items()):
        out.append(f"| {source} | {v['doc_version']} | "
                   f"{'superseded' if v['superseded'] else 'current'} | {v['chunks']} |")
    out.append(f"| **total ({len(counts)} documents)** | | | "
               f"{sum(v['chunks'] for v in counts.values())} |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Count the vector store's chunks by document")
    parser.add_argument("--all-generations", action="store_true")
    parser.add_argument("--source", help="print only the chunk count for this document")
    parser.add_argument("--check-orphans", action="store_true")
    args = parser.parse_args(argv)
    settings = get_settings()
    client = get_client(settings)
    gens = generations(client, settings)
    if not gens:
        print("no index in the vector store")
        return 1
    if not args.all_generations:
        gens = gens[:1]
    status = 0
    for label, name in gens:
        counts = counts_by_source(client.get_collection(name))
        if args.source:
            print(f"{label} generation {name}: {args.source} has "
                  f"{counts.get(args.source, {}).get('chunks', 0)} chunk(s)")
            continue
        print(render(label, name, counts))
        print()
        if args.check_orphans and label == "active":
            from pathlib import Path

            from .run import discover

            expected = {p.name for p, _v, _s in discover(settings.corpus_dir)
                        if Path(p).exists()}
            orphans = sorted(set(counts) - expected)
            if orphans:
                status = 1
                print("ORPHANS (in the store, not in the corpus): " + ", ".join(orphans))
            else:
                print("no orphan chunks: every document in the store is in the corpus")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
