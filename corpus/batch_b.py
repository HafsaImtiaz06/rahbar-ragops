"""Add or remove RAGOps Batch B: documents with planted problems (assignment 3, part A).

    python corpus/batch_b.py add    --corpus corpus/raw     # copy the files, extend _index.json
    python corpus/batch_b.py remove --corpus corpus/raw     # take them out again
    python corpus/batch_b.py list                           # what each file is meant to trigger

The corpus directory must already have the _index.json that corpus/fetch.py wrote: once that
file exists the indexer reads only the files it lists. The exact copy of a real PDF is made
from the original in the same directory, so run corpus/fetch.py first.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent / "batch-b"
EXTRA_KEYS = {"problem", "check", "expect", "copy_of", "effect"}    # test notes, not manifest


def rows() -> list[dict]:
    return json.loads((HERE / "batch_b.json").read_text(encoding="utf-8"))


def _index(corpus: Path) -> tuple[Path, list[dict]]:
    path = corpus / "_index.json"
    if not path.exists():
        sys.exit(f"{path} not found. Run corpus/fetch.py first: Batch B extends that manifest.")
    return path, json.loads(path.read_text(encoding="utf-8"))


def add(corpus: Path) -> None:
    path, existing = _index(corpus)
    have = {r.get("file") for r in existing}
    added = 0
    for row in rows():
        dest = corpus / row["file"]
        if row.get("copy_of"):
            original = corpus / row["copy_of"]
            if not original.exists():
                print(f"skip {row['file']}: {row['copy_of']} is not in {corpus}")
                continue
            shutil.copy2(original, dest)
        else:
            shutil.copy2(HERE / row["file"], dest)
        if row["file"] not in have:
            existing.append({k: v for k, v in row.items() if k not in EXTRA_KEYS})
        added += 1
    path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    print(f"Batch B: {added} file(s) copied to {corpus}, manifest now lists {len(existing)}")


def remove(corpus: Path) -> None:
    path, existing = _index(corpus)
    names = {r["file"] for r in rows()}
    kept = [r for r in existing if r.get("file") not in names]
    for name in names:
        (corpus / name).unlink(missing_ok=True)
    path.write_text(json.dumps(kept, indent=2) + "\n", encoding="utf-8")
    print(f"Batch B removed: manifest lists {len(kept)} document(s)")


def show() -> None:
    for row in rows():
        print(f"{row['file']:<46} {row.get('expect', ''):<11} {row['problem']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("action", choices=["add", "remove", "list"])
    parser.add_argument("--corpus", default="corpus/raw")
    args = parser.parse_args()
    if args.action == "list":
        show()
    else:
        (add if args.action == "add" else remove)(Path(args.corpus))
