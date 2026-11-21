"""Archived versions and rollback (RAGOps A2, paper 4.2.3).

    python -m pipeline.versions list                    # archived versions and restored ones
    python -m pipeline.versions restore <file name>     # make that version the current one
    python -m pipeline.versions unpin <file name>       # back to "the newest version wins"

An older version that verification found superseded is indexed with superseded=true, so retrieval
excludes it by metadata, and a copy sits in the archive folder. `restore` pins a version: the next
ingestion run treats it as current and flags the newer version(s) superseded instead. Nothing is
deleted either way, so the rollback can be undone with `unpin`. Run the ingestion Job afterwards."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .config import IndexerSettings, get_settings
from .verify import _dirs, load_pins


def _save(report_dir: Path, pins: set[str]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "pins.json").write_text(
        json.dumps({"restored": sorted(pins)}, indent=2), encoding="utf-8")


def _archived(archive_dir: Path) -> list[Path]:
    return sorted(p for p in archive_dir.rglob("*") if p.is_file()) if archive_dir.exists() else []


def listing(settings: IndexerSettings) -> str:
    _quarantine, archive, report_dir = _dirs(settings)
    pins = load_pins(report_dir)
    lines = ["Archived versions (older versions kept for rollback):"]
    for path in _archived(archive):
        rel = path.relative_to(archive).as_posix()
        mark = "  [restored]" if path.name in pins else ""
        lines.append(f"  {path.name}    ({rel}){mark}")
    if len(lines) == 1:
        lines.append("  none")
    lines.append("")
    lines.append("Restored (pinned) versions: " + (", ".join(sorted(pins)) or "none"))
    return "\n".join(lines)


def restore(settings: IndexerSettings, name: str) -> str:
    _quarantine, archive, report_dir = _dirs(settings)
    corpus = Path(settings.corpus_dir)
    in_archive = [p for p in _archived(archive) if p.name == name]
    if not in_archive and not (corpus / name).exists():
        raise SystemExit(f"{name} is neither in the archive nor in the corpus: "
                         f"run `python -m pipeline.versions list`")
    if not (corpus / name).exists():
        shutil.copy2(in_archive[0], corpus / name)      # put the archived file back
    pins = load_pins(report_dir) | {name}
    _save(report_dir, pins)
    return (f"{name} is now pinned as the current version. Run the ingestion Job: the newer "
            f"version(s) will be flagged superseded and excluded from retrieval.")


def unpin(settings: IndexerSettings, name: str) -> str:
    _quarantine, _archive, report_dir = _dirs(settings)
    pins = load_pins(report_dir)
    pins.discard(name)
    _save(report_dir, pins)
    return f"{name} is no longer pinned. Run the ingestion Job: the newest version wins again."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List, restore or unpin archived versions")
    parser.add_argument("action", choices=["list", "restore", "unpin"])
    parser.add_argument("name", nargs="?", help="file name of the version")
    args = parser.parse_args(argv)
    settings = get_settings()
    if args.action == "list":
        print(listing(settings))
        return 0
    if not args.name:
        parser.error(f"{args.action} needs a file name")
    print((restore if args.action == "restore" else unpin)(settings, args.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
