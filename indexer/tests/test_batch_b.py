"""Batch B (corpus/batch-b): every planted problem is caught by the check it was made for."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from pipeline.run import discover
from pipeline.verify import CHECKS, collect_candidates, verify
from test_verify import BASE, OTHER, bow, md, settings_for

SCRIPT = Path(__file__).resolve().parents[2] / "corpus" / "batch_b.py"


def load_script():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("batch_b", SCRIPT)
    module = importlib.util.module_from_spec(spec)          # type: ignore[arg-type]
    spec.loader.exec_module(module)                         # type: ignore[union-attr]
    return module


def corpus_with_batch_a(tmp: Path) -> Path:
    """A small stand-in for batch A: the existing Fee Information document and one more."""
    md(tmp / "fee-information-prospectus-2026.md", BASE, version="2026-27",
       title="Fee Information (Prospectus 2026)")
    md(tmp / "student-handbook.md", OTHER, version="2026-06",
       title="Student Handbook June 2026")
    (tmp / "_index.json").write_text(json.dumps([
        {"file": "fee-information-prospectus-2026.md", "name": "Fee Information (Prospectus 2026)",
         "doc_version": "2026-27", "superseded": False},
        {"file": "student-handbook.md", "name": "Student Handbook June 2026",
         "doc_version": "2026-06", "superseded": False},
    ]))
    return tmp


def test_every_planted_problem_is_caught_by_its_check(tmp_path: Path) -> None:
    script = load_script()
    corpus_with_batch_a(tmp_path)
    script.add(tmp_path)                    # the real-PDF copy is skipped: no original here
    settings = settings_for(tmp_path, near_dup_threshold=0.97)
    result = verify(collect_candidates(discover(str(tmp_path)), settings, lambda *a, **k: None),
                    settings, embedder=bow, run_id="B1")
    by_file = {v.candidate.label: v for v in result.verdicts}

    for row in script.rows():
        v = by_file.get(row["file"])
        if row.get("copy_of"):
            assert v is None                # skipped, as printed
            continue
        assert v is not None, row["file"]
        assert v.disposition == row["expect"], (row["file"], v.disposition,
                                                {n: c.reason for n, c in v.checks.items()})
        if row["expect"] == "quarantine":
            assert v.failed(row["check"]), (row["file"], row["check"])

    # the specific reasons the report will show
    assert "exact duplicate of library-rules-2026.md" in \
        by_file["library-rules-2026-copy.md"].checks["uniqueness"].reason
    assert "near-duplicate of library-rules-2026.md" in \
        by_file["library-rules-2026-reworded.md"].checks["uniqueness"].reason
    assert "title" in by_file["notice-board-undated.md"].checks["completeness"].reason
    assert "extraction failed" in by_file["hostel-scan-corrupt.pdf"].checks["quality"].reason

    # the newer Fee Information retires the old one (archived, still flagged superseded)
    old = by_file["fee-information-prospectus-2026.md"]
    assert old.disposition == "archive" and old.final_superseded
    assert "fee-information-prospectus-2027.md" in old.checks["recency"].reason
    # the document with the injected instruction passes every check (D4 catches it later)
    assert all(by_file["campus-notices-injected.md"].checks[n].passed for n in CHECKS)
    # untouched batch A documents stay indexed
    assert by_file["student-handbook.md"].disposition == "index"


def test_add_is_repeatable_and_remove_restores_the_manifest(tmp_path: Path) -> None:
    script = load_script()
    corpus_with_batch_a(tmp_path)
    before = (tmp_path / "_index.json").read_text(encoding="utf-8")
    script.add(tmp_path)
    script.add(tmp_path)                    # twice: no duplicate manifest rows
    rows = json.loads((tmp_path / "_index.json").read_text(encoding="utf-8"))
    files = [r["file"] for r in rows]
    assert len(files) == len(set(files))
    assert not any(k in r for r in rows for k in ("expect", "problem", "check", "copy_of"))
    script.remove(tmp_path)
    assert json.loads((tmp_path / "_index.json").read_text(encoding="utf-8")) == json.loads(before)
    assert not (tmp_path / "library-rules-2026.md").exists()


def test_the_exact_copy_of_a_real_pdf_is_made_from_the_original(tmp_path: Path) -> None:
    script = load_script()
    corpus_with_batch_a(tmp_path)
    (tmp_path / "fee-information-prospectus-2026.pdf").write_bytes(b"%PDF-1.4 original bytes")
    script.add(tmp_path)
    assert (tmp_path / "fee-information-prospectus-2026-copy.pdf").read_bytes() == \
        b"%PDF-1.4 original bytes"
