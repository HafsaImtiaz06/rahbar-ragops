"""Incremental update and rollback (RAGOps A2): counts, changed, deleted, superseded, restored.

The vector store is a dict here; what is under test is what the pipeline embeds and reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from test_verify import BASE, EXAMS, OTHER, bow, md, settings_for

from pipeline import inventory, ledger, versions
from pipeline.models import Chunk
from pipeline.run import run_pipeline


class FakeStore:
    """Stands in for Chroma: remembers the served index between runs."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.chunks: list[Chunk] = []
        self.vectors: dict[str, list[float]] = {}
        self.embedded_texts = 0

        def reuse(chunks: list[Chunk], _s: Any) -> dict[str, list[float]]:
            return {c.id: self.vectors[c.id] for c in chunks if c.id in self.vectors}

        def embed(texts: list[str], _s: Any) -> list[list[float]]:
            self.embedded_texts += len(texts)
            return [[1.0, 0.0] for _ in texts]

        def store(chunks: list[Chunk], vectors: list[list[float]], _s: Any) -> int:
            self.chunks = list(chunks)
            self.vectors = {c.id: v for c, v in zip(chunks, vectors, strict=True)}
            return len(chunks)

        monkeypatch.setattr("pipeline.store.reuse_embeddings", reuse)
        monkeypatch.setattr("pipeline.embed.embed", embed)
        monkeypatch.setattr("pipeline.store.store", store)
        monkeypatch.setattr("pipeline.inventory.active_counts", self.counts)
        monkeypatch.setattr("pipeline.verify.default_embedder", lambda _s: bow)

    def counts(self, _settings: Any = None) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.chunks:
            out[c.source] = out.get(c.source, 0) + 1
        return out


def summary(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    """The last 'ingestion summary' log line of the run, with documents keyed by name."""
    found = None
    for line in capsys.readouterr().out.splitlines():
        if line.startswith("{") and '"ingestion summary"' in line:
            found = json.loads(line)
    assert found is not None, "no ingestion summary was logged"
    found["by_doc"] = {r["document"]: r for r in found["per_document"]}
    return found


def corpus(tmp: Path) -> Path:
    md(tmp / "attendance.md", BASE, version="2026-06", title="Attendance Policy")
    md(tmp / "library.md", OTHER, version="2026-06", title="Library Rules")
    md(tmp / "exams.md", EXAMS, version="2026-06", title="Examination Rules")
    return tmp


def test_unchanged_corpus_embeds_zero_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                             capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeStore(monkeypatch)
    settings = settings_for(corpus(tmp_path))
    first = run_pipeline(settings)
    one = summary(capsys)
    assert one["embedded"] == first and one["reused"] == 0

    again = run_pipeline(settings)
    two = summary(capsys)
    assert again == first
    assert two["embedded"] == 0 and two["reused"] == first      # the A2 requirement
    assert fake.embedded_texts == first                          # nothing embedded the 2nd time


def test_a_changed_document_replaces_only_its_own_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    FakeStore(monkeypatch)
    settings = settings_for(corpus(tmp_path))
    run_pipeline(settings)
    capsys.readouterr()
    md(tmp_path / "library.md", OTHER.replace("six books", "eight books"), version="2026-06",
       title="Library Rules")
    run_pipeline(settings)
    docs = summary(capsys)["by_doc"]
    assert docs["library.md"]["embedded"] >= 1 and docs["library.md"]["reused"] == 0
    for untouched in ("attendance.md", "exams.md"):
        assert docs[untouched]["embedded"] == 0
        assert docs[untouched]["reused"] == docs[untouched]["chunks"]


def test_a_deleted_document_leaves_no_orphan_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeStore(monkeypatch)
    settings = settings_for(corpus(tmp_path))
    run_pipeline(settings)
    capsys.readouterr()
    before = fake.counts()["exams.md"]
    (tmp_path / "exams.md").unlink()
    run_pipeline(settings)
    out = summary(capsys)
    assert out["removed"] == {"exams.md": before}                # reported as removed
    assert fake.counts().get("exams.md", 0) == 0                 # count by document id: zero
    assert not any(c.source == "exams.md" for c in fake.chunks)
    assert "attendance.md" in fake.counts() and "library.md" in fake.counts()


def test_superseded_version_is_flagged_then_restored_and_rolled_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeStore(monkeypatch)
    settings = settings_for(corpus(tmp_path))
    run_pipeline(settings)
    capsys.readouterr()

    # a newer version of the library rules arrives
    md(tmp_path / "library-rules-v2.md", OTHER.replace("fourteen days", "twenty one days"),
       version="2026-11", title="Library Rules")
    run_pipeline(settings)
    docs = summary(capsys)["by_doc"]
    assert docs["library.md"]["superseded"] is True          # excluded from retrieval by metadata
    assert docs["library-rules-v2.md"]["superseded"] is False
    assert docs["attendance.md"]["embedded"] == 0            # nothing else was touched
    archived = list((tmp_path / ".ragops" / "archive").rglob("library.md"))
    assert len(archived) == 1                                # a copy is kept in the archive
    assert "library.md" in versions.listing(settings)

    # roll back: restore the old version, run the pipeline again
    versions.restore(settings, "library.md")
    run_pipeline(settings)
    docs = summary(capsys)["by_doc"]
    assert docs["library.md"]["superseded"] is False
    assert docs["library-rules-v2.md"]["superseded"] is True
    assert docs["attendance.md"]["embedded"] == 0 and docs["exams.md"]["embedded"] == 0
    assert {c.source: c.superseded for c in fake.chunks if "library" in c.source} == {
        "library.md": False, "library-rules-v2.md": True}

    # undo the rollback
    versions.unpin(settings, "library.md")
    run_pipeline(settings)
    docs = summary(capsys)["by_doc"]
    assert docs["library.md"]["superseded"] is True
    assert docs["library-rules-v2.md"]["superseded"] is False


def test_restore_puts_an_archived_file_back_into_the_corpus(tmp_path: Path) -> None:
    settings = settings_for(corpus(tmp_path))
    archive = tmp_path / ".ragops" / "archive" / "old"
    archive.mkdir(parents=True)
    (archive / "old-rules.md").write_text("# Old\n\nolder text kept in the archive.\n")
    message = versions.restore(settings, "old-rules.md")
    assert (tmp_path / "old-rules.md").exists() and "pinned" in message
    with pytest.raises(SystemExit):
        versions.restore(settings, "never-existed.md")


def test_ledger_render() -> None:
    chunks = [Chunk(id="a1", text="x", source="a.md", section="s", page=1, doc_version="1",
                    superseded=False),
              Chunk(id="a2", text="y", source="a.md", section="s", page=1, doc_version="1",
                    superseded=False)]
    rows = ledger.summarise(chunks, {"a1"})
    assert rows[0]["reused"] == 1 and rows[0]["embedded"] == 1
    text = ledger.render(rows, {"gone.md": 7})
    assert "Chunks embedded in this run: 1 of 2" in text
    assert "Removed from the index: gone.md (7 chunks)" in text


# -- inventory: counts by document id --------------------------------------------------
class FakeCollection:
    def __init__(self, name: str, metas: list[dict[str, Any]], documents: list[str] | None = None,
                 meta: dict[str, Any] | None = None) -> None:
        self.name, self.metas, self.documents, self.metadata = name, metas, documents, meta or {}

    def get(self, ids: list[str] | None = None, include: Any = None, limit: int = 1000,
            offset: int = 0) -> dict[str, Any]:
        if ids is not None:
            return {"documents": self.documents or [], "metadatas": self.metas}
        return {"metadatas": self.metas[offset:offset + limit]}


class FakeClient:
    def __init__(self, cols: list[FakeCollection]) -> None:
        self.cols = {c.name: c for c in cols}

    def list_collections(self) -> list[str]:
        return list(self.cols)

    def get_collection(self, name: str) -> FakeCollection:
        return self.cols[name]


def chroma(current: dict[str, int], previous: dict[str, int]) -> FakeClient:
    def metas(counts: dict[str, int]) -> list[dict[str, Any]]:
        return [{"source": s, "doc_version": "2026", "superseded": False}
                for s, n in counts.items() for _ in range(n)]
    return FakeClient([
        FakeCollection("rahbar-active", [{"previous": "rahbar-1"}], ["rahbar-2"]),
        FakeCollection("rahbar-2", metas(current)),
        FakeCollection("rahbar-1", metas(previous)),
    ])


def test_counts_by_source_pages_through_the_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inventory, "PAGE", 2)
    col = chroma({"a.md": 3, "b.md": 2}, {}).get_collection("rahbar-2")
    counts = inventory.counts_by_source(col)
    assert counts["a.md"]["chunks"] == 3 and counts["b.md"]["chunks"] == 2


def test_inventory_shows_active_and_previous_generations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    client = chroma({"a.md": 2}, {"a.md": 2, "gone.md": 4})
    settings = settings_for(tmp_path, collection="rahbar")
    monkeypatch.setattr(inventory, "get_settings", lambda: settings)
    monkeypatch.setattr(inventory, "get_client", lambda _s: client)

    assert inventory.main(["--source", "gone.md"]) == 0
    assert "gone.md has 0 chunk(s)" in capsys.readouterr().out          # active: none
    assert inventory.main(["--all-generations", "--source", "gone.md"]) == 0
    out = capsys.readouterr().out
    assert "active generation rahbar-2: gone.md has 0" in out
    assert "previous generation rahbar-1: gone.md has 4" in out          # kept for rollback


def test_orphan_check_fails_when_the_store_holds_a_missing_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    md(tmp_path / "a.md", BASE, version="2026", title="A")
    settings = settings_for(tmp_path, collection="rahbar")
    monkeypatch.setattr(inventory, "get_settings", lambda: settings)
    monkeypatch.setattr(inventory, "get_client", lambda _s: chroma({"a.md": 2, "gone.md": 1}, {}))
    assert inventory.main(["--check-orphans"]) == 1
    out = capsys.readouterr().out
    assert "ORPHANS" in out and "gone.md" in out
    monkeypatch.setattr(inventory, "get_client", lambda _s: chroma({"a.md": 2}, {}))
    assert inventory.main(["--check-orphans"]) == 0
    assert "no orphan chunks" in capsys.readouterr().out
