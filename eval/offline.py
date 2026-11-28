"""Offline helpers for the M2 suite: the real indexer code, an in-memory index, no Chroma.

The corpus is extracted, cleaned and chunked by the same functions the indexer Job uses, then
embedded with the pinned model through Ollama's /api/embed. Ranking is plain cosine similarity
(no lexical blend, no relevance floor), so this measures the embedding and the chunking, not the
deployed retriever. eval/chunk_sweep.py works the same way.
"""

from __future__ import annotations

import importlib.util
import random
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "indexer"))

from pipeline.chunk import chunk  # noqa: E402
from pipeline.clean import clean  # noqa: E402
from pipeline.embed import prepare  # noqa: E402
from pipeline.extract import extract  # noqa: E402
from pipeline.models import Chunk, Document  # noqa: E402
from pipeline.run import _front_matter, discover  # noqa: E402

from metrics import cosine  # noqa: E402

BATCH = 16


def _load_version_helper():
    """api/app/retrieval/version.py is pure regex; load it by path so the API package (and its
    dependencies) need not be installed to run the evaluation."""
    path = ROOT / "api" / "app" / "retrieval" / "version.py"
    spec = importlib.util.spec_from_file_location("rahbar_version", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.wants_older_version


wants_older_version = _load_version_helper()


class Embedder:
    """Ollama /api/embed client. Applies the same task prefixes as the indexer and the API."""

    def __init__(self, url: str, model: str, client: httpx.Client | None = None,
                 timeout: float = 600.0) -> None:
        self.model = model
        self.client = client or httpx.Client(base_url=url, timeout=timeout)

    def embed(self, texts: list[str], kind: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), BATCH):
            batch = [prepare(self.model, t, kind) for t in texts[start:start + BATCH]]
            response = self.client.post("/api/embed", json={"model": self.model, "input": batch})
            response.raise_for_status()
            got = response.json()["embeddings"]
            if len(got) != len(batch):
                raise ValueError(f"asked for {len(batch)} vectors, got {len(got)}")
            vectors.extend(got)
        return vectors

    def timed(self, texts: list[str], kind: str) -> tuple[list[list[float]], float]:
        started = time.perf_counter()
        vectors = self.embed(texts, kind)
        return vectors, time.perf_counter() - started


def load_documents(corpus_dir: str) -> list[Document]:
    return [clean(extract(str(p), v, s)) for p, v, s in discover(corpus_dir)]


def chunk_documents(docs: list[Document], size: int, overlap: int) -> list[Chunk]:
    return [c for d in docs for c in chunk(d, size, overlap)]


def build_chunks(corpus_dir: str, size: int, overlap: int) -> list[Chunk]:
    return chunk_documents(load_documents(corpus_dir), size, overlap)


def load_distractors(directory: Path, size: int, overlap: int) -> tuple[list[Chunk], list[str]]:
    """Chunks of every distractor-*.md file in `directory` (README and others are ignored)."""
    chunks: list[Chunk] = []
    names: list[str] = []
    for path in sorted(directory.glob("distractor-*.md")):
        meta = _front_matter(path)
        doc = clean(extract(str(path), meta.get("doc_version", "2026"),
                            meta.get("superseded", "false").lower() == "true"))
        chunks.extend(chunk(doc, size, overlap))
        names.append(path.name)
    return chunks, names


class Index:
    """Chunks with their vectors, searched by cosine similarity."""

    def __init__(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be the same length")
        self.chunks = chunks
        self.vectors = vectors

    def with_extra(self, chunks: list[Chunk], vectors: list[list[float]]) -> Index:
        return Index(self.chunks + chunks, self.vectors + vectors)

    def search(self, query_vector: list[float], question: str, k: int = 5) -> list[dict]:
        older = wants_older_version(question)
        scored = [(cosine(query_vector, v), c) for v, c in zip(self.vectors, self.chunks,
                                                               strict=True)
                  if older or not c.superseded]
        scored.sort(key=lambda t: (-t[0], t[1].id))   # ties broken by chunk id: deterministic
        return [{"source": c.source, "section": c.section, "page": c.page,
                 "score": round(s, 6)} for s, c in scored[:k]]


def _norm(text: str) -> str:
    return " ".join(str(text).lower().split())


def evidence_text(refs: list[dict], chunks: list[Chunk], limit_chars: int = 6000) -> str:
    """Text of the chunks that a citation or retrieved ref points to (same source and section)."""
    wanted = {(_norm(r["source"]), _norm(r["section"])) for r in refs}
    parts: list[str] = []
    for c in chunks:
        if (_norm(c.source), _norm(c.section)) in wanted:
            parts.append(c.text)
    return "\n\n".join(parts)[:limit_chars]


def add_typos(text: str, rng: random.Random, rate: float = 0.15) -> str:
    """Seeded, deterministic misspellings: swap two neighbouring letters, drop one, or double
    one. Words of four or more letters are eligible; at least one is changed when any is."""
    words = text.split(" ")
    eligible = [i for i, w in enumerate(words) if len(w) >= 4 and w.isalpha()]
    if not eligible:
        return text

    def mangle(word: str) -> str:
        pos = rng.randrange(1, len(word) - 1)
        op = rng.choice(("swap", "drop", "double"))
        if op == "swap" and word[pos] != word[pos + 1]:
            return word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:]
        if op == "drop":
            return word[:pos] + word[pos + 1:]
        return word[:pos] + word[pos] + word[pos:]

    changed = False
    for i in eligible:
        if rng.random() < rate:
            new = mangle(words[i])
            changed = changed or new != words[i]
            words[i] = new
    if not changed:
        i = eligible[0]
        words[i] = words[i][:1] + words[i][2] + words[i][1] + words[i][3:] \
            if words[i][1] != words[i][2] else words[i][:2] + words[i][3:]
    return " ".join(words)
