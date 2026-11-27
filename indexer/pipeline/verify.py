"""Stage 1b: Verify (RAGOps part A1, paper section 4.2.1).

Runs after extraction and before cleaning and chunking. For every document it records pass or
fail, with a reason, for four checks:

  quality       text is present, not near-empty and not garbled (or extraction failed)
  completeness  required metadata present (source, title, version or date, page count) and the
                document is not cut short
  recency       a newer version of the same document is present in this run
  uniqueness    not an exact duplicate (hash of the normalised text or of the file) and not a
                near-duplicate (cosine similarity of document embeddings)

Outcome per document:

  index       all four checks pass
  archive     only recency failed: an older version. It is still indexed, but flagged
              superseded (retrieval excludes it by metadata) and a copy goes to the archive folder
  quarantine  quality, completeness or uniqueness failed: the file is moved to the quarantine
              folder and is not indexed

Every run writes verification-<run_id>.json and .md to the report directory.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import IndexerSettings
from .models import Document

CHECKS = ("quality", "completeness", "recency", "uniqueness")
Embedder = Callable[[list[str]], list[list[float]]]
Logger = Callable[..., None]

SAMPLE_CHARS = 3000          # how much text represents a document in the similarity check
MAX_GARBLED = 0.02           # share of replacement or control characters
MIN_ALNUM = 0.50             # share of non-space characters that are letters or digits

_MONTHS = (r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
           r"sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?")


@dataclass
class Check:
    passed: bool
    reason: str


@dataclass
class Candidate:
    """One document as the extract stage left it, before verification."""

    path: Path
    doc_version: str
    superseded: bool
    doc: Document | None = None          # None when extraction failed
    error: str | None = None
    entry: dict[str, Any] = field(default_factory=dict)    # its _index.json row, if any
    label: str = ""                      # corpus-relative name used in reports

    @property
    def source(self) -> str:
        return self.path.name

    @property
    def text(self) -> str:
        return "\n".join(p.text for p in self.doc.pages) if self.doc else ""


@dataclass
class Verdict:
    candidate: Candidate
    checks: dict[str, Check] = field(default_factory=dict)
    disposition: str = "index"
    final_superseded: bool = False
    sha256: str = ""
    text_sha256: str = ""
    pages: int = 0
    chars: int = 0
    title: str = ""
    moved_to: str = ""
    note: str = ""
    sample: str = ""

    def failed(self, name: str) -> bool:
        return not self.checks[name].passed

    def as_dict(self) -> dict[str, Any]:
        return {
            "document": self.candidate.label or self.candidate.source,
            "doc_version": self.candidate.doc_version,
            "title": self.title,
            "disposition": self.disposition,
            "superseded_flag": self.final_superseded,
            "pages": self.pages,
            "chars": self.chars,
            "sha256": self.sha256,
            "checks": {n: {"passed": c.passed, "reason": c.reason}
                       for n, c in self.checks.items()},
            "moved_to": self.moved_to,
            "note": self.note,
        }


@dataclass
class VerificationResult:
    run_id: str
    verdicts: list[Verdict]
    report_json: str = ""
    report_md: str = ""

    def indexable(self) -> list[Verdict]:
        return [v for v in self.verdicts if v.disposition in ("index", "archive")]

    def counts(self) -> dict[str, int]:
        out = {"documents": len(self.verdicts), "index": 0, "archive": 0, "quarantine": 0}
        for v in self.verdicts:
            out[v.disposition] += 1
        return out


# -- small helpers --------------------------------------------------------------------
def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
    except OSError:
        return ""
    return h.hexdigest()


def text_sha256(text: str) -> str:
    n = _norm(text)
    return hashlib.sha256(n.encode()).hexdigest() if n else ""


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def doc_key(name: str) -> str:
    """Identity of a document across versions: lower case, with extension, months, years,
    edition and version words and numbers removed."""
    t = re.sub(r"\.[a-z0-9]{2,4}$", "", name.lower())
    t = re.sub(rf"\b(?:{_MONTHS})\b", " ", t)
    t = re.sub(r"\b(?:v|ver|version|rev|edition|final|draft|copy)\s*\d*(?:\.\d+)*\b", " ", t)
    t = re.sub(r"\b\d{4}(?:[-_/]\d{1,4})?\b|\b\d{1,2}\b", " ", t)
    return re.sub(r"[^a-z]+", " ", t).strip()


def version_tuple(*values: str | None) -> tuple[int, ...]:
    """'2026-06' -> (2026, 6); '2023-24' -> (2023, 24); 'v2.1' -> (2, 1); 'unknown' -> ()."""
    for value in values:
        if value and value.lower() != "unknown":
            nums = tuple(int(n) for n in re.findall(r"\d+", value))
            if nums:
                return nums
    return ()


def _front_matter(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return {}
    if not lines or lines[0].strip() != "---":
        return {}
    meta: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta


def load_manifest(corpus_dir: str) -> dict[Path, dict[str, Any]]:
    """Rows of _index.json by resolved path. Tolerant: run.discover already validated it."""
    root = Path(corpus_dir)
    index = root / "_index.json"
    if not index.exists():
        return {}
    try:
        rows = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict[Path, dict[str, Any]] = {}
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict) and isinstance(row.get("file"), str):
            out[(root / row["file"]).resolve()] = row
    return out


def _pdf_info(path: Path) -> tuple[int | None, str | None, str | None]:
    """(page count, title, problem) from the PDF's own structure."""
    try:
        with path.open("rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 2048))
            tail = fh.read()
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = len(reader.pages)
        title = (reader.metadata.title if reader.metadata else None) or None
    except Exception as exc:    # noqa: BLE001 - any parse failure means a broken file
        return None, None, f"PDF structure unreadable ({type(exc).__name__}: {exc})"
    problem = None if b"%%EOF" in tail else "no %%EOF marker at the end of the file (cut short)"
    return pages, title, problem


def _ends_mid_sentence(text: str) -> bool:
    tail = text.rstrip()
    if not tail:
        return False
    last = tail.splitlines()[-1].lstrip()
    if last.startswith(("#", "|", ">")):
        return False
    return (tail[-1].isalpha() and tail[-1].islower()) or tail[-1] in ",;:-("


# -- the four checks ------------------------------------------------------------------
def check_quality(c: Candidate, min_chars: int) -> Check:
    if c.error:
        return Check(False, f"extraction failed: {c.error}")
    visible = [ch for ch in c.text if not ch.isspace()]
    if len(visible) < min_chars:
        return Check(False, f"empty or near-empty: {len(visible)} characters of text "
                            f"(minimum {min_chars})")
    bad = sum(1 for ch in visible if ch == "�" or ord(ch) < 32)
    if bad / len(visible) > MAX_GARBLED:
        return Check(False, f"garbled: {bad / len(visible):.1%} replacement or control characters")
    alnum = sum(1 for ch in visible if ch.isalnum()) / len(visible)
    if alnum < MIN_ALNUM:
        return Check(False, f"garbled: only {alnum:.0%} of characters are letters or digits")
    return Check(True, f"{len(visible)} characters of text, {alnum:.0%} letters or digits")


def check_completeness(c: Candidate, v: Verdict) -> Check:
    entry = c.entry
    fm = _front_matter(c.path) if c.path.suffix.lower() == ".md" else {}
    pdf_pages = pdf_title = pdf_problem = None
    if c.path.suffix.lower() == ".pdf":
        pdf_pages, pdf_title, pdf_problem = _pdf_info(c.path)
    title = entry.get("title") or entry.get("name") or fm.get("title") or pdf_title
    if not title and c.doc and c.path.suffix.lower() == ".md":
        heading = re.search(r"^#\s+(.+)$", c.doc.pages[0].text, re.MULTILINE) if c.doc.pages else None
        title = heading.group(1).strip() if heading else None
    date = entry.get("date") or fm.get("date")
    pages = len(c.doc.pages) if c.doc else (pdf_pages or 0)
    v.title = str(title or "")
    v.pages = pages

    missing = []
    if not c.source:
        missing.append("source")
    if not title:
        missing.append("title")
    if c.doc_version.lower() in ("", "unknown") and not date:
        missing.append("date or version")
    if not pages:
        missing.append("page count")
    if missing:
        return Check(False, "missing metadata: " + ", ".join(missing))

    expected = entry.get("pages")
    if isinstance(expected, int) and expected != pages:
        return Check(False, f"truncated: {pages} page(s) found, the manifest records {expected}")
    if pdf_problem:
        return Check(False, f"truncated or corrupt: {pdf_problem}")
    if pdf_pages is not None and c.doc and pdf_pages != len(c.doc.pages):
        return Check(False, f"truncated: PDF declares {pdf_pages} page(s), "
                            f"{len(c.doc.pages)} extracted")
    if c.path.suffix.lower() != ".pdf" and _ends_mid_sentence(c.text):
        return Check(False, "truncated: the text ends in the middle of a sentence")
    return Check(True, f"source, title, version/date and {pages} page(s) present; "
                       f"text not truncated")


def _rank(v: Verdict, pins: set[str]) -> tuple[int, tuple[int, ...]]:
    """A restored (pinned) version outranks every other version of its document."""
    return (1 if v.candidate.source in pins else 0, version_tuple(v.candidate.doc_version))


def _apply_recency(verdicts: list[Verdict], pins: set[str] | None = None) -> None:
    """Fail every document that has a newer version, among documents that are themselves
    readable and complete (a corrupt newer file must not retire the good older one). A version
    the user restored with `pipeline.versions restore` counts as the newest."""
    pins = pins or set()
    valid = [v for v in verdicts if v.checks["quality"].passed and v.checks["completeness"].passed]
    keys: dict[str, list[Verdict]] = {}
    for v in valid:
        keys.setdefault(_key_of(v), []).append(v)
    for v in verdicts:
        mine = _rank(v, pins)
        newer = [o for o in keys.get(_key_of(v), []) if o is not v and _rank(o, pins) > mine]
        if newer and (mine[1] or mine[0]):
            best = max(newer, key=lambda o: _rank(o, pins))
            name = best.candidate.label or best.candidate.source
            if _rank(best, pins)[0]:
                reason = f"superseded: {name} was restored (rollback) as the current version"
            else:
                reason = (f"superseded: {name} is a newer version "
                          f"({best.candidate.doc_version} > {v.candidate.doc_version})")
            v.checks["recency"] = Check(False, reason)
        else:
            v.checks["recency"] = Check(True, "no newer version of this document in the corpus")


def _key_of(v: Verdict) -> str:
    return doc_key(v.title or v.candidate.source) or v.candidate.source.lower()


def _apply_uniqueness(verdicts: list[Verdict], first_seen: dict[str, str], threshold: float,
                      embedder: Embedder | None) -> None:
    # The copy that was seen first is the original and stays. On a tie (the first run over a
    # corpus nobody has verified yet) the shorter name wins, then the alphabetical one.
    order = sorted(verdicts, key=lambda v: (first_seen.get(v.candidate.label, "~"),
                                            len(v.candidate.label), v.candidate.label))
    kept: list[Verdict] = []
    samples = [v for v in order if v.sample]
    vectors: dict[int, list[float]] = {}
    if embedder is not None and len(samples) > 1:
        for v, vec in zip(samples, embedder([v.sample for v in samples]), strict=True):
            vectors[id(v)] = vec
    for v in order:
        verdict = Check(True, "unique: no exact or near duplicate")
        for k in kept:
            if k.candidate.path == v.candidate.path:
                continue            # the same file listed twice is one document
            if (v.sha256 and v.sha256 == k.sha256) or (
                    v.text_sha256 and v.text_sha256 == k.text_sha256):
                verdict = Check(False, f"exact duplicate of {k.candidate.label} "
                                       f"(same {'text' if v.text_sha256 else 'file'} hash)")
                break
            same_family = _key_of(v) == _key_of(k) and \
                version_tuple(v.candidate.doc_version) != version_tuple(k.candidate.doc_version)
            if same_family or id(v) not in vectors or id(k) not in vectors:
                continue            # different versions of one document are recency, not duplicates
            sim = cosine(vectors[id(v)], vectors[id(k)])
            if sim >= threshold:
                verdict = Check(False, f"near-duplicate of {k.candidate.label} "
                                       f"(embedding cosine {sim:.3f} >= {threshold})")
                break
        v.checks["uniqueness"] = verdict
        if verdict.passed and v.checks["quality"].passed and v.checks["completeness"].passed:
            kept.append(v)


# -- directories, state, report -------------------------------------------------------
def _dirs(settings: IndexerSettings) -> tuple[Path, Path, Path]:
    base = Path(settings.corpus_dir) / ".ragops"
    return (Path(settings.quarantine_dir) if settings.quarantine_dir else base / "quarantine",
            Path(settings.archive_dir) if settings.archive_dir else base / "archive",
            Path(settings.report_dir) if settings.report_dir else base / "reports")


def _load_state(report_dir: Path) -> dict[str, str]:
    try:
        data = json.loads((report_dir / "first-seen.json").read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in data.get("first_seen", {}).items()}
    except (OSError, ValueError, AttributeError):
        return {}


def _safe(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", label)


def _move_files(verdicts: list[Verdict], settings: IndexerSettings, run_id: str,
                log: Logger) -> None:
    quarantine, archive, _ = _dirs(settings)
    for v in verdicts:
        src = v.candidate.path
        try:
            if v.disposition == "quarantine":
                dest = quarantine / run_id / _safe(v.candidate.label)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), dest)
                dest.with_name(dest.name + ".reason.json").write_text(
                    json.dumps(v.as_dict(), indent=2), encoding="utf-8")
                v.moved_to = str(dest)
            elif v.disposition == "archive":
                dest = (archive / _safe(_key_of(v)) / _safe(v.candidate.doc_version)
                        / src.name)
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists() or dest.stat().st_size != src.stat().st_size:
                    shutil.copy2(src, dest)
                v.moved_to = str(dest)
        except OSError as exc:      # a read-only corpus volume: still excluded from the index
            v.note = f"could not move the file ({exc}); it is still excluded from indexing"
            log("verification could not move a file", source=v.candidate.label,
                disposition=v.disposition, error=str(exc))


def render_markdown(result: VerificationResult) -> str:
    c = result.counts()
    out = [f"# Verification report {result.run_id}", "",
           (f"Documents: {c['documents']} | indexed: {c['index']} | "
            f"archived (superseded): {c['archive']} | quarantined: {c['quarantine']}"), "",
           "| Document | Version | " + " | ".join(CHECKS) + " | Outcome |",
           "| --- | --- | " + " | ".join("---" for _ in CHECKS) + " | --- |"]
    for v in result.verdicts:
        cells = ["pass" if v.checks[n].passed else "FAIL" for n in CHECKS]
        out.append(f"| {v.candidate.label} | {v.candidate.doc_version} | "
                   + " | ".join(cells) + f" | {v.disposition} |")
    out += ["", "## Reasons", ""]
    for v in result.verdicts:
        out.append(f"### {v.candidate.label} ({v.disposition})")
        for n in CHECKS:
            chk = v.checks[n]
            out.append(f"- {n}: {'pass' if chk.passed else 'FAIL'}: {chk.reason}")
        if v.moved_to:
            out.append(f"- file: {v.moved_to}")
        if v.note:
            out.append(f"- note: {v.note}")
        out.append("")
    return "\n".join(out)


def _write_report(result: VerificationResult, settings: IndexerSettings,
                  first_seen: dict[str, str], log: Logger) -> None:
    _, _, report_dir = _dirs(settings)
    payload = {"run_id": result.run_id, "counts": result.counts(),
               "thresholds": {"near_duplicate_cosine": settings.near_dup_threshold,
                              "min_chars": settings.verify_min_chars},
               "documents": [v.as_dict() for v in result.verdicts]}
    try:
        report_dir.mkdir(parents=True, exist_ok=True)
        jpath = report_dir / f"verification-{result.run_id}.json"
        mpath = report_dir / f"verification-{result.run_id}.md"
        jpath.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        mpath.write_text(render_markdown(result), encoding="utf-8")
        (report_dir / "first-seen.json").write_text(
            json.dumps({"first_seen": first_seen}, indent=2), encoding="utf-8")
        result.report_json, result.report_md = str(jpath), str(mpath)
    except OSError as exc:
        log("verification report could not be written to disk", error=str(exc),
            report=payload)


def load_pins(report_dir: Path) -> set[str]:
    """Documents restored by hand (`python -m pipeline.versions restore`)."""
    try:
        data = json.loads((report_dir / "pins.json").read_text(encoding="utf-8"))
        return {str(x) for x in data.get("restored", [])}
    except (OSError, ValueError, AttributeError):
        return set()


def default_embedder(settings: IndexerSettings) -> Embedder:
    from . import embed as embed_module

    return lambda texts: embed_module.embed(texts, settings)


# -- entry point ----------------------------------------------------------------------
def verify(candidates: list[Candidate], settings: IndexerSettings, *,
           embedder: Embedder | None = None, run_id: str | None = None,
           log: Logger | None = None, move_files: bool = True,
           pins: set[str] | None = None) -> VerificationResult:
    log = log or (lambda *_a, **_k: None)
    run_id = run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _, _, report_dir = _dirs(settings)
    first_seen = _load_state(report_dir)
    pins = load_pins(report_dir) if pins is None else pins
    root = Path(settings.corpus_dir).resolve()

    verdicts: list[Verdict] = []
    for c in candidates:
        if not c.label:
            try:
                c.label = c.path.resolve().relative_to(root).as_posix()
            except ValueError:
                c.label = c.path.name
        first_seen.setdefault(c.label, now)
        v = Verdict(candidate=c, final_superseded=c.superseded,
                    sha256=file_sha256(c.path), text_sha256=text_sha256(c.text),
                    chars=len(c.text.strip()))
        v.sample = _norm(c.text)[:SAMPLE_CHARS] if not c.error else ""
        v.checks["quality"] = check_quality(c, settings.verify_min_chars)
        v.checks["completeness"] = check_completeness(c, v)
        verdicts.append(v)

    _apply_recency(verdicts, pins)
    # near-duplicate comparison needs embeddings; only readable documents have a sample
    for v in verdicts:
        if not v.checks["quality"].passed:
            v.sample = ""
    if embedder is None and sum(1 for v in verdicts if v.sample) > 1:
        embedder = default_embedder(settings)
    _apply_uniqueness(verdicts, first_seen, settings.near_dup_threshold, embedder)

    for v in verdicts:
        if v.failed("quality") or v.failed("completeness") or v.failed("uniqueness"):
            v.disposition = "quarantine"
        elif v.failed("recency"):
            v.disposition, v.final_superseded = "archive", True
        else:
            v.disposition = "index"
        log("verified", source=v.candidate.label, disposition=v.disposition,
            **{n: v.checks[n].passed for n in CHECKS},
            reasons={n: v.checks[n].reason for n in CHECKS if not v.checks[n].passed})

    result = VerificationResult(run_id=run_id, verdicts=verdicts)
    if move_files:
        _move_files(verdicts, settings, run_id, log)
    _write_report(result, settings, first_seen, log)
    log("verification finished", run_id=run_id, report=result.report_md or "(not written)",
        **result.counts())
    return result


def collect_candidates(documents: list[tuple[Path, str, bool]], settings: IndexerSettings,
                       log: Logger) -> list[Candidate]:
    """Stage 1 of the pipeline: extract every discovered document. An extraction failure is not
    fatal here; it becomes a failed quality check. Files listed twice count once, and files the
    manifest lists but that are gone (quarantined or deleted earlier) are skipped."""
    from . import extract as extract_module

    manifest = load_manifest(settings.corpus_dir)
    candidates: list[Candidate] = []
    listed: set[Path] = set()
    for path, doc_version, superseded in documents:
        if path.resolve() in listed:
            continue
        listed.add(path.resolve())
        if not path.exists():
            log("listed document is missing; skipped", source=path.name)
            continue
        candidate = Candidate(path=path, doc_version=doc_version, superseded=superseded,
                              entry=manifest.get(path.resolve(), {}))
        started = time.perf_counter()
        try:
            candidate.doc = extract_module.extract(str(path), doc_version, superseded)
        except Exception as exc:    # noqa: BLE001
            candidate.error = f"{type(exc).__name__}: {exc}"
            log("extraction failed", source=path.name, error_type=type(exc).__name__,
                error=str(exc))
        else:
            log("extracted", source=path.name, pages=len(candidate.doc.pages),
                seconds=round(time.perf_counter() - started, 2))
        candidates.append(candidate)
    return candidates


def main(argv: list[str] | None = None) -> int:
    """python -m pipeline.verify --corpus corpus/raw [--no-move]: verify without indexing."""
    import argparse
    import dataclasses

    from .config import get_settings
    from .run import discover, log

    parser = argparse.ArgumentParser(description="Verify the corpus without building an index")
    parser.add_argument("--corpus", help="corpus directory (default: CORPUS_DIR)")
    parser.add_argument("--no-move", action="store_true",
                        help="report only: do not quarantine or archive any file")
    parser.add_argument("--quiet", action="store_true", help="hide the JSON log lines")
    args = parser.parse_args(argv)
    settings = get_settings()
    if args.corpus:
        settings = dataclasses.replace(settings, corpus_dir=args.corpus)
    quiet: Logger = (lambda *_a, **_k: None) if args.quiet else log
    candidates = collect_candidates(discover(settings.corpus_dir), settings, quiet)
    if not candidates:
        print("no documents to verify")
        return 1
    result = verify(candidates, settings, log=quiet, move_files=not args.no_move)
    print(render_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
