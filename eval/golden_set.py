#!/usr/bin/env python3
"""Load and validate eval/golden.jsonl (one JSON object per line).

    python eval/golden_set.py                         # structure checks only
    python eval/golden_set.py --corpus corpus/raw     # also check sections and absent topics
                                                      #   against the real corpus (needs the PDFs)

Fields
  id, question                    question is 3-500 characters (the API contract)
  category                        answerable | unanswerable | hard
  type                            free label used for per-type recall (ordinary, exact, version, ...)
  relevant                        [{source, section} or {source, page}]; required unless unanswerable
  must_abstain                    true for unanswerable questions
  expect                          optional substring the answer should contain
  absent_terms                    unanswerable only: words that must NOT occur in the corpus
  hard_reason                     hard only: why the question is hard
  authorship                      human | synthetic
  human_reviewed, reviewed_by     review is only recorded when a person did it
  needs_manual_review             required (true) for synthetic questions that are not reviewed
  provenance                      where the question and its ground truth came from
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from run_eval import read_jsonl, section_matches

HERE = Path(__file__).resolve().parent
DEFAULT_GOLDEN = HERE / "golden.jsonl"

CATEGORIES = ("answerable", "unanswerable", "hard")
AUTHORSHIP = ("human", "synthetic")
REQUIRED = ("id", "question", "category", "type", "authorship", "provenance")


def load(path: Path) -> list[dict]:
    if path.suffix != ".jsonl":
        raise ValueError(f"{path} is not a .jsonl file")
    return read_jsonl(path)


def validate(rows: list[dict], *, min_total: int = 25, min_answerable: int = 15,
             min_unanswerable: int = 5, min_hard: int = 5) -> list[str]:
    """Return a list of problems; an empty list means the file is valid."""
    errors: list[str] = []
    seen: set[str] = set()

    for number, row in enumerate(rows, 1):
        label = row.get("id", f"line {number}")
        for key in REQUIRED:
            if not row.get(key):
                errors.append(f"{label}: missing '{key}'")

        if row.get("id") in seen:
            errors.append(f"{label}: duplicate id")
        seen.add(row.get("id", ""))

        question = row.get("question", "")
        if not isinstance(question, str) or not 3 <= len(question.strip()) <= 500:
            errors.append(f"{label}: question must be 3-500 characters")

        category = row.get("category")
        if category not in CATEGORIES:
            errors.append(f"{label}: category must be one of {', '.join(CATEGORIES)}")
        if row.get("authorship") not in AUTHORSHIP:
            errors.append(f"{label}: authorship must be 'human' or 'synthetic'")

        relevant = row.get("relevant") or []
        if category in ("answerable", "hard"):
            if not relevant:
                errors.append(f"{label}: {category} question needs ground-truth 'relevant'")
            if row.get("must_abstain"):
                errors.append(f"{label}: {category} question must not set must_abstain")
        if category == "unanswerable":
            if row.get("must_abstain") is not True:
                errors.append(f"{label}: unanswerable question needs must_abstain=true")
            if relevant:
                errors.append(f"{label}: unanswerable question must not list relevant passages")
            if not row.get("absent_terms"):
                errors.append(f"{label}: unanswerable question needs 'absent_terms'")
        if category == "hard" and not row.get("hard_reason"):
            errors.append(f"{label}: hard question needs 'hard_reason'")

        for rel in relevant:
            if not rel.get("source"):
                errors.append(f"{label}: relevant item without 'source'")
            if not rel.get("section") and rel.get("page") is None:
                errors.append(f"{label}: relevant item needs 'section' or 'page'")

        reviewed = row.get("human_reviewed", False)
        if reviewed and not row.get("reviewed_by"):
            errors.append(f"{label}: human_reviewed=true needs 'reviewed_by'")
        if row.get("authorship") == "synthetic" and not reviewed \
                and row.get("needs_manual_review") is not True:
            errors.append(f"{label}: unreviewed synthetic question must set "
                          "needs_manual_review=true")

    counts = Counter(r.get("category") for r in rows)
    if len(rows) < min_total:
        errors.append(f"only {len(rows)} questions; need at least {min_total}")
    if counts["answerable"] < min_answerable:
        errors.append(f"only {counts['answerable']} answerable; need at least {min_answerable}")
    if counts["unanswerable"] < min_unanswerable:
        errors.append(f"only {counts['unanswerable']} unanswerable; "
                      f"need at least {min_unanswerable}")
    if counts["hard"] < min_hard:
        errors.append(f"only {counts['hard']} hard; need at least {min_hard}")
    return errors


def summary(rows: list[dict]) -> dict:
    return {
        "questions": len(rows),
        "by_category": dict(sorted(Counter(r.get("category") for r in rows).items())),
        "by_authorship": dict(sorted(Counter(r.get("authorship") for r in rows).items())),
        "human_reviewed": sum(1 for r in rows if r.get("human_reviewed")),
        "needing_manual_review": sum(1 for r in rows if r.get("needs_manual_review")),
    }


def yaml_drift(rows: list[dict], yaml_path: Path) -> list[str]:
    """eval/golden.yaml is still the team's source of truth (query_coverage.py and CI read it).
    Report every question there that is missing from golden.jsonl or differs in text or ground
    truth, so the two files cannot silently diverge."""
    import yaml

    if not yaml_path.exists():
        return []
    by_id = {r["id"]: r for r in rows}
    problems = []
    for q in (yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}).get("questions", []):
        row = by_id.get(q["id"])
        if row is None:
            problems.append(f"{q['id']} is in {yaml_path.name} but not in golden.jsonl")
        elif row["question"] != q["question"] or (row.get("relevant") or []) != (
                q.get("relevant") or []):
            problems.append(f"{q['id']} differs between {yaml_path.name} and golden.jsonl")
    return problems


def check_against_chunks(rows: list[dict], chunks: list) -> list[str]:
    """Compare the golden set with the chunks the indexer produces from the real corpus.

    Returns warnings (not errors): a missing section or page means the ground truth must be
    re-checked by a person; an absent term that DOES occur means an 'unanswerable' question may
    in fact be answerable."""
    warnings: list[str] = []
    for row in rows:
        label = row["id"]
        for rel in row.get("relevant") or []:
            same_source = [c for c in chunks if c.source.lower() == rel["source"].lower()]
            if not same_source:
                warnings.append(f"{label}: source '{rel['source']}' is not in the corpus")
            elif rel.get("section"):
                if not any(section_matches(rel["section"], c.section) for c in same_source):
                    warnings.append(f"{label}: no chunk of '{rel['source']}' has section "
                                    f"'{rel['section']}'")
            elif not any(c.page == rel["page"] for c in same_source):
                warnings.append(f"{label}: no chunk of '{rel['source']}' is on page "
                                f"{rel['page']}")
        for term in row.get("absent_terms") or []:
            hits = [c for c in chunks if term.lower() in c.text.lower()]
            if hits:
                first = hits[0]
                warnings.append(f"{label}: absent term '{term}' occurs in {len(hits)} chunk(s), "
                                f"e.g. {first.source} page {first.page}; this question may be "
                                "answerable")
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the golden JSONL file.")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--corpus", help="corpus directory: also check sections and absent terms")
    args = parser.parse_args()

    rows = load(args.golden)
    errors = validate(rows)
    print(f"{args.golden}: {summary(rows)}")

    for problem in yaml_drift(rows, HERE / "golden.yaml"):
        print(f"  WARNING {problem}: update golden.jsonl")

    if args.corpus:
        sys.path.insert(0, str(HERE.parent / "indexer"))
        from offline import build_chunks

        for warning in check_against_chunks(rows, build_chunks(args.corpus, 400, 50)):
            print(f"  WARNING {warning}")

    if errors:
        print("\nINVALID")
        for message in errors:
            print(f"  - {message}")
        return 1
    print("valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
