#!/usr/bin/env python3
"""Faithfulness judging and the manual-labelling workflow.

Faithfulness = the share of answered questions whose answer is fully supported by the evidence
passages (the chunks the answer cites). Two judges are available:

  lexical  deterministic word-overlap proxy: no model, cheap, weak. Plumbing and smoke tests.
  ollama   an LLM judge: temperature 0 and a fixed seed, but still a model, so it must be
           checked against people.

The judge's labels are only trusted once at least 8 answers have been labelled by a person
and the agreement is reported. Nothing here invents human labels: the sheet is exported
blank and `agree` refuses to report agreement on fewer than 8 filled-in rows.

    python eval/judge.py export eval/results/<run>-eval.json --n 8 --seed 1337
    (a person fills the human_label column with faithful or unfaithful)
    python eval/judge.py agree  eval/results/<run>-eval.json eval/results/<run>-labels.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

import httpx

from metrics import agreement_report

LABELS = ("faithful", "unfaithful")
MIN_MANUAL_LABELS = 8

STOP = set("""a an and are as at be by can do does for from has have how i if in is it its of on
or that the their there this to was were what when where which who will with you your not no
may must shall than then they them these those also any each other such""".split())
WORD = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


def content_words(text: str) -> list[str]:
    return [w for w in WORD.findall(text.lower()) if w not in STOP]


def lexical_support(answer: str, evidence: str) -> float:
    """Share of the answer's distinct content words that occur in the evidence."""
    words = set(content_words(answer))
    if not words:
        return 0.0
    have = set(content_words(evidence))
    return len(words & have) / len(words)


class LexicalJudge:
    kind = "lexical-overlap"

    def __init__(self, threshold: float = 0.6) -> None:
        self.threshold = threshold
        self.model = f"threshold={threshold}"

    def __call__(self, question: str, answer: str, evidence: str) -> dict:
        score = lexical_support(answer, evidence)
        return {"label": "faithful" if score >= self.threshold else "unfaithful",
                "score": round(score, 4), "reason": "word overlap with the cited passages"}


JUDGE_SYSTEM = (
    "You check whether an answer is faithful to the evidence passages. An answer is faithful only "
    "if every claim in it is stated by, or follows directly from, the passages. Information that "
    "is true but absent from the passages makes it unfaithful. Reply with JSON only: "
    '{"verdict": "faithful" or "unfaithful", "reason": "one short sentence"}.'
)


class OllamaJudge:
    kind = "ollama-llm"

    def __init__(self, url: str, model: str, seed: int = 1337,
                 client: httpx.Client | None = None, timeout: float = 300.0) -> None:
        self.model = model
        self.seed = seed
        self.client = client or httpx.Client(base_url=url, timeout=timeout)

    def __call__(self, question: str, answer: str, evidence: str) -> dict:
        prompt = (f"Question:\n{question}\n\nEvidence passages:\n{evidence}\n\n"
                  f"Answer to check:\n{answer}")
        response = self.client.post("/api/chat", json={
            "model": self.model, "stream": False, "format": "json",
            "options": {"temperature": 0, "seed": self.seed},
            "messages": [{"role": "system", "content": JUDGE_SYSTEM},
                         {"role": "user", "content": prompt}]})
        response.raise_for_status()
        return parse_verdict(response.json()["message"]["content"])


def parse_verdict(raw: str) -> dict:
    """Turn the judge model's text into {"label", "reason"}; label 'unparseable' when it is not
    valid JSON with a recognised verdict (the suite counts those as unjudged, not as faithful)."""
    try:
        data = json.loads(raw)
        verdict = str(data.get("verdict", "")).strip().lower()
    except (ValueError, AttributeError):
        return {"label": "unparseable", "reason": raw[:120]}
    if verdict not in LABELS:
        return {"label": "unparseable", "reason": raw[:120]}
    return {"label": verdict, "reason": str(data.get("reason", ""))[:200]}


def make_judge(kind: str, *, url: str = "", model: str = "", seed: int = 1337,
               threshold: float = 0.6):
    if kind == "lexical":
        return LexicalJudge(threshold)
    if kind == "ollama":
        if not url or not model:
            raise ValueError("the ollama judge needs --judge-url and --judge-model")
        return OllamaJudge(url, model, seed)
    raise ValueError(f"unknown judge {kind!r}; expected lexical or ollama")


# ---- manual labelling ---------------------------------------------------------------------

SHEET_FIELDS = ["id", "question", "answer", "evidence", "human_label", "notes"]


def export_sheet(items: list[dict], n: int, seed: int, path: Path) -> list[str]:
    """Write a blank labelling sheet for n judged answers. The judge's label is left out on
    purpose so the person is not anchored by it. Returns the chosen ids."""
    if n < MIN_MANUAL_LABELS:
        raise ValueError(f"label at least {MIN_MANUAL_LABELS} answers")
    usable = [i for i in items if i.get("judge_label") in LABELS]
    if len(usable) < n:
        raise ValueError(f"only {len(usable)} judged answers in the results; need {n}")
    chosen = random.Random(seed).sample(usable, n)
    chosen.sort(key=lambda i: i["id"])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_FIELDS)
        writer.writeheader()
        for item in chosen:
            writer.writerow({"id": item["id"], "question": item["question"],
                             "answer": item["answer"], "evidence": item["evidence"],
                             "human_label": "", "notes": ""})
    return [i["id"] for i in chosen]


def agreement_from_sheet(items: list[dict], sheet: Path) -> dict:
    """Compare the judge with the person's labels. status is 'pending' (and nothing is
    computed) when fewer than 8 rows are labelled."""
    by_id = {i["id"]: i for i in items}
    judge, human, ids = [], [], []
    with sheet.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            label = (row.get("human_label") or "").strip().lower()
            if not label:
                continue
            if label not in LABELS:
                raise ValueError(f"{row['id']}: human_label must be faithful or unfaithful, "
                                 f"not {label!r}")
            if row["id"] not in by_id:
                raise ValueError(f"{row['id']} is not in the results file")
            ids.append(row["id"])
            judge.append(by_id[row["id"]]["judge_label"])
            human.append(label)
    if len(human) < MIN_MANUAL_LABELS:
        return {"status": "pending", "labelled": len(human), "needed": MIN_MANUAL_LABELS,
                "message": "fill the human_label column for at least 8 rows"}
    report = agreement_report(judge, human)
    report["status"] = "complete"
    report["disagreements"] = [i for i, j, h in zip(ids, judge, human, strict=True) if j != h]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export", help="write a blank labelling sheet")
    exp.add_argument("results", type=Path)
    exp.add_argument("--n", type=int, default=MIN_MANUAL_LABELS)
    exp.add_argument("--seed", type=int, default=1337)
    exp.add_argument("--out", type=Path)
    agree = sub.add_parser("agree", help="report judge-versus-human agreement")
    agree.add_argument("results", type=Path)
    agree.add_argument("labels", type=Path)
    agree.add_argument("--out", type=Path)
    args = parser.parse_args()

    results = json.loads(args.results.read_text(encoding="utf-8"))
    items = results.get("answer_items") or []

    if args.command == "export":
        out = args.out or args.results.with_name(args.results.stem + "-labels.csv")
        ids = export_sheet(items, args.n, args.seed, out)
        print(f"wrote {out} with {len(ids)} rows: {', '.join(ids)}")
        print("Fill human_label (faithful or unfaithful) by reading the cited source text, then run"
              " the 'agree' command.")
        return 0

    report = agreement_from_sheet(items, args.labels)
    report["judge"] = results.get("config", {}).get("judge")
    out = args.out or args.results.with_name(args.results.stem + "-agreement.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
