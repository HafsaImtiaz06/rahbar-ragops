"""B (M3 part): how often does the LLM judge agree with your hand labels?

CSV columns: trace_id, human_label, judge_label   (labels: faithful / unfaithful, or 1 / 0)
    python eval/judge_agreement.py eval/hand_labels.csv [--out eval/results/judge_agreement.json]
Label at least 8 answers BY HAND from the trace tool output (read the context, then the answer) BEFORE
looking at the judge's label, and include both faithful and unfaithful ones if the suite has any.
"""
import argparse, csv, json
from collections import Counter


def norm(v):
    v = str(v).strip().lower()
    if v in {"1", "true", "faithful", "yes", "supported"}:
        return 1
    if v in {"0", "false", "unfaithful", "no", "unsupported"}:
        return 0
    raise ValueError(f"unrecognised label: {v!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("--out")
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(a.csv)) if r["human_label"].strip() and r["judge_label"].strip()]
    h = [norm(r["human_label"]) for r in rows]; j = [norm(r["judge_label"]) for r in rows]
    n = len(rows)
    if n == 0:
        raise SystemExit("no labelled rows")
    agree = sum(x == y for x, y in zip(h, j)) / n
    pe = sum((h.count(c) / n) * (j.count(c) / n) for c in (0, 1))
    kappa = None if pe == 1 else round((agree - pe) / (1 - pe), 3)
    cm = Counter(zip(h, j))
    res = {"n": n, "agreement": round(agree, 3), "cohen_kappa": kappa,
           "confusion(human,judge)": {f"human={x},judge={y}": cm[(x, y)] for x in (0, 1) for y in (0, 1)},
           "disagreements": [r["trace_id"] for r, x, y in zip(rows, h, j) if x != y],
           "warning": "n < 8: assignment needs at least 8 hand-labelled answers" if n < 8 else None}
    print(json.dumps(res, indent=2))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
