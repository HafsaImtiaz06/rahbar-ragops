import csv

import pytest

import judge


def _items(n=10):
    return [{"id": f"q{i:02d}", "question": f"question {i}", "answer": f"answer {i}",
             "evidence": f"evidence {i}", "judge_label": "faithful" if i % 2 else "unfaithful",
             "judge_score": 0.5} for i in range(n)]


def test_parse_verdict_accepts_valid_and_rejects_the_rest():
    assert judge.parse_verdict('{"verdict": "Faithful", "reason": "ok"}')["label"] == "faithful"
    assert judge.parse_verdict('{"verdict": "maybe"}')["label"] == "unparseable"
    assert judge.parse_verdict("not json")["label"] == "unparseable"


def test_lexical_judge_separates_supported_from_unsupported():
    j = judge.LexicalJudge(0.6)
    evidence = "A student whose CGPA falls below 2.00 is placed on academic warning."
    assert j("q", "A student below 2.00 CGPA is placed on academic warning.", evidence)[
        "label"] == "faithful"
    assert j("q", "Students receive a free laptop every semester.", evidence)[
        "label"] == "unfaithful"


def test_ollama_judge_needs_url_and_model():
    with pytest.raises(ValueError):
        judge.make_judge("ollama")
    with pytest.raises(ValueError):
        judge.make_judge("nope")


def test_export_is_blind_and_needs_eight(tmp_path):
    out = tmp_path / "labels.csv"
    ids = judge.export_sheet(_items(), 8, seed=1, path=out)
    assert len(ids) == 8
    with out.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert "judge_label" not in rows[0]
    assert all(r["human_label"] == "" for r in rows)
    with pytest.raises(ValueError):
        judge.export_sheet(_items(), 5, seed=1, path=out)
    with pytest.raises(ValueError):
        judge.export_sheet(_items(6), 8, seed=1, path=out)


def test_export_is_reproducible_for_a_seed(tmp_path):
    a = judge.export_sheet(_items(), 8, 7, tmp_path / "a.csv")
    b = judge.export_sheet(_items(), 8, 7, tmp_path / "b.csv")
    assert a == b


def _fill(path, labels):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row, label in zip(rows, labels, strict=False):
        row["human_label"] = label
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=judge.SHEET_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_agreement_is_pending_until_eight_are_labelled(tmp_path):
    sheet = tmp_path / "labels.csv"
    judge.export_sheet(_items(), 8, 1, sheet)
    assert judge.agreement_from_sheet(_items(), sheet)["status"] == "pending"
    _fill(sheet, ["faithful"] * 5)
    report = judge.agreement_from_sheet(_items(), sheet)
    assert report["status"] == "pending" and report["labelled"] == 5


def test_agreement_computed_from_filled_sheet(tmp_path):
    sheet = tmp_path / "labels.csv"
    items = _items()
    ids = judge.export_sheet(items, 8, 1, sheet)
    by_id = {i["id"]: i["judge_label"] for i in items}
    _fill(sheet, [by_id[i] for i in ids])          # a person who agrees with the judge on all 8
    report = judge.agreement_from_sheet(items, sheet)
    assert report["status"] == "complete"
    assert report["n"] == 8 and report["raw_agreement"] == 1.0
    assert report["disagreements"] == []


def test_bad_label_is_rejected(tmp_path):
    sheet = tmp_path / "labels.csv"
    judge.export_sheet(_items(), 8, 1, sheet)
    _fill(sheet, ["yes"] * 8)
    with pytest.raises(ValueError):
        judge.agreement_from_sheet(_items(), sheet)
