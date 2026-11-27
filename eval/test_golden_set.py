import json
from pathlib import Path

import golden_set

HERE = Path(__file__).resolve().parent


def _row(i, category="answerable", **over):
    row = {"id": f"q{i}", "question": f"question number {i}", "category": category,
           "type": "ordinary", "authorship": "human", "provenance": "test",
           "relevant": [{"source": "a.pdf", "section": "1"}], "must_abstain": False}
    if category == "unanswerable":
        row.update(relevant=[], must_abstain=True, absent_terms=["zzz"])
    if category == "hard":
        row["hard_reason"] = "because"
    row.update(over)
    return row


def _set(n_ans=15, n_un=5, n_hard=5):
    rows = [_row(i) for i in range(n_ans)]
    rows += [_row(100 + i, "unanswerable") for i in range(n_un)]
    rows += [_row(200 + i, "hard") for i in range(n_hard)]
    return rows


def test_the_committed_golden_file_is_valid():
    rows = golden_set.load(HERE / "golden.jsonl")
    assert golden_set.validate(rows) == []
    info = golden_set.summary(rows)
    assert info["questions"] >= 25
    assert info["by_category"]["answerable"] >= 15
    assert info["by_category"]["unanswerable"] >= 5
    assert info["by_category"]["hard"] >= 5
    assert set(info["by_authorship"]) <= {"human", "synthetic"}


def test_committed_file_does_not_claim_human_review():
    rows = golden_set.load(HERE / "golden.jsonl")
    assert not any(r.get("human_reviewed") for r in rows)
    assert all(r["needs_manual_review"] for r in rows if r["authorship"] == "synthetic")


def test_valid_set_passes():
    assert golden_set.validate(_set()) == []


def test_too_few_questions_reported():
    errors = golden_set.validate(_set(n_ans=3))
    assert any("answerable" in e for e in errors)
    assert any("need at least 25" in e for e in errors)


def test_missing_ground_truth_and_bad_label():
    rows = _set()
    rows[0]["relevant"] = []
    rows[1]["authorship"] = "robot"
    errors = golden_set.validate(rows)
    assert any("q0" in e and "relevant" in e for e in errors)
    assert any("q1" in e and "authorship" in e for e in errors)


def test_unanswerable_rules():
    rows = _set()
    rows[15]["must_abstain"] = False
    rows[16]["absent_terms"] = []
    errors = golden_set.validate(rows)
    assert any("q100" in e and "must_abstain" in e for e in errors)
    assert any("q101" in e and "absent_terms" in e for e in errors)


def test_synthetic_must_be_flagged_and_review_needs_a_name():
    rows = _set()
    rows[0].update(authorship="synthetic")
    rows[1].update(human_reviewed=True)
    errors = golden_set.validate(rows)
    assert any("q0" in e and "needs_manual_review" in e for e in errors)
    assert any("q1" in e and "reviewed_by" in e for e in errors)


def test_duplicate_ids_and_question_length():
    rows = _set()
    rows[1]["id"] = rows[0]["id"]
    rows[2]["question"] = "x"
    errors = golden_set.validate(rows)
    assert any("duplicate" in e for e in errors)
    assert any("3-500" in e for e in errors)


def test_page_or_section_required():
    rows = _set()
    rows[0]["relevant"] = [{"source": "a.pdf"}]
    assert any("section' or 'page" in e for e in golden_set.validate(rows))


def test_invalid_json_line_is_reported(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(_row(1)) + "\n{not json\n", encoding="utf-8")
    try:
        golden_set.load(path)
    except SystemExit as exc:
        assert "bad.jsonl:2" in str(exc)
    else:
        raise AssertionError("expected SystemExit")


class _Chunk:
    def __init__(self, source, section, page, text):
        self.source, self.section, self.page, self.text = source, section, page, text


def test_check_against_chunks_flags_missing_section_and_present_absent_term():
    rows = [_row(1, relevant=[{"source": "a.pdf", "section": "9"}]),
            _row(2, "unanswerable", absent_terms=["shuttle"])]
    chunks = [_Chunk("a.pdf", "1", 1, "text about the Shuttle bus")]
    warnings = golden_set.check_against_chunks(rows, chunks)
    assert any("q1" in w and "section" in w for w in warnings)
    assert any("q2" in w and "shuttle" in w for w in warnings)


def test_golden_jsonl_is_in_sync_with_golden_yaml():
    rows = golden_set.load(HERE / "golden.jsonl")
    assert golden_set.yaml_drift(rows, HERE / "golden.yaml") == [], \
        "golden.yaml changed: add or update the same question in eval/golden.jsonl"


def test_yaml_drift_detects_missing_and_changed(tmp_path):
    path = tmp_path / "g.yaml"
    path.write_text("questions:\n  - {id: a1, question: hello there, relevant: []}\n"
                    "  - {id: a2, question: other one, relevant: []}\n", encoding="utf-8")
    rows = [_row(1, id="a1", question="changed text", relevant=[])]
    problems = golden_set.yaml_drift(rows, path)
    assert any("a1 differs" in p for p in problems)
    assert any("a2 is in" in p for p in problems)
