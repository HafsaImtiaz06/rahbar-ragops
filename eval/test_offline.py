import random
from types import SimpleNamespace

import offline


def _chunk(cid, source, section, text, superseded=False, page=1):
    return SimpleNamespace(id=cid, source=source, section=section, text=text,
                           superseded=superseded, page=page,
                           embed_text=lambda: text)


def test_typos_are_seeded_and_change_the_text():
    q = "What minimum attendance is required to sit the final examination?"
    a = offline.add_typos(q, random.Random(5), 0.3)
    b = offline.add_typos(q, random.Random(5), 0.3)
    assert a == b and a != q
    assert offline.add_typos("a b c", random.Random(1)) == "a b c"


def test_a_typo_is_always_applied_when_a_word_is_eligible():
    for seed in range(20):
        assert offline.add_typos("grades matter", random.Random(seed), 0.0) != "grades matter"


def test_index_filters_superseded_unless_asked_for_an_older_version():
    chunks = [_chunk("a", "new.pdf", "1", "x"), _chunk("b", "old.pdf", "1", "x", True)]
    index = offline.Index(chunks, [[1.0, 0.0], [1.0, 0.0]])
    now = index.search([1.0, 0.0], "What is the rule?", 5)
    older = index.search([1.0, 0.0], "What did the 2023 rules say?", 5)
    assert [r["source"] for r in now] == ["new.pdf"]
    assert {r["source"] for r in older} == {"new.pdf", "old.pdf"}


def test_index_ties_are_broken_by_chunk_id():
    chunks = [_chunk("b", "b.pdf", "1", "x"), _chunk("a", "a.pdf", "1", "x")]
    index = offline.Index(chunks, [[1.0], [1.0]])
    assert [r["source"] for r in index.search([1.0], "q", 5)] == ["a.pdf", "b.pdf"]


def test_evidence_text_joins_chunks_of_the_cited_section():
    chunks = [_chunk("a", "a.pdf", "5.4 Warning", "first"), _chunk("b", "a.pdf", "5.4 Warning",
                                                                 "second"),
              _chunk("c", "a.pdf", "6", "other")]
    text = offline.evidence_text([{"source": "A.pdf", "section": "5.4  warning"}], chunks)
    assert "first" in text and "second" in text and "other" not in text


def test_distractors_load_from_the_repo_and_there_are_at_least_three():
    chunks, names = offline.load_distractors(offline.Path(__file__).parent / "noise_docs", 120, 20)
    assert len(names) >= 3 and chunks
    assert "README.md" not in names
