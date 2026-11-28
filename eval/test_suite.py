import json
from pathlib import Path

import pytest

import fake_ollama
import offline
import suite
from judge import LexicalJudge

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"


class FakeEmbedder:
    """The same deterministic hashing embedder the CI fake Ollama serves."""

    def embed(self, texts, kind):
        return [fake_ollama.embed(t) for t in texts]

    def timed(self, texts, kind):
        return self.embed(texts, kind), 0.01 * len(texts)


class FakeAsker:
    """Answers from the golden ground truth: perfect retrieval, abstains when asked to."""

    def __init__(self, questions, chunks, fail_ids=()):
        self.by_question = {q["question"]: q for q in questions}
        self.chunks = chunks
        self.fail_ids = set(fail_ids)

    def ask(self, question):
        q = self.by_question[question]
        if q["id"] in self.fail_ids:
            raise RuntimeError("boom")
        if q.get("must_abstain"):
            return {"answered_from_context": False, "answer": "no", "citations": [],
                    "retrieved": [], "provider": "fake"}, 0.5
        refs = [{"source": r["source"], "section": r["section"], "page": 1}
                for r in q["relevant"]]
        text = offline.evidence_text(refs, self.chunks)
        return {"answered_from_context": True, "answer": text[:120], "citations": refs,
                "retrieved": refs, "provider": "fake"}, 1.0 + len(question) / 100


def _cfg(tmp_path, **over):
    base = dict(golden=FIXTURES / "golden.fixture.yaml", corpus=str(FIXTURES), chunk_size=120,
                chunk_overlap=20, out_dir=tmp_path, profile="smoke", judge="lexical", repeat=2,
                api="http://fake")
    base.update(over)
    return suite.Config(**base)


def _prepare(cfg):
    questions = suite.load_golden(cfg.golden)
    pairs = suite.load_pairs(cfg.pairs)
    chunks = offline.build_chunks(cfg.corpus, cfg.chunk_size, cfg.chunk_overlap)
    return questions, pairs, chunks


def _run(cfg, asker_cls=FakeAsker, **asker_kw):
    questions, pairs, chunks = _prepare(cfg)
    asker = asker_cls(questions, chunks, **asker_kw) if asker_cls else None
    runs = [suite.run_once(cfg, questions, pairs, chunks, FakeEmbedder(), asker,
                           LexicalJudge(0.5)) for _ in range(cfg.repeat)]
    return suite.build_results(cfg, "20260101T000000Z", "t", runs, questions), questions


def test_full_run_is_complete_and_has_every_required_metric(tmp_path):
    results, _ = _run(_cfg(tmp_path))
    assert results["complete"] is True, results["incomplete_reasons"]
    m = results["metrics"]
    assert m["embedding"]["paraphrase_pairs"] >= 15 and m["embedding"]["unrelated_pairs"] >= 15
    for key in ("paraphrase_mean_cosine", "unrelated_mean_cosine", "similarity_gap",
                "embed_seconds_per_chunk"):
        assert key in m["embedding"]
    assert m["embedding"]["determinism"]["deterministic"] is True
    for source in ("retrieval_offline", "retrieval_api"):
        for key in ("recall_at_3", "recall_at_5", "mrr", "ndcg_at_5"):
            assert m[source][key] is not None
    assert len(m["noise"]["distractor_documents"]) >= 3
    assert set(m["noise"]["conditions"]) == {"clean", "distractors", "typos",
                                             "distractors_and_typos"}
    for key in ("faithfulness", "abstention_correctness", "citation_validity",
                "latency_seconds_p50", "latency_seconds_p95"):
        assert m["answer"][key] is not None
    assert m["answer"]["abstention_correctness"] == 1.0
    assert m["answer"]["citation_validity"] == 1.0


def test_two_runs_with_the_same_seed_agree_exactly(tmp_path):
    results, _ = _run(_cfg(tmp_path))
    assert len(results["runs"]) == 2
    assert results["variation"]
    assert all(v["identical"] for v in results["variation"].values())


def test_files_are_written_with_a_timestamp_and_a_summary(tmp_path):
    cfg = _cfg(tmp_path)
    results, _ = _run(cfg)
    json_path, md_path = suite.write_results(cfg, results)
    assert json_path.name == "20260101T000000Z-eval.json"
    assert json.loads(json_path.read_text(encoding="utf-8"))["complete"] is True
    text = md_path.read_text(encoding="utf-8")
    for heading in ("## Embedding", "## Retrieval", "## Noise robustness", "## Answer quality"):
        assert heading in text


def test_missing_corpus_makes_the_run_incomplete(tmp_path):
    cfg = _cfg(tmp_path)
    questions, pairs, _ = _prepare(cfg)
    runs = [suite.run_once(cfg, questions, pairs, [], FakeEmbedder(), None, LexicalJudge())
            for _ in range(2)]
    results = suite.build_results(cfg, "x", "t", runs, questions)
    assert results["complete"] is False
    assert any("retrieval" in r for r in results["incomplete_reasons"])
    assert "INCOMPLETE" in suite.render_summary(results)


def test_no_api_means_incomplete(tmp_path):
    results, _ = _run(_cfg(tmp_path, api=None), asker_cls=None)
    assert results["complete"] is False
    assert any("answer" in r for r in results["incomplete_reasons"])


def test_a_failed_question_makes_the_answer_stage_partial(tmp_path):
    results, _ = _run(_cfg(tmp_path), fail_ids=["f01"])
    assert results["complete"] is False
    assert results["runs"][0]["stages"]["answer"]["status"] == "partial"


def test_a_stage_subset_is_never_complete(tmp_path):
    results, _ = _run(_cfg(tmp_path, stages=("embedding", "retrieval")))
    assert results["complete"] is False
    assert any("only these stages" in r for r in results["incomplete_reasons"])


def test_fewer_repetitions_than_requested_is_incomplete(tmp_path):
    cfg = _cfg(tmp_path, repeat=2)
    questions, pairs, chunks = _prepare(cfg)
    asker = FakeAsker(questions, chunks)
    runs = [suite.run_once(cfg, questions, pairs, chunks, FakeEmbedder(), asker, LexicalJudge())]
    assert suite.build_results(cfg, "x", "t", runs, questions)["complete"] is False
    assert suite.build_results(cfg, "x", "t", runs * 2, questions, interrupted=True)[
        "complete"] is False


def test_unfaithful_answers_lower_the_score(tmp_path):
    class Liar(FakeAsker):
        def ask(self, question):
            resp, sec = super().ask(question)
            if resp["answered_from_context"]:
                resp["answer"] = "Completely unrelated claims about dragons and spaceships."
            return resp, sec

    results, _ = _run(_cfg(tmp_path), asker_cls=Liar)
    assert results["metrics"]["answer"]["faithfulness"] == 0.0


def test_a_must_abstain_question_that_is_answered_is_caught(tmp_path):
    class Eager(FakeAsker):
        def ask(self, question):
            resp, sec = super().ask(question)
            return {"answered_from_context": True, "answer": "sure", "citations": [],
                    "retrieved": [], "provider": "fake"} if not resp["answered_from_context"] \
                else (resp, sec)[0], sec

    results, _ = _run(_cfg(tmp_path), asker_cls=Eager)
    assert results["metrics"]["answer"]["abstention_correctness"] == 0.0


def test_load_pairs_enforces_minimums(tmp_path):
    path = tmp_path / "p.jsonl"
    path.write_text(json.dumps({"id": "p1", "kind": "paraphrase", "a": "x", "b": "y"}) + "\n")
    with pytest.raises(ValueError):
        suite.load_pairs(path)


def test_committed_pairs_meet_the_minimum():
    pairs = suite.load_pairs(HERE / "embedding_pairs.jsonl")
    assert sum(p["kind"] == "paraphrase" for p in pairs) >= 15
    assert sum(p["kind"] == "unrelated" for p in pairs) >= 15
