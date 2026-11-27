import math

import pytest

from metrics import (
    agreement_report,
    cohens_kappa,
    compare_runs,
    cosine,
    determinism_report,
    embedding_report,
    ndcg_at_k,
    percentile,
    reciprocal_rank,
    retrieval_metrics,
)

A = {"source": "a.pdf", "section": "1"}
B = {"source": "a.pdf", "section": "2"}
C = {"source": "a.pdf", "section": "3"}


def ref(section, page=1):
    return {"source": "a.pdf", "section": section, "page": page}


def test_cosine_basic_and_zero_vector():
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine([0, 0], [1, 1]) == 0.0
    with pytest.raises(ValueError):
        cosine([1], [1, 2])


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert percentile([10], 95) == 10
    assert percentile(list(range(1, 101)), 95) == pytest.approx(95.05)
    assert percentile([], 50) is None


def test_reciprocal_rank_positions():
    assert reciprocal_rank([A], [ref("9"), ref("1")]) == pytest.approx(0.5)
    assert reciprocal_rank([A], [ref("9")]) == 0.0
    assert reciprocal_rank([], [ref("1")]) is None


def test_ndcg_perfect_and_shifted():
    assert ndcg_at_k([A], [ref("1")], 5) == pytest.approx(1.0)
    assert ndcg_at_k([A], [ref("9"), ref("1")], 5) == pytest.approx(1 / math.log2(3))
    assert ndcg_at_k([A], [ref("9")], 5) == 0.0


def test_ndcg_duplicate_chunks_do_not_exceed_one():
    assert ndcg_at_k([A], [ref("1"), ref("1"), ref("1")], 5) == pytest.approx(1.0)


def test_ndcg_two_relevant_one_found_second():
    got = ndcg_at_k([A, B], [ref("9"), ref("1")], 5)
    ideal = 1 + 1 / math.log2(3)
    assert got == pytest.approx((1 / math.log2(3)) / ideal)


def test_page_only_ground_truth():
    rel = {"source": "a.pdf", "page": 4}
    assert reciprocal_rank([rel], [ref("x", page=3), ref("y", page=4)]) == pytest.approx(0.5)


def test_retrieval_metrics_skips_unanswerable():
    rows = [{"relevant": [A], "retrieved": [ref("1")]},
            {"relevant": [B], "retrieved": [ref("9")]},
            {"relevant": [], "retrieved": [ref("1")]}]
    out = retrieval_metrics(rows)
    assert out["questions"] == 2
    assert out["recall_at_5"] == 0.5
    assert out["mrr"] == 0.5
    assert retrieval_metrics([])["recall_at_5"] is None


def test_recall_at_3_versus_5():
    retrieved = [ref("9"), ref("8"), ref("7"), ref("6"), ref("1")]
    out = retrieval_metrics([{"relevant": [A], "retrieved": retrieved}])
    assert out["recall_at_3"] == 0.0
    assert out["recall_at_5"] == 1.0


def test_embedding_report_gap_and_overlap():
    r = embedding_report([0.9, 0.8], [0.2, 0.1])
    assert r["similarity_gap"] == pytest.approx(0.7)
    assert r["overlap_fraction"] == 0.0
    assert embedding_report([0.5], [0.6])["overlap_fraction"] == 1.0
    with pytest.raises(ValueError):
        embedding_report([], [0.1])


def test_determinism_report():
    same = determinism_report([[1.0, 2.0]], [[1.0, 2.0]])
    assert same["deterministic"] is True
    diff = determinism_report([[1.0, 0.0]], [[0.0, 1.0]])
    assert diff["deterministic"] is False


def test_kappa_and_agreement():
    j = ["faithful", "faithful", "unfaithful", "unfaithful"]
    assert cohens_kappa(j, j) == pytest.approx(1.0)
    assert cohens_kappa(["faithful"] * 3, ["faithful"] * 3) is None
    h = ["faithful", "unfaithful", "unfaithful", "unfaithful"]
    rep = agreement_report(j, h)
    assert rep["raw_agreement"] == 0.75
    assert rep["cohens_kappa"] == pytest.approx(0.5)


def test_compare_runs_reports_range_and_identity():
    runs = [{"a": {"x": 0.5, "y": 1.0}}, {"a": {"x": 0.6, "y": 1.0}}]
    out = compare_runs(runs)
    assert out["a.x"]["range"] == pytest.approx(0.1)
    assert out["a.y"]["identical"] is True
