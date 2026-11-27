"""Pure metric functions for the M2 evaluation suite.

No network, no model, no files: everything here is unit-tested in test_metrics.py.
Relevance matching is shared with run_eval.py (source + section prefix, or source + page),
so the numbers agree with the existing CI scorer.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from run_eval import _hit, recall_at_k  # noqa: F401  (recall_at_k is re-exported)


def mean(values: Iterable[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return sum(kept) / len(kept) if kept else None


def stdev(values: Sequence[float]) -> float | None:
    """Population standard deviation; None for an empty list."""
    if not values:
        return None
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def percentile(values: Sequence[float], p: float) -> float | None:
    """Linear-interpolation percentile (same convention as numpy's default). p in [0, 100]."""
    if not values:
        return None
    if not 0 <= p <= 100:
        raise ValueError("p must be between 0 and 100")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * p / 100
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return float(ordered[low])
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have the same length")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _gains(relevant: list[dict], retrieved: list[dict], k: int) -> list[int]:
    """Binary gain per retrieved position. Each relevant item is credited at most once, so
    several chunks of the same relevant section cannot push nDCG above 1."""
    remaining = list(range(len(relevant)))
    gains: list[int] = []
    for ref in retrieved[:k]:
        match = next((i for i in remaining if _hit(relevant[i], ref)), None)
        if match is None:
            gains.append(0)
        else:
            remaining.remove(match)
            gains.append(1)
    return gains


def reciprocal_rank(relevant: list[dict], retrieved: list[dict]) -> float | None:
    """1/rank of the first retrieved item that matches any relevant item; 0 if none.
    None when the question has no relevant items (unanswerable)."""
    if not relevant:
        return None
    for position, ref in enumerate(retrieved, 1):
        if any(_hit(rel, ref) for rel in relevant):
            return 1.0 / position
    return 0.0


def ndcg_at_k(relevant: list[dict], retrieved: list[dict], k: int) -> float | None:
    if not relevant:
        return None
    gains = _gains(relevant, retrieved, k)
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def retrieval_metrics(rows: list[dict]) -> dict:
    """rows: [{"relevant": [...], "retrieved": [...]}]. Unanswerable rows (no relevant) are
    skipped. Returns the four headline retrieval numbers."""
    scored = [r for r in rows if r["relevant"]]
    return {
        "questions": len(scored),
        "recall_at_3": _round(mean(recall_at_k(r["relevant"], r["retrieved"], 3) for r in scored)),
        "recall_at_5": _round(mean(recall_at_k(r["relevant"], r["retrieved"], 5) for r in scored)),
        "mrr": _round(mean(reciprocal_rank(r["relevant"], r["retrieved"]) for r in scored)),
        "ndcg_at_5": _round(mean(ndcg_at_k(r["relevant"], r["retrieved"], 5) for r in scored)),
    }


def embedding_report(paraphrase: Sequence[float], unrelated: Sequence[float]) -> dict:
    """Mean cosine of paraphrase vs unrelated pairs and the gap between the two means.

    `overlap_fraction` is the share of (paraphrase, unrelated) combinations where the unrelated
    pair scored at least as high as the paraphrase pair: 0 means the two sets are separable."""
    if not paraphrase or not unrelated:
        raise ValueError("need at least one paraphrase and one unrelated pair")
    mp = sum(paraphrase) / len(paraphrase)
    mu = sum(unrelated) / len(unrelated)
    combos = len(paraphrase) * len(unrelated)
    bad = sum(1 for p in paraphrase for u in unrelated if u >= p)
    return {
        "paraphrase_pairs": len(paraphrase),
        "unrelated_pairs": len(unrelated),
        "paraphrase_mean_cosine": round(mp, 4),
        "unrelated_mean_cosine": round(mu, 4),
        "similarity_gap": round(mp - mu, 4),
        "overlap_fraction": round(bad / combos, 4),
    }


def determinism_report(first: Sequence[Sequence[float]], second: Sequence[Sequence[float]],
                       tolerance: float = 1e-6) -> dict:
    """Compare two embeddings of the same texts. Deterministic when every pair has
    cosine >= 1 - tolerance."""
    if len(first) != len(second) or not first:
        raise ValueError("need two equally long, non-empty lists of vectors")
    cosines = [cosine(a, b) for a, b in zip(first, second, strict=True)]
    max_abs = max(abs(x - y) for a, b in zip(first, second, strict=True)
                  for x, y in zip(a, b, strict=True))
    return {
        "texts": len(first),
        "min_cosine": round(min(cosines), 8),
        "max_abs_difference": max_abs,
        "tolerance": tolerance,
        "deterministic": min(cosines) >= 1 - tolerance,
    }


def cohens_kappa(a: Sequence[str], b: Sequence[str]) -> float | None:
    """Cohen's kappa for two raters over any labels. None when chance agreement is 1
    (both raters used one single label), because kappa is undefined there."""
    if len(a) != len(b) or not a:
        raise ValueError("need two equally long, non-empty label lists")
    n = len(a)
    observed = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    labels = set(a) | set(b)
    expected = sum((a.count(lab) / n) * (b.count(lab) / n) for lab in labels)
    if expected == 1:
        return None
    return (observed - expected) / (1 - expected)


def agreement_report(judge: Sequence[str], human: Sequence[str]) -> dict:
    kappa = cohens_kappa(judge, human)
    n = len(judge)
    agree = sum(1 for x, y in zip(judge, human, strict=True) if x == y)
    return {
        "n": n,
        "raw_agreement": round(agree / n, 4),
        "cohens_kappa": None if kappa is None else round(kappa, 4),
    }


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def flatten(data: dict, prefix: str = "") -> dict[str, float]:
    """{"a": {"b": 1.5}} -> {"a.b": 1.5}. Keeps only real numbers (bool is not a number here)."""
    out: dict[str, float] = {}
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flatten(value, path))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            out[path] = float(value)
    return out


def compare_runs(runs: Sequence[dict]) -> dict[str, dict]:
    """Per-metric variation across repeated runs of the same suite.

    runs: one nested metrics dict per run. Only metrics present in every run are compared."""
    flat = [flatten(r) for r in runs]
    if not flat:
        return {}
    common = set(flat[0]).intersection(*[set(f) for f in flat[1:]])
    report = {}
    for key in sorted(common):
        values = [f[key] for f in flat]
        spread = max(values) - min(values)
        report[key] = {
            "values": [round(v, 6) for v in values],
            "min": round(min(values), 6),
            "max": round(max(values), 6),
            "range": round(spread, 6),
            "identical": spread == 0,
        }
    return report
