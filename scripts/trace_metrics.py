#!/usr/bin/env python3
"""D3 - summarise the trace log over a time window.

    python scripts/trace_metrics.py --source traces/ --last 24h --out eval/results/trace_summary.json
    kubectl logs -n rahbar -l app=api --tail=2000 --max-log-requests=20 | python scripts/trace_metrics.py --source - --last 1h
(Live numbers for Prometheus are on the API's existing /metrics; this is the offline report over the trace log.)
"""
import argparse
import json
import re
from datetime import datetime, timedelta, timezone

from _traceio import iter_records


def parse_ts(s):
    d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def parse_duration(s):
    m = re.fullmatch(r"(\d+)([smhd])", s)
    if not m:
        raise SystemExit("--last like 90m, 24h, 7d")
    return timedelta(**{{"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}[m[2]]: int(m[1])})


def percentile(values, q):
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return round(v[lo] + (v[hi] - v[lo]) * (k - lo), 3)


def is_inj(a):
    return "injection" in str(a.get("rail", "")).lower() and a.get("action") != "pass"


def summarize(recs):
    n = len(recs)
    top1 = [max(h["score"] for h in r["retrieved"]) for r in recs if r.get("retrieved")]
    bins = {f"{i/10:.1f}-{(i+1)/10:.1f}": 0 for i in range(10)}
    for s in top1:
        i = min(max(int(s * 10), 0), 9)
        bins[f"{i/10:.1f}-{(i+1)/10:.1f}"] += 1
    stages = {}
    for r in recs:
        for k, v in r.get("latency_ms", {}).items():
            stages.setdefault(k, []).append(v)
    by = {}
    for r in recs:
        key = f"{r.get('collection')} | embed={r.get('embedding', {}).get('model')} | release={r.get('release')}"
        by[key] = by.get(key, 0) + 1
    return {
        "n_queries": n,
        "window": {"first": min((r["timestamp"] for r in recs), default=None),
                   "last": max((r["timestamp"] for r in recs), default=None)},
        "top1_similarity": {"n": len(top1), "min": percentile(top1, 0), "p25": percentile(top1, .25),
                            "median": percentile(top1, .5), "p75": percentile(top1, .75), "max": percentile(top1, 1),
                            "mean": round(sum(top1) / len(top1), 4) if top1 else None, "histogram": bins},
        "abstention_rate": round(sum(1 for r in recs if r.get("abstained")) / n, 4) if n else None,
        "cache_hit_rate": round(sum(1 for r in recs if r.get("cache") == "hit") / n, 4) if n else None,
        "latency_ms": {k: {"p50": percentile(v, .5), "p95": percentile(v, .95)} for k, v in sorted(stages.items())},
        "injection": {"queries_flagged": sum(1 for r in recs if any(is_inj(a) for a in r.get("guardrail_actions", []))),
                      "actions": sum(1 for r in recs for a in r.get("guardrail_actions", []) if is_inj(a))},
        "served_by": by,
    }


def table(s):
    t = s["top1_similarity"]
    o = [f"Trace summary: {s['n_queries']} queries, {s['window']['first']} -> {s['window']['last']}", "",
         f"  Abstention rate     : {s['abstention_rate']}",
         f"  Cache hit rate      : {s['cache_hit_rate']}",
         f"  Injection flagged   : {s['injection']['queries_flagged']} queries ({s['injection']['actions']} actions)",
         f"  Top-1 similarity    : min {t['min']}  p25 {t['p25']}  median {t['median']}  p75 {t['p75']}  max {t['max']}",
         "  Top-1 histogram     : " + "  ".join(f"{k}:{v}" for k, v in t["histogram"].items() if v), "",
         f"  {'stage':<14}{'p50 ms':>10}{'p95 ms':>10}"]
    o += [f"  {k:<14}{v['p50']:>10}{v['p95']:>10}" for k, v in s["latency_ms"].items()]
    o += [""] + [f"  served by {k}: {v}" for k, v in s["served_by"].items()]
    return "\n".join(o)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", nargs="+", default=["traces"])
    ap.add_argument("--last"); ap.add_argument("--since"); ap.add_argument("--until")
    ap.add_argument("--out"); ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    since = parse_ts(a.since) if a.since else None
    until = parse_ts(a.until) if a.until else None
    if a.last:
        since = datetime.now(timezone.utc) - parse_duration(a.last)
    recs = [r for r in iter_records(a.source)
            if (not since or parse_ts(r["timestamp"]) >= since) and (not until or parse_ts(r["timestamp"]) <= until)]
    s = summarize(recs)
    if a.out:
        json.dump(s, open(a.out, "w"), indent=2)
    print(json.dumps(s, indent=2) if a.json else table(s))
