#!/usr/bin/env python3
"""D2 - give a trace id (= the request_id in any /api/ask response), print the exact context and prompt.

    python scripts/trace_tool.py <id|prefix> [--source traces/] [--prompt-only]
    kubectl logs -n rahbar -l app=api --tail=500 --max-log-requests=20 | python scripts/trace_tool.py <id> --source -
"""
import argparse
import sys

from _traceio import iter_records, sha256_text

BAR = "=" * 78


def find(trace_id, sources):
    matches = [r for r in iter_records(sources) if r["trace_id"].startswith(trace_id)]
    if not matches:
        sys.exit(f"No trace starting with '{trace_id}' in {sources}")
    if len({m["trace_id"] for m in matches}) > 1:
        sys.exit(f"Ambiguous prefix: {sorted({m['trace_id'] for m in matches})}")
    return matches[-1]


def render(r, prompt_only=False):
    out = []
    p = out.append
    if not prompt_only:
        p(BAR)
        p(f"TRACE {r['trace_id']}   {r['timestamp']}   pod={r.get('pod')}   release={r.get('release')}")
        p(BAR)
        p(f"Query            : {r['user_query']}")
        if r.get("rewritten_query") and r["rewritten_query"] != r["user_query"]:
            p(f"Rewritten query  : {r['rewritten_query']}")
        e = r["embedding"]
        p(f"Embedding        : {e['model']} (version {e['version']})")
        p(f"Collection       : {r['collection']}")
        p(f"Filters applied  : {r['filters']}")
        p(f"Cache            : {r['cache']}      Abstained: {r['abstained']}      Outcome: {r.get('outcome')}")
        p(f"LLM              : {r['llm']}")
        p("")
        p("RETRIEVED (in rank order)")
        p(f"  {'#':<3}{'score':<8}{'page':<6}{'ver':<9}{'old?':<7}source | section")
        for h in r["retrieved"]:
            p(f"  {h['rank']:<3}{h['score']:<8}{str(h.get('page')):<6}{str(h.get('doc_version')):<9}"
              f"{str(h.get('superseded')):<7}{h['doc_id']} | {h.get('section')}")
            p(f"  {'':<33}chunk_id: {h['chunk_id']}")
        rank_order = [h["chunk_id"] for h in r["retrieved"]]
        if r["reranked_order"] == rank_order:
            p(f"Order in prompt  : same as retrieval rank (#1..#{len(rank_order)}): no reranking changed it")
        else:
            nums = {c: i + 1 for i, c in enumerate(rank_order)}
            p("Order in prompt  : retrieval ranks " + str([nums.get(c, c) for c in r["reranked_order"]]))
        p("")
        p("GUARDRAIL ACTIONS")
        for a in r["guardrail_actions"] or ["(none)"]:
            p(f"  - {a}")
        p("")
        p("CONTEXT CHUNKS (text as stored in the trace)")
        by_id = {h["chunk_id"]: h for h in r["retrieved"]}
        for cid in r["reranked_order"] or [h["chunk_id"] for h in r["retrieved"]]:
            h = by_id.get(cid)
            if h:
                p(f"--- [{cid}] {h['doc_id']} p.{h.get('page')} v={h.get('doc_version')} score={h['score']}")
                p(h["text"])
        p("")
    p(BAR)
    p("EXACT PROMPT SENT TO THE MODEL")
    p(BAR)
    p(r["prompt"] if r["prompt"] is not None else
      "(no prompt: the answer came from the cache or the relevance floor abstained before calling the model)")
    if r["prompt"] is not None:
        ok = sha256_text(r["prompt"]) == r["prompt_hash"]
        p("")
        p(f"prompt_hash      : {r['prompt_hash']}  [{'MATCH' if ok else 'MISMATCH - trace was altered'}]")
    if not prompt_only:
        p("")
        p("RESPONSE")
        p(str(r["response"]))
        p(f"Citations        : {r['citations']}")
        p(f"Latency (ms)     : {r['latency_ms']}")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace_id")
    ap.add_argument("--source", nargs="+", default=["traces"])
    ap.add_argument("--prompt-only", action="store_true")
    a = ap.parse_args()
    print(render(find(a.trace_id, a.source), a.prompt_only))
