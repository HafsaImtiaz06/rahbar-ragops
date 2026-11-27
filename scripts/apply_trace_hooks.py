#!/usr/bin/env python3
"""Adds the D1/D4 trace hooks to api/app/services/answer.py (written against the dev branch, 8 anchored edits).

    python scripts/apply_trace_hooks.py --check      # dry run: reports which anchors match
    python scripts/apply_trace_hooks.py              # applies; safe to re-run (already-applied edits are skipped)
    git diff api/app/services/answer.py              # review, then `git restore` it if you do not like it

Each edit replaces text that must occur EXACTLY ONCE. If any anchor is missing (a teammate changed the file),
nothing is written and you are told which one; then make that edit by hand from the NEW text below.
"""
import argparse
import sys
from pathlib import Path

EDITS = []


def edit(name, old, new, marker):
    EDITS.append((name, old, new, marker))


edit("imports",
     "from ..logjson import request_id_var\n",
     "from ..logjson import request_id_var\nfrom ..obs.guardrails import escape_action, scan_chunks\nfrom ..obs.tracing import Trace, current_trace\n",
     "from ..obs.tracing import Trace")

edit("helpers",
     "def _refs(passages: list[Passage]) -> list[RetrievedRef]:\n",
     '''def _trace_index(trace: Trace, retriever: Retriever, settings: Settings) -> None:
    """Record the collection/index the retriever actually resolved, and the pinned embedding model."""
    try:
        index_id = getattr(retriever, "index_id", lambda: None)()
    except Exception:  # tracing must never break answering
        index_id = None
    trace.set_collection(index_id or getattr(settings, "chroma_collection", "unknown"),
                         getattr(settings, "embed_model", None))


def _trace_guardrails(trace: Trace, passages: list[Passage]) -> None:
    """D4: log the injection detector's findings and that the delimiter escaping was applied."""
    if not passages:
        return
    _, actions = scan_chunks(passages)
    for action in actions:
        trace.add_guardrail(**action)
    trace.add_guardrail(**escape_action(len(passages)))
    trace.set_reranked([p.chunk_id for p in passages])   # order of the passages that reach the prompt


def _refs(passages: list[Passage]) -> list[RetrievedRef]:
''',
     "def _trace_index(")

edit("entrypoint wrapper",
     '''def answer_question(
    question: str,
    retriever: Retriever,
    provider: LLMProvider,
    settings: Settings,
    cache: SemanticCache | None = None,
    follow_up: bool = False,
) -> AskResponse:
    older = wants_older_version(question)
''',
     '''def answer_question(
    question: str,
    retriever: Retriever,
    provider: LLMProvider,
    settings: Settings,
    cache: SemanticCache | None = None,
    follow_up: bool = False,
) -> AskResponse:
    """Traced entry point (D1): exactly one trace record per request, even when the pipeline raises."""
    rid = request_id_var.get()
    trace = Trace(question, trace_id=rid if rid and rid != "-" else None)
    trace.rec["llm"]["params"].update(
        top_k=settings.top_k, relevance_floor=settings.relevance_floor,
        temperature=getattr(settings, "temperature", None))
    trace.rec["llm"]["provider"] = provider.name
    with trace.activate():
        try:
            response = _answer_question(question, retriever, provider, settings, cache, follow_up, trace)
        except Exception as exc:
            trace.add_guardrail(rail="pipeline", action="exception", error=type(exc).__name__)
            trace.finish(None, [], abstained=False)
            raise
    trace.finish(response.answer, response.citations, abstained=not response.answered_from_context)
    return response


def _answer_question(
    question: str,
    retriever: Retriever,
    provider: LLMProvider,
    settings: Settings,
    cache: SemanticCache | None,
    follow_up: bool,
    trace: Trace,
) -> AskResponse:
    older = wants_older_version(question)
''',
     "def _answer_question(")

edit("cache hit",
     '                    metrics.CACHE_LOOKUPS.labels(result="hit").inc()\n',
     '                    metrics.CACHE_LOOKUPS.labels(result="hit").inc()\n'
     '                    trace.set_cache(True)\n'
     '                    trace.set_collection(index_id, getattr(settings, "embed_model", None))\n'
     '                    trace.set_outcome("cache_hit")\n',
     "trace.set_cache(True)")

edit("retrieve",
     '''    started = time.perf_counter()
    passages = _retrieve(retriever, question, settings.top_k, older)
    metrics.RETRIEVAL_SECONDS.observe(time.perf_counter() - started)
''',
     '''    started = time.perf_counter()
    with trace.stage("retrieve"):
        passages = _retrieve(retriever, question, settings.top_k, older)
    metrics.RETRIEVAL_SECONDS.observe(time.perf_counter() - started)
    _trace_index(trace, retriever, settings)
    trace.set_retrieved(passages, filters={"superseded": "included" if older else False,
                                           "relevance_floor": settings.relevance_floor})
''',
     'with trace.stage("retrieve")')

edit("guardrails",
     "    grounded = [p for p in passages if p.score >= settings.relevance_floor]\n",
     "    grounded = [p for p in passages if p.score >= settings.relevance_floor]\n"
     "    _trace_guardrails(trace, grounded)\n",
     "_trace_guardrails(trace, grounded)")

edit("generate",
     '''    started = time.perf_counter()
    validated = generate_validated(question, grounded, provider)
    metrics.GENERATION_SECONDS.observe(time.perf_counter() - started)
''',
     '''    started = time.perf_counter()
    with trace.stage("generate"):          # generation + validation (+ one retry)
        validated = generate_validated(question, grounded, provider)
    metrics.GENERATION_SECONDS.observe(time.perf_counter() - started)
    if validated is not None:
        n_dropped = len(validated[0].citations) - len(validated[1])
        if n_dropped > 0:
            trace.add_guardrail(rail="citation_check", action="dropped_fabricated", n=n_dropped)
''',
     'with trace.stage("generate")')

edit("outcome in _log",
     "         passages: list[Passage], response: AskResponse, reason: str | None) -> None:\n",
     "         passages: list[Passage], response: AskResponse, reason: str | None) -> None:\n"
     "    active = current_trace()\n"
     "    if active is not None:\n"
     "        active.set_outcome(reason)\n",
     "active = current_trace()")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="api/app/services/answer.py")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    path = Path(a.file)
    text = path.read_text(encoding="utf-8")
    problems, todo = [], []
    for name, old, new, marker in EDITS:
        if marker in text:
            print(f"  already applied : {name}")
            continue
        n = text.count(old)
        if n != 1:
            problems.append(f"{name}: anchor found {n} times (need exactly 1)")
            print(f"  ANCHOR PROBLEM  : {name} ({n} matches)")
        else:
            todo.append((name, old, new))
            print(f"  will apply      : {name}")
    if problems:
        sys.exit("\nNothing written. Fix by hand:\n  " + "\n  ".join(problems))
    if a.check or not todo:
        return print("\n(check only)" if a.check else "\nNothing to do.")
    for _, old, new in todo:
        text = text.replace(old, new, 1)
    compile(text, str(path), "exec")          # refuse to write a file with a syntax error
    path.write_text(text, encoding="utf-8")
    print(f"\nPatched {path}: {len(todo)} edit(s). Review with: git diff {path}")


if __name__ == "__main__":
    main()
