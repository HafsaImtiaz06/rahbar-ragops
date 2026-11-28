# Observability and traceability (Part D) and cluster observability (F4, F5)

Code: `api/app/obs/tracing.py` (trace record), `api/app/obs/guardrails.py` (injection rail), `scripts/trace_tool.py` (D2),
`scripts/trace_metrics.py` (D3), `api/tests/test_obs.py`. `answer_question()` is a traced wrapper around the original
`_answer_question()`, so `routes/ask.py` is unchanged. The trace id is the request id: it is the `X-Request-ID` header
of the response, and you can send your own.

## D1: one JSON line per query
Written to stdout (`kubectl logs`) and to `$TRACE_DIR/traces-<pod>.jsonl` on the `traces` PVC (mounted at `/data/traces`).

| D1 item in the brief | Field |
|---|---|
| trace id, timestamp, user query, rewritten query | `trace_id`, `timestamp`, `user_query`, `rewritten_query` (null: the pipeline does not rewrite) |
| embedding model and version, collection | `embedding.model`, `embedding.version` (= index id), `collection` (resolved name, e.g. `rahbar-<timestamp>`) |
| chunk ids, document ids, versions, scores | `retrieved[]`: `chunk_id`, `doc_id`, `doc_version`, `superseded`, `score`, `section`, `page`, `text` |
| filters, reranked order | `filters`, `reranked_order` (the order the chunks appear in the prompt) |
| prompt hash or full prompt, model, parameters | `prompt` (full text), `prompt_hash` (sha256), `llm.provider`, `llm.model`, `llm.params` |
| response, citations | `response`, `citations[]` |
| guardrail actions | `guardrail_actions[]` |
| latency per stage, cache | `latency_ms.{retrieve,generate,total}`, `cache` (`hit`/`miss`) |
| also | `abstained`, `outcome`, `release` (`live`/`candidate`), `pod` |

Verified offline with the stub providers and a provider that builds the prompt like the Ollama one: one record per
request, `prompt` filled when the model is called, `x-request-id` equals `trace_id`. A cache hit or an abstention has
`prompt: null` because no model call (and so no prompt) happened.

## D2: print the exact context and prompt for a trace id
```bash
ID=demo-1
kubectl -n rahbar port-forward svc/api 18000:8000 &
curl -s -X POST localhost:18000/api/ask -H 'Content-Type: application/json' -H "X-Request-ID: $ID" \
     -d '{"question": "What CGPA puts a student on academic warning?"}'
kubectl -n rahbar logs -l app=api --tail=500 --max-log-requests=20 --prefix=false | python scripts/trace_tool.py "$ID" --source -
python scripts/trace_tool.py "$ID" --source traces/            # same, from the copied trace files
python scripts/trace_tool.py "$ID" --source traces/ --prompt-only
```
The tool also re-hashes the prompt and reports a mismatch with `prompt_hash`.

## D3: metrics
`scripts/trace_metrics.py` summarises a log over a time window: top-1 similarity distribution, abstention rate, cache hit
rate, p50/p95 latency per stage, injection actions flagged, and which collection/embed model/release served the queries.
```bash
mkdir -p traces && kubectl -n rahbar exec deploy/api -- tar c -C /data/traces . | tar x -C traces/
python scripts/trace_metrics.py --source traces/ --last 24h --out eval/results/trace_summary.json
kubectl -n rahbar port-forward deploy/api 18000:8000 & curl -s localhost:18000/metrics | grep rahbar_trace   # F4
```
The `/metrics` endpoint also carries `rahbar_trace_top1_similarity` (histogram) and `rahbar_trace_injection_flagged_total`.
Each API pod writes its own file and `kubectl exec deploy/api` reaches one pod; the PVC is shared, so the tar above
collects the files of all pods that mount it.

## D4: guardrail logging
Two rails write into `guardrail_actions`:
- `prompt_delimiter_escape` / `applied`: the existing defence (chunk text is HTML-escaped inside `<evidence>`, so it cannot close the block).
  It is a neutraliser, so it never said when it mattered.
- `injection_filter` / `flagged_chunk`: a detector over the retrieved chunks (instruction-override patterns such as "ignore previous
  instructions"). Each action records `chunk_id`, `doc_id`, `score` and the matching `pattern`. The mode is `flag`: the chunk is kept
  because the escaping and the prompt rules already neutralise it, and the action is logged; `drop` mode (remove the chunk) exists in
  `scan_chunks` but is not used. Flagged queries feed `rahbar_trace_injection_flagged_total` and the D3 summary.

Run offline against M1's `corpus/batch-b/campus-notices-injected.md`, the detector flags only the injected paragraph and none of the
benign ones. Test on the cluster once Batch B is ingested: ask "Are fees waived on campus?" and open that trace with `trace_tool.py`.
Limits (state them in the report): it is a pattern list, so a paraphrased injection can pass; it inspects chunks, not the question.

## F5: resources
```bash
kubectl -n rahbar port-forward deploy/api 18000:8000 &
python scripts/load_test.py --questions eval/golden.yaml --n 60 --c 4            # cached path
python scripts/load_test.py --questions eval/golden.yaml --n 60 --c 4 --bust     # uncached path (report both)
scripts/measure_resources.sh '^api-|^chroma-|^redis-' 120 eval/results/resources_serving.csv   # in a second terminal during the load
scripts/measure_resources.sh 'indexer' 300 eval/results/resources_indexer.csv                  # while make index runs
scripts/measure_resources.sh '^eval-'  300 eval/results/resources_eval.csv                     # while make eval-job runs
```
`summarize_resources.py` turns each CSV into suggested requests and limits (rule printed in its header). Copy the measured
numbers into the report and the manifests.
