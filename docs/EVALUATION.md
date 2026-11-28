# Evaluation (M2): metrics, suite, experiment and release gate

Status of this document: the code, the golden set, the unit tests and the manifests exist and are
tested. **No evaluation has been run against the real model, the real corpus PDFs or a cluster by the
author of this change.** Every number you will see in the repository from this work comes from test
fixtures or from a plumbing check with a fake model, and is labelled as such. The first real run is
listed under "Still pending" at the end.

## 1. Commands

Windows CMD, from the repository root, after `python -m venv .venv` and
`.venv\Scripts\pip install -r api\requirements-dev.txt -r indexer\requirements-dev.txt`.
(Building the offline index OCRs the scanned 2023 rules, so Tesseract and Poppler must be installed
and on PATH, exactly as for `eval\chunk_sweep.py`.)

```
:: 1. Check the golden set (structure; add --corpus to check sections against the PDFs)
.venv\Scripts\python eval\golden_set.py
.venv\Scripts\python eval\golden_set.py --corpus corpus\raw

:: 2. Unit tests for the evaluation code
cd eval && ..\.venv\Scripts\python -m pytest -q && cd ..

:: 3. THE ONE COMMAND: the complete suite, twice, with fixed seeds
::    (API reachable, e.g. `kubectl port-forward -n rahbar svc/api 8000:8000` in another window;
::     Ollama running with the embedding and generation models)
.venv\Scripts\python eval\suite.py --api http://localhost:8000 --ollama-url http://localhost:11434 --corpus corpus\raw --judge ollama --judge-model llama3.2:3b

:: 4. Manual faithfulness labels (see section 5), then the release gate
.venv\Scripts\python eval\judge.py export eval\results\<run>-eval.json --n 8
.venv\Scripts\python eval\judge.py agree  eval\results\<run>-eval.json eval\results\<run>-eval-labels.csv
.venv\Scripts\python eval\gate.py
```

`make eval-golden`, `make eval-test`, `make eval-suite ARGS=...`, `make eval-gate` do the same from
WSL. The suite exits 0 only when every stage finished in every repetition; 2 when anything is
missing; 130 when interrupted. It saves `eval/results/<UTC timestamp>-eval.json` (everything) and
`<timestamp>-eval-summary.md` (embedding, retrieval, answer quality, variation) even for an
incomplete run, and the file says `"complete": false` and why.

Seeds: `--seed` (default 1337) fixes the typo generator, the manual-labelling sample and the judge
model's sampling. `--repeat` (default 2) runs the whole suite that many times and `variation` in the
JSON reports every metric's spread. Two things are *not* under the suite's control: the API's
generation (temperature 0.1 from the ConfigMap, no seed is sent), and CPU timing. Expect identical
embedding and offline retrieval numbers between repetitions, and small differences in answer metrics
and latency.

The suite asks the API with `follow_up: true`, which the API already treats as "skip the cache", so
repetitions are not served from Redis. `--use-cache` turns that off.

## 2. What is measured

Golden set: `eval/golden.jsonl`, 33 questions: 21 answerable, 5 unanswerable, 7 hard. Each has
`authorship` (`human` or `synthetic`), `provenance`, and ground truth as `(source, section)` or
`(source, page)`. `eval/golden.yaml` is untouched and stays the team's source of truth (the E2/E4 coverage tooling reads it);
`golden_set.py` warns, and a unit test fails, if it and `golden.jsonl` diverge. New questions written for
M2 have ids `s01`-`s12`.
`python eval/golden_set.py` enforces the minimums (25 / 15 / 5 / 5), unique ids, the 3-500 character
question rule of the API, ground truth on every answerable and hard question, `absent_terms` on every
unanswerable one, and that no question claims a human review without a reviewer's name.

**Embedding** (`eval/embedding_pairs.jsonl`: 16 paraphrase and 16 unrelated pairs, written for
this evaluation, no corpus facts in them).
- Mean cosine of the paraphrase pairs and of the unrelated pairs, and their difference (similarity gap).
- Overlap fraction: share of (paraphrase, unrelated) combinations where the unrelated pair scored at
  least as high. 0 means the two sets separate perfectly.
- Embedding time per chunk: wall time to embed the whole corpus index divided by the number of chunks.
- Deterministic repeated text: 8 texts embedded again, in reverse order and a separate request; passes
  when every pair has cosine >= 1 - 1e-6. The largest absolute component difference is recorded.

**Retrieval** (28 answerable and hard questions; unanswerable ones have no ground truth and are skipped).
- recall@k: per question, the share of its ground-truth items found in the top k (same matching as
  `run_eval.py`: file name equal, and the section equal or a clause-number prefix; or the page equal
  for page-only ground truth). Mean over questions.
- MRR: 1 / rank of the first hit in the returned list (0 if none). nDCG@5: binary gain, each
  ground-truth item credited once, log2 discount, divided by the ideal ordering.
- Reported twice: `retrieval_api` is what the deployed API returned (hybrid score, relevance floor,
  top-k 5); `retrieval_offline` is plain cosine over an in-memory index built by the indexer's own code.
  The gate uses the API one. The offline one exists because noise tests and the C1 experiment need an
  index that can be changed.

**Noise robustness** (offline index): clean; with 4 irrelevant documents added
(`eval/noise_docs/`: gardening, astronomy, football, pasta); with seeded typos in every question
(rate 0.15, always at least one); and both. The summary shows each metric and its change from clean.

**End to end** (through the API): faithfulness, abstention correctness (share of unanswerable questions
the system declined), over-abstention (share of answerable ones it declined), citation validity
(share of citations whose source and section are in the retrieved list, the same check as
`run_eval.py`; it does not check that the cited text supports the claim, faithfulness does),
p50 and p95 latency (client wall time, linear interpolation).

## 3. Release gate (C2)

`eval/gate.py` with thresholds in `eval/gate.yaml`. Exit 0 pass, 1 gate failed, 2 cannot decide.

- recall@5 may not fall more than `max_recall_at_5_drop_pp` (2.0) percentage points below baseline
- faithfulness at least `min_faithfulness` (0.80)
- p95 latency at most `max_p95_latency_seconds` (240, the ConfigMap's `LLM_TIMEOUT`)
- abstention correctness no worse than baseline (`abstention_tolerance` 0.0)
- citation validity at least `min_citation_validity` (0.95, the old gate's value)
- optional `max_over_abstention_rate`, off by default

The numeric thresholds are starting values. Nobody has measured the real system on this golden set
yet, so none of them is "tuned"; set them from the first complete run and write the reason here.

The gate fails closed: an incomplete run, a `smoke` run, results marked as a test fixture, a missing
metric or a missing baseline is never a pass (exit 2 or 1). The baseline is `eval/baseline.m2.json`,
created with `python eval/gate.py promote <results.json>` from a complete, full-profile run. It does
not exist yet. The older `eval/baseline.json` is from the 15-question `run_eval.py` golden set and is
not comparable; the gate can read its format, but do not use it as the baseline for this suite.

Demonstrations that need no cluster, using data marked `"fixture": true` (invented numbers,
accepted only with `--allow-fixture`):
```
.venv\Scripts\python eval\gate.py --results eval\fixtures\gate\results_pass.json --baseline eval\fixtures\gate\baseline.json --allow-fixture
.venv\Scripts\python eval\gate.py --results eval\fixtures\gate\results_fail.json --baseline eval\fixtures\gate\baseline.json --allow-fixture
```
The first exits 0, the second exits 1 on recall@5, abstention, faithfulness and p95.
Statistical note: with 28 scored questions, one question that loses its only ground-truth item moves
recall@5 by 3.6 points, so the 2-point rule fails on a single lost question; losing one of two items
moves it by 1.8 points, which is inside the limit.

## 4. Experiment C1: chunk size 400:50 against 200:30

`eval/chunk_sweep.py` (extended: it now also reports MRR, nDCG@5 and embedding time per chunk, and
reads `.jsonl`) supplies the retrieval numbers through `evaluate_config()`; `eval/experiment_c1.py`
wraps it and adds the embedding and end-to-end comparison.

Baseline: 400:50, the value in `k8s/base/configmap.yaml`. Candidate: 200:30. The hypothesis is only a
hypothesis: smaller chunks might help exact-term questions and might split long definitions such as
the cheating list that runs across PDF pages 12 and 13. Method: same corpus, same golden questions, same
model, same seed; only size and overlap change.

```
.venv\Scripts\python eval\experiment_c1.py --corpus corpus\raw --ollama-url http://localhost:11434 --golden eval\golden.jsonl --baseline 400:50 --candidate 200:30
```
That measures embedding (chunks, amplification, time per chunk; pair similarity is the same model
and cannot change) and offline retrieval for both sides. The end-to-end side needs an API whose index
was built with that configuration: add `--api-baseline <url>` while the cluster still runs 400:50;
then set `CHUNK_SIZE=200` and `CHUNK_OVERLAP=30` in the ConfigMap, `make index`, and run again with
`--api-candidate <url> --merge <the earlier c1 json>`. The tool compares the API's chunk count from
`/readyz` with the count the configuration produces and refuses to record end-to-end numbers from an
API that is serving a different configuration. Anything not measured is printed as PENDING with the
command, and the experiment's status is `pending`, never `complete`. **Result: not run.**

## 5. Faithfulness: judge and manual labels

Two judges. `ollama` (default) asks a model for a verdict at temperature 0 with a fixed seed, given the
text of the chunks the answer cites. `lexical` is a deterministic word-overlap proxy for smoke
tests. A model judging a model of the same family is biased towards agreeing, so the LLM judge is only
trusted once it has been compared with a person.

Workflow: `judge.py export` writes a CSV of 8 (or `--n`) answers sampled with the seed, without the
judge's verdict so the labeller is not anchored; a person reads the evidence in the PDF and fills
`human_label` (`faithful` or `unfaithful`); `judge.py agree` reports raw agreement and Cohen's kappa
and writes `<run>-eval-agreement.json`. Below 8 labelled rows it reports `pending` and computes
nothing. **No human labels exist yet; the summary says "pending".**

## 6. Kubernetes evaluation Job (F2)

`k8s/eval-job.yaml`: a ConfigMap `rahbar-eval-thresholds` (the contents of `eval/gate.yaml`), a PVC
`eval-results`, two NetworkPolicies (eval to api, eval to Ollama), and the Job `eval`, which runs
`suite.py` then `gate.py`, mounts the thresholds at `/etc/rahbar/eval`, the corpus PVC read-only and
the results PVC at `/results`, and exits non-zero on an incomplete suite or a failed gate.
`k8s/kustomization.yaml` makes the file its own opt-in kustomization; it is deliberately not in
`k8s/base` so that `make apply` and CI deploys never start an hour-long run. Image: `make eval-image`
(`docker build -f eval/Dockerfile`, context is the repository root).

```
make eval-image
make eval-job
:: change a threshold, no rebuild:
kubectl -n rahbar edit configmap rahbar-eval-thresholds
make eval-job
```
Verified without a cluster: `kustomize build k8s | kubeconform -strict` (5 resources valid; the dev and prod
overlays still build, 31 resources valid) and `eval/test_manifests.py`, which checks that the ConfigMap equals
`eval/gate.yaml` and that editing one threshold in the ConfigMap text flips the gate result for the same
results. **The Job has not been applied to a cluster.** `eval/baseline.m2.json` must be created before the
Job's gate can decide, and the Job image needs the corpus PVC populated (`make corpus-load`).

## 7. Limitations

- 12 of 33 golden questions are synthetic and flagged `needs_manual_review`; their ground truth comes
  from section names and facts visible in this repository (`golden.yaml`, `answer.py`,
  `docs/CORPUS.md`), not from reading the PDFs. s04, s05 and s10 use page numbers, which must equal the
  indexer's `page` for them to match. The 21 carried-over questions (g01-g15, and g16-g21 from the team's E4 commit) are labelled `human` because
  they come from the team's `golden.yaml` and `docs/AI-USAGE.md` calls the golden set the team's own; M2 did not
  verify who wrote them or re-check their sections. `human_reviewed` is false for all 33.
- Unanswerable means absent from the six-document corpus. **Loading Batch B makes g13 (shuttle) and s06
  (library hours) answerable** and changes the fee answers (the 2027 prospectus), so a before-and-after
  comparison across Batch B must either exclude them or use a golden set versioned for each corpus.
  `golden_set.py --corpus` reports absent terms that do occur.
- Offline retrieval has no lexical blend and no relevance floor, so it measures the embedding and the
  chunking, not the deployed retriever.
- Faithfulness evidence is looked up in an offline re-chunk of the corpus by cited source and section;
  it matches the deployed index only if the corpus and chunk settings are the same.
- 28 scored questions is a small sample: see the granularity note in section 3.
- Two repetitions show repeatability, not statistical confidence.
- A full run is long: the earlier 15-question run had a p50 of 75 s per answer, so about 34 minutes per
  repetition and over an hour for two, plus judge calls.
- Failure analysis (`docs/FAILURE-ANALYSIS.md`) is untouched: it needs deliberate breakages and traces
  from a running system, and none was observed here.

## 8. Still pending (needs your machine)

1. A complete real run: `suite.py` against the cluster with Ollama and `corpus/raw`.
2. Labelling 8 answers and running `judge.py agree`.
3. `golden_set.py --corpus corpus\raw`, then a person confirming each synthetic question.
4. `gate.py promote` of that run, then tuning `gate.yaml`.
5. Experiment C1 end to end (both sides), and recording the result here.
6. `make eval-image` and `make eval-job` on minikube; the threshold-change demonstration; screenshots.
