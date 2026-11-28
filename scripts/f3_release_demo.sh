#!/usr/bin/env bash
# F3 / C3 controlled release for RAHBAR. One step per invocation so each can be screenshotted.
#   scripts/f3_release_demo.sh <step>
#   steps: candidate-index | candidate-up | eval-candidate | release | bad-release | undo | history | candidate-down
#
# Run from the repository root, with the project venv active (it needs PyYAML):  source $VENV/bin/activate
#
# THE CHANGE (Part C1, M2's experiment): chunk size/overlap 400:50 -> 200:30, built into its own collection.
#   CAND_COLLECTION=cand           base name of the candidate index. NEVER `rahbar-<something>`: the live indexer
#                                  deletes every collection starting with `rahbar-` that is not active/previous.
#   CAND_CHUNK_SIZE=200  CAND_CHUNK_OVERLAP=30
#   CANDIDATE_ENV="CHROMA_COLLECTION=cand"     env overrides of the candidate API AND of the release (default).
#       Gate-failing demo candidate (no new index needed):  CANDIDATE_ENV="TOP_K=1 RELEVANCE_FLOOR=0.0"
#   BAD_ENV="CHROMA_COLLECTION=cand RELEVANCE_FLOOR=0.0 TOP_K=1"   a release that should never have shipped: it
#       serves traffic from a DIFFERENT collection, so the trace after `undo` visibly shows the previous one.
# Overrides land in the Deployment's pod template (kubectl set env), so `kubectl rollout undo` restores the old values.
#
# RECOMMENDED ORDER (so every screenshot shows something different):
#   1. candidate-index (once)      2. candidate-up      3. eval-candidate   -> a candidate that FAILS the gate is never released
#   4. bad-release  -> traffic is served from collection `cand` with TOP_K=1 / floor 0.0 (needs step 1, else /readyz fails and
#      the old pods keep serving: also a valid maxUnavailable:0 screenshot)    5. undo -> trace shows the ORIGINAL collection
#   6. only when a candidate PASSES the gate: release (rolling update, 0 failed requests) and keep it.
#   `rollout undo` returns to the previous revision, so do 4-5 while live is still on the original collection.
#
# CONTRACT with M2's k8s/eval-job.yaml (verified against the file): Job `eval`, label app=eval, backoffLimit 0,
# exit 1 when the gate fails (2 when incomplete). Its URL is hard-coded, so eval-candidate derives a copy
# (scripts/make_candidate_jobs.py) that targets api-candidate and writes to /results/candidate.
set -uo pipefail
NS=${NS:-rahbar}
LIVE=${LIVE:-api}
API_PORT=${API_PORT:-8000}
API_SELECTOR=${API_SELECTOR:-app=api}
OVERLAY=${OVERLAY:-k8s/overlays/dev}
CAND_COLLECTION=${CAND_COLLECTION:-cand}
CAND_CHUNK_SIZE=${CAND_CHUNK_SIZE:-200}
CAND_CHUNK_OVERLAP=${CAND_CHUNK_OVERLAP:-30}
CANDIDATE_ENV=${CANDIDATE_ENV:-"CHROMA_COLLECTION=$CAND_COLLECTION"}
BAD_ENV=${BAD_ENV:-"CHROMA_COLLECTION=$CAND_COLLECTION RELEVANCE_FLOOR=0.0 TOP_K=1"}
QUESTION=${QUESTION:-"What CGPA puts a student on academic warning?"}
CAND_SVC_PORT=${CAND_SVC_PORT:-8000}
if [ -z "${PY:-}" ]; then
  if [ -x "${VENV:-/nonexistent}/bin/python" ]; then PY="$VENV/bin/python"; else PY=python3; fi
fi
mkdir -p k8s/candidate

env_args() { for kv in $1; do printf -- '--env %s ' "$kv"; done; }

wait_job() {   # $1 = job name, $2 = timeout seconds; prints PASS/FAIL, returns 0 only on success
  local end=$((SECONDS + $2)) s f
  while [ $SECONDS -lt $end ]; do
    s=$(kubectl -n "$NS" get job "$1" -o jsonpath='{.status.succeeded}' 2>/dev/null)
    f=$(kubectl -n "$NS" get job "$1" -o jsonpath='{.status.failed}' 2>/dev/null)
    [ "${s:-0}" -ge 1 ] && return 0
    [ "${f:-0}" -ge 1 ] && return 1
    sleep 10
  done
  echo "timed out waiting for job/$1"; return 2
}

ask() {        # ask the live api with a known request id (= trace id), print it
  local id="demo-$(date +%s)" pf
  echo "$id" > /tmp/last_trace_id
  kubectl -n "$NS" port-forward deploy/$LIVE 18000:$API_PORT >/dev/null 2>&1 & pf=$!
  sleep 3
  curl -s -X POST localhost:18000/api/ask -H 'Content-Type: application/json' -H "X-Request-ID: $id" \
       -d "{\"question\": \"$QUESTION\"}"; echo
  kill $pf 2>/dev/null
  echo "trace id: $id"
}

case "${1:-}" in
  candidate-index)   # build the candidate collection with the candidate chunking (the only writer is still an indexer Job)
    kubectl kustomize "$OVERLAY" | $PY scripts/make_candidate_jobs.py indexer --collection "$CAND_COLLECTION" \
        --env CHUNK_SIZE="$CAND_CHUNK_SIZE" --env CHUNK_OVERLAP="$CAND_CHUNK_OVERLAP" \
        > k8s/candidate/indexer-candidate.yaml || exit 1
    kubectl -n "$NS" delete job indexer-candidate --ignore-not-found --wait=true
    kubectl apply -f k8s/candidate/indexer-candidate.yaml
    echo "waiting for indexer-candidate (full embedding of the corpus: slow on CPU)..."
    wait_job indexer-candidate 3600; rc=$?
    kubectl -n "$NS" logs job/indexer-candidate --tail=30
    exit $rc ;;
  candidate-up)
    kubectl kustomize "$OVERLAY" | $PY scripts/make_candidate.py --live "$LIVE" $(env_args "$CANDIDATE_ENV") \
        > k8s/candidate/api-candidate.yaml || exit 1
    kubectl apply -f k8s/candidate/api-candidate.yaml
    kubectl -n "$NS" rollout status deploy/api-candidate --timeout=180s
    kubectl -n "$NS" get deploy,svc,pods -l track=candidate ;;
  eval-candidate)    # the Part B suite + the gate, as a Job, against the candidate Service only
    $PY scripts/make_candidate_jobs.py eval --input k8s/eval-job.yaml \
        --target "http://api-candidate:$CAND_SVC_PORT" \
        --env CHUNK_SIZE="$CAND_CHUNK_SIZE" --env CHUNK_OVERLAP="$CAND_CHUNK_OVERLAP" \
        > k8s/candidate/eval-candidate.yaml || exit 1
    kubectl -n "$NS" delete job eval-candidate --ignore-not-found --wait=true
    kubectl apply -f k8s/candidate/eval-candidate.yaml
    echo "waiting for eval-candidate (two full suite repetitions: can take over an hour on CPU)..."
    wait_job eval-candidate 10800; rc=$?
    kubectl -n "$NS" logs job/eval-candidate --tail=60
    if [ $rc -eq 0 ]; then echo "GATE: PASS (Job succeeded)"; else echo "GATE: FAIL/ERROR (rc=$rc) -> do NOT roll this candidate out"; fi
    exit $rc ;;
  release)           # only after eval-candidate printed GATE: PASS
    kubectl -n "$NS" annotate deploy/$LIVE kubernetes.io/change-cause="release: $CANDIDATE_ENV" --overwrite
    kubectl -n "$NS" set env deploy/$LIVE $CANDIDATE_ENV
    kubectl -n "$NS" rollout status deploy/$LIVE --timeout=240s
    kubectl -n "$NS" rollout history deploy/$LIVE ;;
  bad-release)
    kubectl -n "$NS" annotate deploy/$LIVE kubernetes.io/change-cause="BAD release: $BAD_ENV" --overwrite
    kubectl -n "$NS" set env deploy/$LIVE $BAD_ENV
    kubectl -n "$NS" rollout status deploy/$LIVE --timeout=240s \
      || echo ">> rollout not finishing: old pods keep serving (maxUnavailable: 0, /readyz fails)"
    echo "--- query served by the bad release:"; ask ;;
  undo)
    kubectl -n "$NS" rollout undo deploy/$LIVE
    kubectl -n "$NS" rollout status deploy/$LIVE --timeout=240s
    echo "--- new query after the undo:"; ask
    echo "--- its trace (must show the PREVIOUS collection and embedding model):"
    kubectl -n "$NS" logs -l "$API_SELECTOR" --tail=300 --max-log-requests=20 --prefix=false \
      | $PY scripts/trace_tool.py "$(cat /tmp/last_trace_id)" --source - ;;
  history) kubectl -n "$NS" rollout history deploy/$LIVE ;;
  candidate-down)
    kubectl -n "$NS" delete deploy/api-candidate svc/api-candidate job/eval-candidate job/indexer-candidate --ignore-not-found ;;
  *) sed -n '2,24p' "$0"; exit 1 ;;
esac
