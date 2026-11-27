#!/usr/bin/env bash
# F3 / C3 controlled release for RAHBAR. One step per invocation so each can be screenshotted.
#   scripts/f3_release_demo.sh <step>
#   steps: candidate-up | eval-candidate | release | bad-release | undo | history
#
# You set what "the change" is, as explicit env overrides (they beat the ConfigMap values):
#   CANDIDATE_ENV="TOP_K=8 CHROMA_COLLECTION=rahbar-v2"    # the change from Part C1
#   BAD_ENV="CHROMA_COLLECTION=rahbar-bad RELEVANCE_FLOOR=0.0"  # a release that should never have shipped
# Because they land in the Deployment's pod template, `kubectl rollout undo` restores the old values too.
#
# CONTRACT WITH M2 (k8s/eval-job.yaml): backoffLimit: 0, restartPolicy: Never, target read from ${TARGET_URL},
# non-zero exit when the gate fails, and a NetworkPolicy that lets the eval Job pod reach the api pods.
set -uo pipefail
NS=${NS:-rahbar}
LIVE=${LIVE:-api}
API_PORT=${API_PORT:-8000}                      # container port of the api (STARTER.md: 8000)
API_SELECTOR=${API_SELECTOR:-app=api}           # check: kubectl get pods -n rahbar --show-labels
OVERLAY=${OVERLAY:-k8s/overlays/dev}
CANDIDATE_ENV=${CANDIDATE_ENV:?set CANDIDATE_ENV, e.g. \"TOP_K=8\"}
BAD_ENV=${BAD_ENV:-"RELEVANCE_FLOOR=0.0 TOP_K=1"}   # passes /readyz, serves traffic, answers badly. Pin-mismatch variant: BAD_ENV="EMBED_MODEL=mxbai-embed-large" -> /readyz fails, rollout stalls, old pods keep serving
QUESTION=${QUESTION:-"What CGPA puts a student on academic warning?"}
CAND_SVC_PORT=${CAND_SVC_PORT:-8000}            # Service api-candidate copies the live Service (port 8000)

env_args() { for kv in $1; do printf -- '--env %s ' "$kv"; done; }

run_eval() {   # $1 = target URL
  kubectl -n $NS delete job rag-eval --ignore-not-found --wait=true
  TARGET_URL="$1" envsubst < k8s/eval-job.yaml | kubectl -n $NS apply -f -
  echo "waiting for the eval Job to succeed or fail..."
  while true; do
    s=$(kubectl -n $NS get job rag-eval -o jsonpath='{.status.succeeded}' 2>/dev/null)
    f=$(kubectl -n $NS get job rag-eval -o jsonpath='{.status.failed}' 2>/dev/null)
    [ "${s:-0}" -ge 1 ] && { echo "GATE: PASS (Job succeeded)"; return 0; }
    [ "${f:-0}" -ge 1 ] && { echo "GATE: FAIL (Job failed) -> do NOT roll this candidate out"; return 1; }
    sleep 5
  done
}

ask() {        # ask the live api with a known request id (= trace id), print it
  id="demo-$(date +%s)"; echo "$id" > /tmp/last_trace_id
  kubectl -n $NS port-forward deploy/$LIVE 18000:$API_PORT >/dev/null 2>&1 & pf=$!; sleep 3
  curl -s -X POST localhost:18000/api/ask -H 'Content-Type: application/json' -H "X-Request-ID: $id" \
       -d "{\"question\": \"$QUESTION\"}"; echo
  kill $pf 2>/dev/null; echo "trace id: $id"
}

case "${1:-}" in
  candidate-up)
    mkdir -p k8s/candidate
    kubectl kustomize $OVERLAY | python scripts/make_candidate.py --live $LIVE $(env_args "$CANDIDATE_ENV") \
        > k8s/candidate/api-candidate.yaml || exit 1
    kubectl apply -f k8s/candidate/api-candidate.yaml
    kubectl -n $NS rollout status deploy/api-candidate --timeout=180s
    kubectl -n $NS get deploy,svc,pods -l track=candidate ;;
  eval-candidate)
    run_eval "http://api-candidate:$CAND_SVC_PORT"; rc=$?
    kubectl -n $NS logs job/rag-eval --tail=40; exit $rc ;;
  release)       # only after eval-candidate printed GATE: PASS
    kubectl -n $NS annotate deploy/$LIVE kubernetes.io/change-cause="release: $CANDIDATE_ENV" --overwrite
    kubectl -n $NS set env deploy/$LIVE $CANDIDATE_ENV
    kubectl -n $NS rollout status deploy/$LIVE --timeout=240s
    kubectl -n $NS rollout history deploy/$LIVE ;;
  bad-release)
    kubectl -n $NS annotate deploy/$LIVE kubernetes.io/change-cause="BAD release: $BAD_ENV" --overwrite
    kubectl -n $NS set env deploy/$LIVE $BAD_ENV
    kubectl -n $NS rollout status deploy/$LIVE --timeout=90s \
      || echo ">> rollout not finishing: old pods keep serving (maxUnavailable: 0, /readyz fails)"
    echo "--- query served by the bad release:"; ask ;;
  undo)
    kubectl -n $NS rollout undo deploy/$LIVE
    kubectl -n $NS rollout status deploy/$LIVE --timeout=240s
    echo "--- new query after the undo:"; ask
    echo "--- its trace (must show the PREVIOUS collection and embedding model):"
    kubectl -n $NS logs -l $API_SELECTOR --tail=300 --max-log-requests=20 --prefix=false \
      | python scripts/trace_tool.py "$(cat /tmp/last_trace_id)" --source - ;;
  history) kubectl -n $NS rollout history deploy/$LIVE ;;
  *) sed -n '2,9p' "$0"; exit 1 ;;
esac
