#!/usr/bin/env bash
# F5 step 1: sample `kubectl top pod` while something is running (minikube addons enable metrics-server; the
# README's `make cluster` already does).   scripts/measure_resources.sh <regex> <seconds> <out.csv>
#   scripts/measure_resources.sh '^api-|^chroma-|^redis-' 120 eval/results/resources_serving.csv   # during load_test
#   scripts/measure_resources.sh 'indexer'               300 eval/results/resources_indexer.csv    # during the Job
#   scripts/measure_resources.sh 'rag-eval'              300 eval/results/resources_eval.csv
set -euo pipefail
NS=${NS:-rahbar}; PAT=${1:?regex}; SECS=${2:-120}; OUT=${3:-eval/results/resources.csv}
mkdir -p "$(dirname "$OUT")"; echo "ts,pod,cpu_m,mem_mi" > "$OUT"
end=$((SECONDS+SECS))
while [ $SECONDS -lt $end ]; do
  kubectl top pod -n "$NS" --no-headers 2>/dev/null | grep -E "$PAT" | \
    awk -v ts="$(date -u +%FT%TZ)" '{gsub(/m/,"",$2); gsub(/Mi/,"",$3); print ts","$1","$2","$3}' >> "$OUT" || true
  sleep 5
done
echo "wrote $OUT"; python3 "$(dirname "$0")/summarize_resources.py" "$OUT"
