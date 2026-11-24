#!/bin/bash
# Submit only from the ARC project root. No training runs on the login node.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs provenance/experiment07
[[ -f provenance/experiment07/pending.tsv && -f experiment07.sbatch ]] || exit 1
[[ $(wc -l < provenance/experiment07/pending.tsv) -eq 22 ]] || { echo 'Expected 22 tasks'; exit 1; }
MAX_PARALLEL="${EXPERIMENT07_MAX_PARALLEL:-4}"
[[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]] || { echo 'EXPERIMENT07_MAX_PARALLEL must be positive'; exit 2; }
FLAGS=(--clusters=htc --partition=short --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64000M --gres=gpu:l40s:1 --export=ALL)
sbatch --test-only "${FLAGS[@]}" --time=01:00:00 --job-name=07-smoke experiment07.sbatch --smoke
sbatch --test-only "${FLAGS[@]}" --time=12:00:00 --array="0-21%$MAX_PARALLEL" experiment07.sbatch prospective-smoke-receipt.json
if [[ "${1:-}" == --dry-run ]]; then
    echo 'Scheduler checks passed; no jobs submitted.'
    exit 0
fi
[[ $# -eq 0 ]] || { echo 'Usage: bash submit-experiment07.sh [--dry-run]'; exit 2; }
STATE=provenance/experiment07/submission.txt
if ! (set -o noclobber; : > "$STATE") 2>/dev/null; then
    echo "Existing pipeline recorded in $STATE. Inspect it before submitting a duplicate." >&2
    exit 1
fi
# The receipt binds the full sweep to precisely the code/data tested on CUDA.
SMOKE=$(sbatch --parsable "${FLAGS[@]}" --time=01:00:00 --job-name=07-smoke experiment07.sbatch --smoke)
SMOKE=${SMOKE%%;*}
[[ "$SMOKE" =~ ^[0-9]+$ ]] || { echo 'Unexpected smoke job ID'; exit 1; }
echo "SMOKE=$SMOKE" >> "$STATE"
RECEIPT="provenance/experiment07/smoke-${SMOKE}/smoke-results.json"
ARRAY=$(sbatch --parsable "${FLAGS[@]}" --time=12:00:00 --array="0-21%$MAX_PARALLEL" \
  --dependency="afterok:$SMOKE" --kill-on-invalid-dep=yes --job-name=07-sweep experiment07.sbatch "$RECEIPT")
ARRAY=${ARRAY%%;*}
[[ "$ARRAY" =~ ^[0-9]+$ ]] || { echo 'Unexpected array job ID'; exit 1; }
echo "ARRAY=$ARRAY" >> "$STATE"
echo "GPU smoke: $SMOKE; 22-run array: $ARRAY; maximum simultaneous training jobs: $MAX_PARALLEL"
echo 'The array starts only after a successful GPU smoke test.'
echo "squeue --clusters=htc -j $SMOKE,$ARRAY"
