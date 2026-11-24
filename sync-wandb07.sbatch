#!/bin/bash
# Upload offline W&B runs from a compute node. Submit from the project root:
#   sbatch sync-wandb07.sbatch                 # all Experiment 07 offline runs
#   sbatch sync-wandb07.sbatch 'graph-pes-results/wandb/offline-run-*-07-SymmetricTensor-head-N10-*'
# Login nodes cap each user process at 1 h CPU time and run a different OS
# image, so long syncs there can die. Compute nodes reach api.wandb.ai
# (Experiment 05 logged live), and credentials come from ~/.netrc.
#SBATCH --job-name=07-wandb-sync
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=02:00:00
#SBATCH --output=logs/07-wandb-sync-%j.out

set -o pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from Tensor_NMR_Transfer}"
module purge
module load Anaconda3/2025.06-1
conda activate "${TENSOR_NMR_ARC_ENV:-/data/oums-atom-elec-ml/trin4011/envs/graph-pes-arc}"
export WANDB_MODE=online
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/wandb-sync-${SLURM_JOB_ID}"

PATTERN="${1:-graph-pes-results/wandb/offline-run-*-07-SymmetricTensor-*}"
shopt -s nullglob
RUNS=( $PATTERN )
[[ ${#RUNS[@]} -gt 0 ]] || { echo "No offline runs match: $PATTERN"; exit 1; }
echo "Syncing ${#RUNS[@]} offline runs with $(command -v wandb)"

FAILED=0
# One run per process, so one failure cannot take the others down.
for RUN in "${RUNS[@]}"; do
    echo "== $RUN"
    if wandb sync --include-synced "$RUN"; then
        echo "OK $RUN"
    else
        echo "FAILED $RUN"
        FAILED=$((FAILED + 1))
    fi
done
echo "Finished: $(( ${#RUNS[@]} - FAILED )) of ${#RUNS[@]} synced"
[[ $FAILED -eq 0 ]]
