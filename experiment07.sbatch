#!/bin/bash
#SBATCH --job-name=07-Symm
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/experiment07-%A_%a.out
#SBATCH --error=logs/experiment07-%A_%a.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from Tensor_NMR_Transfer}"
module purge
module load Anaconda3/2025.06-1
if [[ -n "${TENSOR_NMR_CUDA_MODULE:-}" ]]; then module load "$TENSOR_NMR_CUDA_MODULE"; fi
ARC_ENV="${TENSOR_NMR_ARC_ENV:-/data/oums-atom-elec-ml/trin4011/envs/graph-pes-arc}"
conda activate "$ARC_ENV"
PYTHON="${TENSOR_NMR_ARC_PYTHON:-$ARC_ENV/bin/python}"
[[ -x "$PYTHON" ]] || { echo "Missing ARC interpreter: $PYTHON" >&2; exit 1; }
export TENSOR_NMR_ARC_ENV="$ARC_ENV" TENSOR_NMR_ARC_PYTHON="$PYTHON"
export TENSOR_NMR_CONFIG_BATCHING=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
# HTC compute nodes reach W&B (Experiments 05/06 synced live); set WANDB_MODE=offline to opt out.
export WANDB_MODE="${WANDB_MODE:-online}"
# Keep transient caches isolated; all fit results and W&B files stay on /data.
TASK_CACHE="${TMPDIR:-/tmp}/tensor-nmr-07-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID:-single}"
mkdir -p "$TASK_CACHE"
export XDG_CACHE_HOME="$TASK_CACHE/xdg" MPLCONFIGDIR="$TASK_CACHE/matplotlib"
echo "Mode: ${1:-array}; W&B: $WANDB_MODE; interpreter: $PYTHON"

if [[ "${1:-}" == --smoke ]]; then
    srun "$PYTHON" scripts/experiment07.py smoke --device cuda \
      --output "provenance/experiment07/smoke-${SLURM_JOB_ID}"
elif [[ "${1:-}" == --resume ]]; then
    [[ $# -eq 2 || $# -eq 3 ]] || { echo 'Usage: sbatch experiment07.sbatch --resume RUN_DIR [MAX_EPOCHS]'; exit 2; }
    srun "$PYTHON" scripts/experiment07.py resume "$2" ${3:+--max-epochs "$3"}
    srun "$PYTHON" scripts/experiment07.py complete "$2"
elif [[ "${1:-}" == --evaluate ]]; then
    srun "$PYTHON" scripts/evaluate_experiment07.py --device cuda
else
    [[ $# -eq 1 ]] || { echo 'Array job requires a smoke-results.json receipt'; exit 2; }
    srun "$PYTHON" scripts/experiment07.py receipt "$1"
    TASK_INDEX="${SLURM_ARRAY_TASK_ID:?Expected Slurm array}"
    ROW=$(awk -F '\t' -v task="$TASK_INDEX" '$1==task {print; exit}' provenance/experiment07/pending.tsv)
    [[ -n "$ROW" ]] || { echo "No config for array task $TASK_INDEX"; exit 2; }
    IFS=$'\t' read -r INDEX CONFIG RUN_NAME <<< "$ROW"
    echo "Task $INDEX: $CONFIG -> $RUN_NAME"
    bash train.sbatch "$CONFIG" "$RUN_NAME"
    srun "$PYTHON" scripts/experiment07.py complete "graph-pes-results/${RUN_NAME}-${SLURM_JOB_ID}"
fi
