#!/bin/bash
#SBATCH --job-name=04-EF-OEQ
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/experiment-%j.out
#SBATCH --error=logs/experiment-%j.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch from the project root}"

run_name=04-EF-Pretrain-TensorBackbone-OpenEquivariance
config="configs/tensor/$run_name.yaml"
for file in train.sbatch "$config" nmr_helpers/openequivariance_tp.py nmr_helpers/tensor_ef_pretraining.py data/train.xyz data/valid.xyz; do
    [[ -r "$file" ]] || { echo "Missing required file: $file" >&2; exit 1; }
done
grep -Fq 'TENSOR_NMR_ARC_PYTHON' train.sbatch || {
    echo 'Copy the updated train.sbatch before submitting Experiment 04.' >&2
    exit 1
}

if [[ "${TENSOR_NMR_VERIFY_ONLY:-0}" == 1 ]]; then
    verify_checkpoint="${TENSOR_NMR_VERIFY_CHECKPOINT:-graph-pes-results/04-EF-Pretrain-TensorBackbone-OpenEquivariance-9265688/model.pt}"
    for file in scripts/smoke_openequivariance.py nmr_helpers/openequivariance_compatibility.py "$verify_checkpoint"; do
        [[ -r "$file" ]] || { echo "Missing verification input: $file" >&2; exit 1; }
    done
fi

module purge
module load Anaconda3/2025.06-1
export TENSOR_NMR_ARC_ENV="${TENSOR_NMR_ARC_ENV:-/data/oums-atom-elec-ml/trin4011/envs/graph-pes-arc}"
conda activate "$TENSOR_NMR_ARC_ENV"
baseline_python="$TENSOR_NMR_ARC_ENV/bin/python"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1

cuda_version=$("$baseline_python" -c 'import torch; print(torch.version.cuda or "")')
[[ -n "$cuda_version" ]] || { echo 'Baseline PyTorch has no CUDA support.' >&2; exit 1; }
if [[ -z "${TENSOR_NMR_CUDA_MODULE:-}" ]] && ! command -v nvcc >/dev/null 2>&1; then
    cuda_modules=$(module -t avail CUDA 2>&1 || true)
    TENSOR_NMR_CUDA_MODULE=$(printf '%s\n' "$cuda_modules" |
        awk -v prefix="CUDA/$cuda_version" '
            $1 == prefix || index($1, prefix ".") == 1 || index($1, prefix "-") == 1 { print $1 }
        ' | sort -V | tail -n 1)
fi
if [[ -n "${TENSOR_NMR_CUDA_MODULE:-}" ]]; then
    echo "Loading $TENSOR_NMR_CUDA_MODULE for PyTorch CUDA $cuda_version"
    module load "$TENSOR_NMR_CUDA_MODULE"
    export TENSOR_NMR_CUDA_MODULE
fi
echo "Compute node: $(hostname), PyTorch CUDA: $cuda_version"
if command -v nvcc >/dev/null 2>&1; then
    nvcc --version
else
    echo 'No matching CUDA module found on PATH. Trying the existing CUDA libraries.'
fi

# Keep the overlay and extension build files on this job's node-local storage.
job_dir=$(mktemp -d "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/tensor-oeq-${SLURM_JOB_ID:?}.XXXXXX")
trap 'rm -rf -- "$job_dir"' EXIT
export TMPDIR="$job_dir" TORCH_EXTENSIONS_DIR="$job_dir/torch-extensions"
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"
overlay="$job_dir/venv"
echo 'Installing OpenEquivariance into the job-local Python overlay'
"$baseline_python" -m venv --system-site-packages "$overlay"
export TENSOR_NMR_ARC_PYTHON="$overlay/bin/python"
export PATH="$overlay/bin:$PATH"
"$baseline_python" -m pip list --format=freeze > "$job_dir/baseline-constraints.txt"
timeout --kill-after=30s 20m "$TENSOR_NMR_ARC_PYTHON" -m pip --require-virtualenv install \
    --disable-pip-version-check --retries 2 --timeout 30 \
    -c "$job_dir/baseline-constraints.txt" 'openequivariance==0.3.0'

# Catch import/extension failures before graph-pes prepares the training data.
echo 'Importing OpenEquivariance and loading its CUDA extension'
"$TENSOR_NMR_ARC_PYTHON" - <<'PY'
from importlib.metadata import version
import sys
import torch
import openequivariance

print("Python:", sys.executable)
print("PyTorch:", torch.__version__, torch.__file__)
print("OpenEquivariance:", version("openequivariance"))
PY

if [[ "${TENSOR_NMR_VERIFY_ONLY:-0}" == 1 ]]; then
    echo 'Checking new-model and saved-checkpoint CPU/CUDA compatibility; no training run is started'
    "$TENSOR_NMR_ARC_PYTHON" scripts/smoke_openequivariance.py --device cuda
    "$TENSOR_NMR_ARC_PYTHON" scripts/smoke_openequivariance.py --device cuda --checkpoint \
        "${TENSOR_NMR_VERIFY_CHECKPOINT:-graph-pes-results/04-EF-Pretrain-TensorBackbone-OpenEquivariance-9265688/model.pt}"
    exit 0
fi

export TENSOR_NMR_CONFIG_BATCHING=1
echo 'Starting Experiment 04 with batch size 2 and accumulation 2'
bash train.sbatch "$config" "$run_name"
