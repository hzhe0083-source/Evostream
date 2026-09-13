#!/usr/bin/env bash
set -euo pipefail

# run_joint_predictive.sh - Foreground 2-GPU launcher for stage 3 joint predictive fine-tuning.
# Usage: bash scripts/run_joint_predictive.sh <output_dir> [additional_args...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <output_dir> [additional_args...]" >&2
    echo "Error: Missing required argument <output_dir>." >&2
    exit 1
fi

OUTPUT_DIR="$1"
shift

if [ -z "${OUTPUT_DIR}" ]; then
    echo "Error: <output_dir> cannot be empty." >&2
    exit 1
fi

PYTHON="${PYTHON:-/root/fabrivla_env/bin/python}"

FA2_PKG_PATH="/root/FabriVLA/diagnostics/fa2_repro_20260908/python_packages"
if [ -n "${PYTHONPATH:-}" ]; then
    export PYTHONPATH="${FA2_PKG_PATH}:${REPO_ROOT}:${PYTHONPATH}"
else
    export PYTHONPATH="${FA2_PKG_PATH}:${REPO_ROOT}"
fi

export CUDA_VISIBLE_DEVICES="0,1"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export HF_DATASETS_OFFLINE="1"
export WANDB_DISABLED="true"
export OMP_NUM_THREADS="2"
export MKL_NUM_THREADS="2"

exec "${PYTHON}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node=2 \
    -m fabri_moss.train_joint_predictive \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
