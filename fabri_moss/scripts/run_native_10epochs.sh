#!/usr/bin/env bash
set -euo pipefail

# run_native_10epochs.sh - Foreground 2-GPU launcher for FabriVLA / Moss native full-model fine-tuning.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

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
if [ ! -x "${PYTHON}" ]; then
    echo "Error: Python interpreter not executable at ${PYTHON}. Set PYTHON env var to override." >&2
    exit 1
fi

FA2_PKG_PATH="/root/FabriVLA/diagnostics/fa2_repro_20260908/python_packages"
if [ ! -d "${FA2_PKG_PATH}" ]; then
    echo "Error: Required FlashAttention-2 package directory not found at ${FA2_PKG_PATH}." >&2
    exit 1
fi

# Prepend FA2 package directory and repo root to PYTHONPATH
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
    -m fabri_moss.train_native \
    --output-dir "${OUTPUT_DIR}" \
    --epochs 10 \
    --history-frames 16 \
    --target-frames 8 \
    --global-batch-size 8 \
    --workers 2 \
    --seed 4042 \
    --lr-vision 1e-6 \
    --lr-projector 5e-6 \
    --lr-llm 2e-6 \
    --lr-head 1e-5 \
    --warmup-updates 100 \
    --save-every 250 \
    --eval-every 250 \
    "$@"
