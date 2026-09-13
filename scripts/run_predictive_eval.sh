#!/usr/bin/env bash
set -euo pipefail

# run_predictive_eval.sh - Single-GPU foreground launcher for FabriVLA / MOSS predictive memory closed-loop evaluation.
# Usage: bash scripts/run_predictive_eval.sh <output_dir> [additional_flags...]

if [ $# -lt 1 ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    echo "Usage: $0 <output_dir> [flags...]"
    echo ""
    echo "Arguments:"
    echo "  output_dir                   Target directory for evaluation artifacts and reports (required)"
    echo ""
    echo "Example:"
    echo "  $0 /tmp/eval_run --checkpoint /path/to/checkpoint.pt --tasks all"
    exit 0
fi

OUTPUT_DIR="$1"
shift

if [ -z "${OUTPUT_DIR}" ]; then
    echo "Error: <output_dir> cannot be empty." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON="${PYTHON:-/root/fabrivla_env/bin/python}"

FA2_PKG_PATH="/root/FabriVLA/diagnostics/fa2_repro_20260908/python_packages"
if [ -n "${PYTHONPATH:-}" ]; then
    export PYTHONPATH="${FA2_PKG_PATH}:${REPO_ROOT}:${PYTHONPATH}"
else
    export PYTHONPATH="${FA2_PKG_PATH}:${REPO_ROOT}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Offline environment flags
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export HF_DATASETS_OFFLINE="1"
export WANDB_DISABLED="true"

# Thread limits
export OMP_NUM_THREADS="2"
export MKL_NUM_THREADS="2"

# Headless OpenGL / MuJoCo defaults
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"

# Preserve existing LD_PRELOAD and append libstdc++.so.6 if present
if [ -f "/usr/lib/x86_64-linux-gnu/libstdc++.so.6" ]; then
    if [ -n "${LD_PRELOAD:-}" ]; then
        export LD_PRELOAD="${LD_PRELOAD}:/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
    else
        export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
    fi
fi

exec "${PYTHON}" -u -m fabri_moss.evaluate_predictive \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
