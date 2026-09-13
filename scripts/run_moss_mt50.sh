#!/usr/bin/env bash
set -euo pipefail

# Official synchronous MOSS cross-attention MT50 runner.
# Usage: bash scripts/run_moss_mt50.sh OUTPUT_DIR --checkpoint ... --adapter ...
if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: $0 OUTPUT_DIR [evaluate_moss.py flags...]"
    exit 0
fi

OUTPUT_DIR="$1"
shift
[[ -n "$OUTPUT_DIR" ]] || { echo 'OUTPUT_DIR cannot be empty.' >&2; exit 1; }
for arg in "$@"; do
    case "$arg" in
        --output-dir|--output-dir=*)
            echo 'Use positional OUTPUT_DIR only.' >&2
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-/root/fabrivla_env/bin/python}"
FA2_PKG_PATH="${FA2_PKG_PATH:-/root/FabriVLA/diagnostics/fa2_repro_20260908/python_packages}"

if [[ -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="$FA2_PKG_PATH:$REPO_ROOT:$PYTHONPATH"
else
    export PYTHONPATH="$FA2_PKG_PATH:$REPO_ROOT"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_DISABLED=true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"

exec "$PYTHON" -u -m fabri_moss.evaluate_moss \
    --output-dir "$OUTPUT_DIR" \
    --episodes 10 --episode-horizon 400 --exec-horizon 5 \
    --num-inference-timesteps 50 --observation-stride 1 \
    "$@"
