#!/usr/bin/env bash
set -euo pipefail

# MOSS-style visual cross-attention recovery. This deliberately does not use
# CausalMemoryWriter: each frame is projected to per-layer visual K/V and read
# by the text/readout queries through MossInternVL.
if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: $0 OUTPUT_DIR [train.py flags...]"
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

GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=, read -r -a GPU_LIST <<< "$GPU_IDS"
if (( ${#GPU_LIST[@]} >= 1 )); then
    TRAIN_MODULE="fabri_moss.train_moss_ddp"
    DIST_ARGS=("-m" "torch.distributed.run" "--standalone" "--nproc_per_node=${#GPU_LIST[@]}" "-m")
fi

CADENCE_ARGS=()
if [[ -n "${MOSS_DECISION_STRIDE:-}" ]]; then
    CADENCE_ARGS+=(--decision-stride "$MOSS_DECISION_STRIDE")
fi

exec env \
    PYTHONPATH="/root/FabriVLA/diagnostics/fa2_repro_20260908/python_packages:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_DISABLED=true \
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}" \
    CUDA_VISIBLE_DEVICES="$GPU_IDS" \
    "$PYTHON" -u "${DIST_ARGS[@]}" "$TRAIN_MODULE" \
    --checkpoint "${CHECKPOINT:-/root/models/FabriVLA/checkpoint_step_93000.pt}" \
    --data-root "${DATA_ROOT:-/root/evo1_metaworld_dataset}" \
    --stage "${MOSS_STAGE:-bridge}" \
    --context-mode "${MOSS_CONTEXT_MODE:-causal}" \
    --window "${MOSS_WINDOW:-16}" \
    --frame-stride "${MOSS_FRAME_STRIDE:-1}" \
    --min-context-frames "${MOSS_MIN_CONTEXT_FRAMES:-1}" \
    --execution-horizon "${MOSS_EXECUTION_HORIZON:-5}" \
    "${CADENCE_ARGS[@]}" \
    "$@" \
    --output-dir "$OUTPUT_DIR"
