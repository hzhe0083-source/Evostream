#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: $0 OUTPUT_ROOT [extra trainer flags...]"
    exit 0
fi

ROOT="$1"
shift
[[ -n "$ROOT" ]] || { echo 'OUTPUT_ROOT cannot be empty.' >&2; exit 1; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="$SCRIPT_DIR/run_moss_cross_recovery.sh"
BRIDGE_DIR="$ROOT/bridge"
JOINT_DIR="$ROOT/joint"

BRIDGE_EPOCHS="${MOSS_BRIDGE_EPOCHS:-1}"
# Epochs are absolute targets: bridge 1, joint 2 (resume joint with
# --epochs 2 to finish the same two-epoch budget).
JOINT_EPOCHS="${MOSS_JOINT_EPOCHS:-2}"
GLOBAL_BATCH_SIZE="${MOSS_GLOBAL_BATCH_SIZE:-32}"
DECISION_STRIDE="${MOSS_DECISION_STRIDE:-5}"
EXECUTION_HORIZON="${MOSS_EXECUTION_HORIZON:-5}"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1}"
BRIDGE_RESUME_ARGS=()
if [[ -f "$BRIDGE_DIR/last.pt" && "${MOSS_FORCE_FRESH:-0}" != "1" ]]; then
    BRIDGE_RESUME_ARGS=(--resume "$BRIDGE_DIR/last.pt")
fi

CUDA_VISIBLE_DEVICES="$GPU_IDS" \
MOSS_STAGE=bridge MOSS_CONTEXT_MODE=causal \
"$LAUNCHER" "$BRIDGE_DIR" \
    --stage bridge \
    --context-mode causal \
    --decision-stride "$DECISION_STRIDE" \
    --execution-horizon "$EXECUTION_HORIZON" \
    --epochs "$BRIDGE_EPOCHS" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    "${BRIDGE_RESUME_ARGS[@]}" \
    "$@"

BRIDGE_INIT="$BRIDGE_DIR/epoch_001.pt"
if [[ ! -f "$BRIDGE_INIT" ]]; then
    BRIDGE_INIT="$BRIDGE_DIR/last.pt"
fi
[[ -f "$BRIDGE_INIT" ]] || { echo "Bridge checkpoint not found: $BRIDGE_INIT" >&2; exit 1; }

JOINT_RESUME_ARGS=()
if [[ -f "$JOINT_DIR/last.pt" && "${MOSS_FORCE_FRESH:-0}" != "1" ]]; then
    JOINT_RESUME_ARGS=(--resume "$JOINT_DIR/last.pt")
else
    JOINT_RESUME_ARGS=(--init-adapter "$BRIDGE_INIT")
fi

CUDA_VISIBLE_DEVICES="$GPU_IDS" \
MOSS_STAGE=joint MOSS_CONTEXT_MODE=causal \
"$LAUNCHER" "$JOINT_DIR" \
    --stage joint \
    --context-mode causal \
    --decision-stride "$DECISION_STRIDE" \
    --execution-horizon "$EXECUTION_HORIZON" \
    --lora-rank 8 --lora-alpha 16 --lora-dropout 0.1 \
    --epochs "$JOINT_EPOCHS" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    "${JOINT_RESUME_ARGS[@]}" \
    "$@"
