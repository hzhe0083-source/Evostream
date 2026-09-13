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
JOINT_EPOCHS="${MOSS_JOINT_EPOCHS:-1}"
GLOBAL_BATCH_SIZE="${MOSS_GLOBAL_BATCH_SIZE:-32}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
MOSS_STAGE=bridge MOSS_CONTEXT_MODE=causal \
"$LAUNCHER" "$BRIDGE_DIR" \
    --epochs "$BRIDGE_EPOCHS" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    "$@"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
MOSS_STAGE=joint MOSS_CONTEXT_MODE=causal \
"$LAUNCHER" "$JOINT_DIR" \
    --epochs "$JOINT_EPOCHS" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --init-adapter "$BRIDGE_DIR/adapter_final.pt" \
    "$@"
