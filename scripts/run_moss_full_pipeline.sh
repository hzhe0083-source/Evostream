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
PYTHON="${PYTHON:-/root/fabrivla_env/bin/python}"
BRIDGE_DIR="$ROOT/bridge"
JOINT_DIR="$ROOT/joint"

BRIDGE_EPOCHS="${MOSS_BRIDGE_EPOCHS:-1}"
# Epochs are absolute targets: bridge 1, joint 2 (resume joint with
# --epochs 2 to finish the same two-epoch budget).
JOINT_EPOCHS="${MOSS_JOINT_EPOCHS:-2}"
GLOBAL_BATCH_SIZE="${MOSS_GLOBAL_BATCH_SIZE:-32}"
DECISION_STRIDE="${MOSS_DECISION_STRIDE:-5}"
EXECUTION_HORIZON="${MOSS_EXECUTION_HORIZON:-5}"
NATIVE_KD_WEIGHT="${MOSS_NATIVE_KD_WEIGHT:-1.0}"
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

# Do not let a Bridge that damages the native current-frame path feed Joint.
# The replay is fixed-observation diagnostics, not a success benchmark.
REPLAY_OUTPUT="$BRIDGE_DIR/fixed_replay.json"
if [[ "${MOSS_SKIP_REPLAY_GATE:-0}" != "1" ]]; then
    PYTHONPATH="/root/FabriVLA/diagnostics/fa2_repro_20260908/python_packages:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    CUDA_VISIBLE_DEVICES="$GPU_IDS" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$PYTHON" -u "$SCRIPT_DIR/paired_moss_replay.py" \
        --checkpoint "${CHECKPOINT:-/root/models/FabriVLA/checkpoint_step_93000.pt}" \
        --adapter "$BRIDGE_INIT" --adapter-stage bridge \
        --data-root "${DATA_ROOT:-/root/evo1_metaworld_dataset}" \
        --output "$REPLAY_OUTPUT" --fabri-root "${FABRI_ROOT:-/root/FabriVLA}" \
        --vlm "${VLM:-/root/models/InternVL3_5-1B}" --device cuda:0 \
        --split val --split-seed 4042 --val-fraction 0.1 \
        --rows 0 10 20 30 40 50 --window 16 --seed 4048 \
        --seed-policy per-plan --flow-steps 50 \
        --tasks "Lock the door by rotating the lock clockwise" \
            "Pull a handle up sideways" "Pull a lever down 90 degrees" "Open a drawer"
    "$PYTHON" - "$REPLAY_OUTPUT" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
payload = json.loads(path.read_text())
summary = payload.get("summary", {})
original = float(summary["original"]["mean_command_mae"])
window1 = float(summary["window1"]["mean_command_mae"])
window16 = float(summary["window16"]["mean_command_mae"])
if abs(window1 - original) > 1e-5:
    raise SystemExit(f"replay gate failed: window1/original MAE mismatch ({window1} vs {original})")
if window16 > window1 + 1e-8:
    raise SystemExit(f"replay gate failed: window16 MAE {window16} > window1 {window1}")
print(f"replay gate passed: original={original:.8f} window1={window1:.8f} window16={window16:.8f}")
PY
else
    echo "WARNING: MOSS_SKIP_REPLAY_GATE=1; Joint gate is bypassed." >&2
fi

JOINT_RESUME_ARGS=()
if [[ -f "$JOINT_DIR/last.pt" && "${MOSS_FORCE_FRESH:-0}" != "1" ]]; then
    JOINT_RESUME_ARGS=(--resume "$JOINT_DIR/last.pt")
else
    JOINT_RESUME_ARGS=(--init-adapter "$BRIDGE_INIT")
fi

CUDA_VISIBLE_DEVICES="$GPU_IDS" \
MOSS_STAGE=joint MOSS_CONTEXT_MODE=causal MOSS_NATIVE_KD_WEIGHT="$NATIVE_KD_WEIGHT" \
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
