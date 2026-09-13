#!/usr/bin/env bash
set -euo pipefail

# run_stream_10epochs.sh - Foreground 2-GPU launcher for stream replay v1 fine-tuning.
# Wraps run_native_10epochs.sh with --stream-protocol stream_replay_v1 and passes additional args.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <output_dir> [additional_args...]" >&2
    echo "Error: Missing required argument <output_dir>." >&2
    exit 1
fi

OUTPUT_DIR="$1"
shift

exec "${SCRIPT_DIR}/run_native_10epochs.sh" "${OUTPUT_DIR}" --stream-protocol stream_replay_v1 "$@"
