#!/bin/bash
# Start the block_ar Rokae policy server from the package root (same launcher as serve.sh, block_ar defaults).
#   ./serve_blockar.sh [checkpoint_dir] [port] [extra server args...]
# Defaults: config configs/rokae_blockar_infer.yaml, checkpoint checkpoints/blockar_8800_vlm0.0338_ae0.0045, port 8000.
# The config and the checkpoint must be the same architecture; the server refuses a mismatch at start-up.
# Optional: --terminal-stop-agree 2 (default 0 = the planner's end marker is only reported; see docs/17 §6).
set -euo pipefail
cd "$(dirname "$0")"
export CONFIG=${CONFIG:-configs/rokae_blockar_infer.yaml}
exec ./serve.sh "${1:-checkpoints/blockar_8800_vlm0.0338_ae0.0045}" "${2:-8000}" "${@:3}"
