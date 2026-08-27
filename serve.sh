#!/bin/bash
# Start the Rokae policy server from the package root.
#   ./serve.sh [checkpoint_dir] [port] [extra server args...]
# Defaults (unchanged since the 2026-08-26 package): config configs/rokae_tokenar_infer.yaml,
# checkpoint checkpoints/8800_vlm0.0148_ae0.0040 (token_ar), port 8000.
# block_ar planner: use ./serve_blockar.sh (same script with block_ar defaults), or set CONFIG=... yourself.
# The config and the checkpoint must be the same architecture; the server refuses a mismatch at start-up.
# Requires: the venv from SETUP.md activated (or PYTHON=/path/to/python), one free GPU (~8 GB).
set -euo pipefail
cd "$(dirname "$0")"
CKPT=${1:-checkpoints/8800_vlm0.0148_ae0.0040}; PORT=${2:-8000}
CONFIG=${CONFIG:-configs/rokae_tokenar_infer.yaml}
PYTHON=${PYTHON:-python}
export PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-$PWD/.openpi_cache}
mkdir -p "$OPENPI_DATA_HOME/big_vision"; cp -n assets/big_vision/paligemma_tokenizer.model "$OPENPI_DATA_HOME/big_vision/" 2>/dev/null || true
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
# Keep the shipped package clean: without this, importing the code writes __pycache__/
# directories all through src/, which then travel along if the package is re-tarred.
export PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE:-1}
exec "$PYTHON" -m openpi.waypoint.rokae_policy serve --config "$CONFIG" --checkpoint "$CKPT" --port "$PORT" "${@:3}"
