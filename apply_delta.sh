#!/bin/bash
# Apply this block_ar delta onto an existing rokae_tokenar_deploy_20260826 package.
#   bash apply_delta.sh <path to the 2026-08-26 package root>
# What it does: (1) checks the target really is that package, (2) copies every file of this
# delta over it (new files are added, updated files replace the 2026-08-26 versions -- the
# token_ar checkpoint, config, base weights and expected numbers are untouched), (3) verifies
# the whole merged package against the SHA256SUMS shipped in this delta.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TARGET=${1:?usage: bash apply_delta.sh <path to rokae_tokenar_deploy_20260826>}
TARGET="$(cd "$TARGET" && pwd)"
for f in models/pi05_base/model.safetensors src/openpi/waypoint/rokae_policy.py SETUP.md configs/rokae_tokenar_infer.yaml; do
  [ -e "$TARGET/$f" ] || { echo "✗ $TARGET does not look like the 2026-08-26 package: missing $f"; exit 2; }
done
echo "target: $TARGET"
# copy everything (this script and DELTA_MANIFEST.txt included, so the merged package records what
# was applied; SHA256SUMS lists every merged file except itself and DELTA_MANIFEST.txt)
( cd "$HERE" && find . -type f -print0 ) | while IFS= read -r -d '' rel; do
  mkdir -p "$TARGET/$(dirname "$rel")"
  cp -f "$HERE/$rel" "$TARGET/$rel"
  echo "  + $rel"
done
chmod +x "$TARGET/serve.sh" "$TARGET/serve_blockar.sh" "$TARGET/scripts/"*.py 2>/dev/null || true
echo "verifying the merged package (SHA256SUMS covers every file, ~1 minute for the 6.8 GB base weights)..."
( cd "$TARGET" && sha256sum -c SHA256SUMS --quiet ) && echo "✓ merged package verified: every file matches SHA256SUMS" || { echo "✗ verification FAILED -- see the lines above; re-copy those files"; exit 1; }
cat <<EOF

Next (all from $TARGET, with the venv from SETUP.md activated):
  export PYTHONPATH=\$PWD/src:\$PWD/packages/openpi-client/src
  export OPENPI_DATA_HOME=\$PWD/.openpi_cache
  python scripts/check_env.py                 # expects ALL OK (now also checks checkpoint <-> config architecture)
  CUDA_VISIBLE_DEVICES=0 ./serve_blockar.sh   # block_ar server; ./serve.sh still starts the token_ar server
  # then the 对拍 of SETUP.md §4b (expected numbers: data/expected_val_ep2_blockar.json)
EOF
