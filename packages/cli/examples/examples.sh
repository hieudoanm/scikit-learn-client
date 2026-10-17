#!/usr/bin/env bash
#
# Smoke test for the skl standalone binary.
#
# Trains a linear regression on examples/csv/demo.csv (x -> y = 2x) and
# exports the model to examples/models/demo.omnx, entirely offline.
#
# Usage:
#   ./examples/examples.sh               # uses ./dist/skl/skl
#   ./examples/examples.sh /path/to/skl  # custom binary path
set -euo pipefail

cd "$(dirname "$0")/.."

BINARY="${1:-./dist/skl}"
CSV="examples/csv/demo.csv"
OUT="examples/models/demo.omnx"

echo "==> binary : $BINARY"
echo "==> input  : $CSV"
echo "==> output : $OUT"

[ -x "$BINARY" ] || {
  echo "error: executable binary not found at $BINARY (run 'make build' first)" >&2
  exit 1
}
[ -f "$CSV" ] || {
  echo "error: input CSV not found: $CSV" >&2
  exit 1
}

mkdir -p examples/models
rm -f "$OUT"

"$BINARY" run "$CSV" \
  --features x \
  --label y \
  --task-type regression \
  --algorithm linear_regression \
  --format onnx \
  --output "$OUT" \
  --no-interactive

[ -s "$OUT" ] || {
  echo "error: expected output missing or empty: $OUT" >&2
  exit 1
}

echo "==> trained model exported: $OUT ($(wc -c < "$OUT") bytes)"

if command -v python3 >/dev/null 2>&1; then
  python3 - "$OUT" <<'PY' || true
import sys
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession(sys.argv[1])
feed = {i.name: np.array([[42.0]], dtype=np.float32) for i in sess.get_inputs()}
pred = float(sess.run(None, feed)[0].ravel()[0])
print(f"==> inference x=42 -> {pred:.4f} (expect ~84.0)")
PY
fi
