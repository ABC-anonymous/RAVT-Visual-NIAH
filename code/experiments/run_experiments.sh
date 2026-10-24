#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY=""
GROUP=""

while [ $# -gt 0 ]; do
  case "$1" in
    --base) BASE="${2:?}"; shift 2 ;;
    --python) PY="${2:?}"; shift 2 ;;
    robustness|ablations|baselines|all) GROUP="$1"; shift ;;
    *)
      echo "usage: $0 {robustness|ablations|baselines|all} [--base PATH] [--python PATH]" >&2
      exit 2
      ;;
  esac
done

[ -n "$GROUP" ] || {
  echo "usage: $0 {robustness|ablations|baselines|all} [--base PATH] [--python PATH]" >&2
  exit 2
}
[ -n "$PY" ] || PY="$(command -v python)"

cd "$BASE"
mkdir -p results/logs results/analysis

run() {
  local name="$1"
  shift
  echo "==================== $name ===================="
  "$@"
}

if [ "$GROUP" = "robustness" ] || [ "$GROUP" = "all" ]; then
  run "aggregation probe" \
    "$PY" code/experiments/aggregation_probe.py --root "$BASE"
  run "native-page probe" \
    "$PY" code/experiments/native_page_probe.py --root "$BASE"
  run "fixed versus detected boxes" \
    "$PY" code/experiments/fixed_vs_detected_boxes.py --root "$BASE"
  run "line detector probe" \
    "$PY" code/experiments/line_detector_probe.py --root "$BASE"
  run "curriculum endpoints" \
    "$PY" code/experiments/train_curriculum_endpoints.py --root "$BASE" --python "$PY"
fi

if [ "$GROUP" = "ablations" ] || [ "$GROUP" = "all" ]; then
  run "row geometry" \
    "$PY" code/experiments/row_geometry.py --root "$BASE"
  run "efficiency" \
    "$PY" code/experiments/efficiency.py --root "$BASE" --measure
  run "curriculum evaluation" \
    "$PY" code/experiments/evaluate_curriculum_endpoints.py --root "$BASE"
fi

if [ "$GROUP" = "baselines" ] || [ "$GROUP" = "all" ]; then
  run "native text baselines" \
    "$PY" code/experiments/native_text_baselines.py --root "$BASE"
  run "OCR backend comparison" \
    "$PY" code/experiments/ocr_backend_comparison.py --root "$BASE"
  run "ColQwen2.5 baseline" \
    "$PY" code/experiments/evaluate_colqwen25.py --root "$BASE"
fi

echo "==================== experiments complete ===================="
