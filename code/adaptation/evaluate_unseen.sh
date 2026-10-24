#!/usr/bin/env bash
# 评 RAVT（5 seeds）与 char3+OCR 在未见腐蚀族上的表现。
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(cd "$SCRIPT_DIR/../.." && pwd)"; PY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --base) BASE="${2:?}"; shift 2 ;;
    --python) PY="${2:?}"; shift 2 ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done
[ -n "$PY" ] || PY="$(command -v python)"
cd "$BASE" || exit 1
export PYTHONUNBUFFERED=1

TMP="results/unseen240_ckpt"
rm -rf "$TMP"; mkdir -p "$TMP/rh42"
for s in 0 1 2 3 4; do
  ln -s "$BASE/checkpoints/full_p14_rh42_s$s.pth" "$TMP/rh42/"
done

status=0
run() { local name="$1"; shift
  echo; echo "==================== $name ===================="
  if "$@"; then
    echo "[$name 完成]"
  else
    echo "[$name 失败]"
    status=1
  fi
}

run "RAVT on unseen families" "$PY" -u code/ravt/evaluate.py \
  --models RAVT --ckpt-dir "$TMP/rh42" --benches main \
  --bench-main data/test_bench_unseen240 --row-height 42 \
  --out-dir results/jsonl/eval_unseen240/ravt --on-missing error

run "char3+OCR on unseen families" "$PY" -u code/ravt/evaluate.py \
  --models OCR --only-names BM25-char3 --benches main \
  --bench-main data/test_bench_unseen240 --row-height 42 \
  --ocr-backend tesseract \
  --out-dir results/jsonl/eval_unseen240/ocr --on-missing error

exit "$status"
