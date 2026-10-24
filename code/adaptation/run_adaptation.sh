#!/usr/bin/env bash
# 只跑支撑论文结论所需的补充实验。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --base) BASE="${2:?}"; shift 2 ;;
    --python) PY="${2:?}"; shift 2 ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done
[ -n "$PY" ] || PY="$(command -v python)"
cd "$BASE"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p results/logs

if [ ! -d data/test_bench_unseen240 ]; then
  "$PY" code/adaptation/build_unseen_benchmark.py --root "$BASE"
fi
n=$(find data/test_bench_unseen240 -mindepth 1 -maxdepth 1 -type d -name 'Task_*' | wc -l)
if [ "$n" -ne 480 ]; then
  echo "unseen240 task 数异常：$n（应为 480），重建"
  "$PY" code/adaptation/build_unseen_benchmark.py --root "$BASE"
fi

echo "==================== unseen240 eval ===================="
bash code/adaptation/evaluate_unseen.sh --base "$BASE" --python "$PY"
"$PY" code/adaptation/summarize_unseen.py --root "$BASE"

echo "==================== grouped proxy / routing ===================="
"$PY" code/adaptation/calibrate_cer_proxy.py --root "$BASE" --skip-real
"$PY" code/adaptation/analyze_routing.py --root "$BASE"

echo "==================== page models: seeds 1/2 at 40% ===================="
bash code/adaptation/run_lora_adaptation.sh --base "$BASE" --python "$PY"

echo "==================== adaptation reproduction complete ===================="
