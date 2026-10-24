#!/usr/bin/env bash
# 页面模型核心补实验：新增 seed 1/2，仅在 40% 腐蚀的 240 tasks 上评测。
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

run_logged() {
  local name="$1"
  shift
  local log="results/logs/${name}.log"
  echo "==================== $name ===================="
  echo "日志: $BASE/$log"
  "$@" >"$log" 2>&1
  echo "[$name 完成]"
}

adapter_ready() {
  [ -s "$1/adapter_config.json" ] && [ -s "$1/adapter_model.safetensors" ]
}

"$PY" code/adaptation/link_hard_subset.py --root "$BASE"

for seed in 1 2; do
  qout="checkpoints/colqwen2_lora_s${seed}"
  if ! adapter_ready "$qout"; then
    run_logged "core_train_colqwen2_s${seed}" \
      "$PY" code/adaptation/train_colqwen2_lora.py --root "$BASE" \
        --epochs 2 --batch 16 --accum 1 --max-steps 7000 --save-every 0 \
        --workers 4 --seed "$seed" --out "$qout"
  else
    echo "复用 ColQwen2 adapter：$qout"
  fi
  run_logged "core_eval_colqwen2_s${seed}" \
    "$PY" -u code/ravt/evaluate.py --models ColQwen2 \
      --ckpt-dir checkpoints --colqwen-dir "$qout" \
      --colqwen-base models_local/colqwen2-base \
      --benches main --bench-main data/test_bench_tmm_rh42_n40 \
      --row-height 42 --out-dir "results/jsonl/eval_lora_core/colqwen2_s${seed}" \
      --on-missing error

  pout="checkpoints/colpali_lora_s${seed}"
  if ! adapter_ready "$pout"; then
    run_logged "core_train_colpali_s${seed}" \
      "$PY" code/adaptation/train_colpali_lora.py --root "$BASE" \
        --epochs 2 --batch 2 --accum 4 --max-steps 7000 --save-every 0 \
        --workers 4 --seed "$seed" --out "$pout"
  else
    echo "复用 ColPali adapter：$pout"
  fi
  run_logged "core_eval_colpali_s${seed}" \
    "$PY" -u code/ravt/evaluate.py --models ColPali \
      --ckpt-dir checkpoints --colpali-base colpali_offline_weights/base_model \
      --colpali-adapter "$pout" \
      --benches main --bench-main data/test_bench_tmm_rh42_n40 \
      --row-height 42 --out-dir "results/jsonl/eval_lora_core/colpali_s${seed}" \
      --on-missing error
done

"$PY" code/adaptation/summarize_lora.py --root "$BASE"
echo "页面模型核心补实验完成"
