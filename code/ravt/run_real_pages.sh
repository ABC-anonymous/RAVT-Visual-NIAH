#!/usr/bin/env bash
# =============================================================================
# 公开基准单项测试：ViDoRe / DocVQA-BEIR（500 页 / 451 查询）
#
#   bash run_real_pages.sh --check                      # 体检：数据、权重、python
#   bash run_real_pages.sh                              # 全量（RAVT + BM25/BGE + ColPali/ColQwen2）
#   bash run_real_pages.sh --models RAVT,BM25,BGE       # 跳过慢的 VLM
#   bash run_real_pages.sh --data /path/to/vidore_docvqa --num-workers 8
#
# 依赖：项目根有 checkpoints/ 与 data/vidore_docvqa/
# 结果：results/vidore_docvqa/real_doc_results.json
# 汇总：python report_real_pages.py --json results/vidore_docvqa/real_doc_results.json
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="${PYTHON:-python}"
DATA="$ROOT/data/vidore_docvqa"
WORK="$ROOT/results/vidore_docvqa"
MODELS="RAVT,BM25,BGE,ColPali,ColQwen2"
SEEDS="0 1 2 3 4"
DEGREES="0"
CHECK=0
NW=8

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check)        CHECK=1; shift ;;
    --data)         DATA="$2"; shift 2 ;;
    --work)         WORK="$2"; shift 2 ;;
    --models)       MODELS="$2"; shift 2 ;;
    --python)       PY="$2"; shift 2 ;;
    --num-workers)  NW="$2"; shift 2 ;;
    --seeds)        SEEDS="$2"; shift 2 ;;
    --degrade)      DEGREES="$2"; shift 2 ;;
    -h|--help)      sed -n '2,12p' "$0"; exit 0 ;;
    *)              echo "未知参数: $1"; exit 2 ;;
  esac
done

export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
export RAVT_BLOCK_HEIGHT=42
export RAVT_OCR_WORKERS="$NW"

echo "RAVT 公开基准测试（ViDoRe / DocVQA-BEIR）"
echo "  脚本    : $HERE"
echo "  数据    : $DATA"
echo "  输出    : $WORK"
echo "  模型    : $MODELS | OCR 线程: $NW"

PYOK=1
"$PY" -c "import sys, torch; print('  python  :', sys.executable); print('  [ok] torch', torch.__version__)" || PYOK=0
if [[ $PYOK == 0 ]]; then
  echo ""
  echo "❌ 这个 python 里没有 torch。用能跑 torch 的那个："
  echo "   bash run_real_pages.sh --python python"
  which -a python python3 2>/dev/null | sed 's/^/      /' || true
  exit 1
fi
RAVT_EVAL_DIR="$ROOT/code/ravt" "$PY" -c "import os,sys; sys.path.insert(0, os.environ['RAVT_EVAL_DIR']); from scorers import ocr_ready; ok,why=ocr_ready('tesseract'); print('  [ok] tesseract 可用' if ok else '  [WARN] tesseract 不可用: '+str(why))" \
  || echo "  [WARN] 无法确认 tesseract（code/ravt 不在同级？）"

for f in "$DATA/pages.jsonl" "$DATA/queries.jsonl"; do
  [[ -f "$f" ]] || { echo "❌ 缺 $f —— 先把 vidore_docvqa_bundle.zip 解压到项目根"; exit 1; }
done
N_PAGES=$(ls "$DATA/pages" 2>/dev/null | wc -l | tr -d ' ')
N_QRY=$(wc -l < "$DATA/queries.jsonl" | tr -d ' ')
N_TILE=$(ls "$WORK/tiles" 2>/dev/null | wc -l | tr -d ' ')
echo "  页面    : $N_PAGES 张 | 查询: $N_QRY 条 | 已缓存 tile: $N_TILE"

EVAL="$ROOT/code/ravt/evaluate_real_pages.py"

CKPTS=()
for s in $SEEDS; do
  if [[ -f "$ROOT/checkpoints/full_p14_rh42_s$s.pth" ]]; then
    CKPTS+=("checkpoints/full_p14_rh42_s$s.pth")
  fi
  if [[ -f "$ROOT/checkpoints/deg_rh42_p14_s$s.pth" ]]; then
    CKPTS+=("checkpoints/deg_rh42_p14_s$s.pth")
  fi
done
echo "  RAVT 权重: ${#CKPTS[@]} 个 checkpoint（找于 $ROOT/checkpoints）"
for w in colpali_offline_weights/base_model colpali_offline_weights/adapter_model models_local/colqwen2-base models_local/colqwen2; do
  if [[ -e "$ROOT/$w" ]]; then echo "            OK  $w"; else echo "            缺  $w（对应 VLM 基线会跳过）"; fi
done
[[ -f "$EVAL" ]] || { echo "❌ 找不到 code/ravt/evaluate_real_pages.py"; exit 1; }

if [[ $CHECK == 1 ]]; then
  echo ""
  echo "✅ 体检结束（--check 不会真的跑）。去掉 --check 即开始测试。"
  exit 0
fi

mkdir -p "$ROOT/results/logs"
cd "$ROOT" || exit 1

IFS=',' read -r -a DEG_ARR <<< "$DEGREES"
echo "  退化档  : ${DEG_ARR[*]}（0=干净；其他值是页面退化强度）"
echo ""

for s in "${DEG_ARR[@]}"; do
  s_trim="$(echo "$s" | tr -d ' ')"
  if [[ "$s_trim" == "0" || "$s_trim" == "0.0" ]]; then
    w="$WORK"
    extra=()
    tag="clean"
  else
    tag="deg$(echo "$s_trim" | tr -d '.')"
    w="$WORK/$tag"
    extra=( --degrade "$s_trim" )
  fi
  LOG="$ROOT/results/logs/vidore_${tag}_$(date +%m%d_%H%M).log"
  echo "==================== severity $s_trim -> $tag ===================="
  echo "日志: $LOG"

  RUN=( "$PY" -u "$EVAL" --docs "$DATA" --work "$w" --stage all
        --models "$MODELS" --row-height 42 )
  if [[ ${#extra[@]} -gt 0 ]]; then RUN+=( "${extra[@]}" ); fi
  if [[ ${#CKPTS[@]} -gt 0 ]]; then RUN+=( --ckpt "${CKPTS[@]}" ); fi

  if ! "${RUN[@]}" 2>&1 | tee "$LOG"; then
    echo ""
    echo "════════ severity $s_trim 失败（日志尾部）════════"
    tail -25 "$LOG"
    exit 1
  fi

  if [[ -f "$w/real_doc_results.json" && -f "$ROOT/code/ravt/report_real_pages.py" ]]; then
    "$PY" "$ROOT/code/ravt/report_real_pages.py" --json "$w/real_doc_results.json" \
      --out "$ROOT/results/analysis/vidore_table_${tag}.md" || true
  fi
  echo ""
done

echo "全部完成。各档结果："
echo "  clean -> $WORK/real_doc_results.json"
for s in "${DEG_ARR[@]}"; do
  s_trim="$(echo "$s" | tr -d ' ')"
  [[ "$s_trim" == "0" || "$s_trim" == "0.0" ]] && continue
  echo "  $s_trim -> $WORK/deg$(echo "$s_trim" | tr -d '.')/real_doc_results.json"
done
