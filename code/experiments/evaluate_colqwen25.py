"""阶段 C：ColQwen2.5（ColPali 家族 2025 年的强形态）在主网格上的评测。

独立脚本，不改评测框架：复用 `scorers.ColQwen2Scorer` 的图像编码与 MaxSim 打分，只把
模型/processor 换成 `ColQwen2_5` / `ColQwen2_5_Processor`（需要 colpali-engine ≥ 0.3.18）。

为了不动服务器上已有的 colpali-engine，把新版本装到独立目录再用 PYTHONPATH 前置：

    base=.; PY=python
    "$PY" -m pip install --no-index --no-deps --target "$base/third_party/colpali_engine_0318" \
        weights_upload/wheels/colpali_engine-0.3.18-py3-none-any.whl
    cd "$base" && PYTHONPATH="$base/third_party/colpali_engine_0318:$PYTHONPATH" \
      "$PY" code/experiments/evaluate_colqwen25.py --root "$base" \
        --base-dir models_local/colqwen2.5-base --adapter-dir models_local/colqwen2.5 \
        --bench data/test_bench_tmm_rh42 --out results

冒烟：加 --max-tasks 2 --levels 0.4
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def make_scorer(root: Path, args):
    """构造一个只换 backbone 的 ColQwen2.5 scorer。"""
    from scorers import ColQwen2Scorer

    class ColQwen25Scorer(ColQwen2Scorer):
        def prepare(self):
            import torch
            from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
            from peft import PeftModel

            self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.bfloat16 if self.dtype_name == "bfloat16" else torch.float32
            last_err = None
            self._model = None
            for kw in ({"attn_implementation": self.attn, "use_cache": False},
                       {"attn_implementation": self.attn},
                       {"attn_implementation": "eager"},
                       {}):
                for dtype_key in ("dtype", "torch_dtype"):
                    try:
                        self._model = ColQwen2_5.from_pretrained(
                            self.model_id, device_map=self.device,
                            local_files_only=self.local_files_only,
                            **{dtype_key: dtype}, **kw).eval()
                        break
                    except (TypeError, ValueError) as exc:
                        last_err = exc
                if self._model is not None:
                    print(f"  ColQwen2.5 加载成功（{kw or '默认参数'}）")
                    break
            if self._model is None:
                raise RuntimeError(f"ColQwen2.5 加载失败: {type(last_err).__name__}: {last_err}")
            if self.adapter_dir:
                self._model = PeftModel.from_pretrained(
                    self._model, self.adapter_dir,
                    local_files_only=self.local_files_only).eval()
                print(f"  适配器已挂载：{self.adapter_dir}")
            self._processor = ColQwen2_5_Processor.from_pretrained(
                self.processor_dir, local_files_only=self.local_files_only)

    return ColQwen25Scorer("ColQwen2.5", model_id=str(root / args.base_dir),
                           adapter_dir=str(root / args.adapter_dir),
                           processor_dir=str(root / args.base_dir),
                           batch_size=args.batch, dtype=args.dtype, attn=args.attn)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--base-dir", default="models_local/colqwen2.5-base")
    ap.add_argument("--adapter-dir", default="models_local/colqwen2.5")
    ap.add_argument("--bench", default="data/test_bench_tmm_rh42")
    ap.add_argument("--levels", nargs="+", type=float, default=[0.0, 0.2, 0.4])
    ap.add_argument("--max-tasks", type=int, default=0, help="每个档位最多评多少 task（0=全部）")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    for p in (root, root / "code" / "ravt"):
        sys.path.insert(0, str(p))

    try:
        from colpali_engine.models import ColQwen2_5  # noqa: F401
    except Exception as e:                                       # noqa: BLE001
        print(f"❌ 当前 Python 找不到 ColQwen2_5（{type(e).__name__}: {str(e)[:150]}）\n"
              "   请先按文件头的命令，把 colpali-engine 0.3.18 装到独立目录并用 "
              "PYTHONPATH 前置，再运行本脚本。")
        return 1

    import numpy as np
    from datasets_eval import list_tasks

    out = root / a.out
    out.mkdir(parents=True, exist_ok=True)
    sc = make_scorer(root, a)
    sc.prepare()

    rows, records = [], []
    for lv in a.levels:
        tasks = list_tasks(root / a.bench, noise_levels=[lv],
                           max_tasks=(a.max_tasks or None))
        print(f"\n=== noise={lv}：{len(tasks)} 个 task ===")
        if not tasks:
            continue
        hit = n = tied = 0
        lats, kbs = [], []
        for t in tasks:
            t0 = time.perf_counter()
            docs = sc.encode_documents(t)
            scores, _ = sc.score_query(t.query, docs, task=t)
            dt = (time.perf_counter() - t0) * 1000.0
            scores = np.asarray(scores, dtype=np.float64)
            correct = 0
            if scores.size and float(np.nanstd(scores)) == 0.0:
                tied += 1                      # 全同分：保守记为未命中
            elif int(np.argmax(scores)) == int(t.gt_block):
                correct = 1
                hit += 1
            n += 1
            lats.append(dt)
            kbs.append(sc.index_bytes(docs) / 1024.0 / max(len(scores), 1))
            records.append({"record": "task_eval", "model": "ColQwen2.5@main",
                            "benchmark": "main", "task_id": t.task_id, "noise": lv,
                            "blocks": t.blocks, "target_block": t.gt_block,
                            "pred_block": int(np.argmax(scores)) if scores.size else -1,
                            "correct": correct, "score_all_tied": int(
                                bool(scores.size and float(np.nanstd(scores)) == 0.0)),
                            "latency_ms": round(dt, 1),
                            "bytes_per_doc": int(sc.index_bytes(docs) / max(len(scores), 1))})
        row = {"level": lv, "n": n, "top1": 100.0 * hit / max(n, 1), "ties": tied,
               "p50_ms": st.median(lats), "kb_per_doc": st.mean(kbs)}
        rows.append(row)
        print(f"   top-1 = {row['top1']:.1f}%  全同分 {tied}  P50 {row['p50_ms']:.0f} ms  "
              f"{row['kb_per_doc']:.1f} KB/块")

    lines = ["| 模型 | 噪声 0.0 | 噪声 0.2 | 噪声 0.4 | KB/块 | P50 ms |",
             "|---|---|---|---|---|---|"]
    cells = {r["level"]: r for r in rows}
    lines.append("| ColQwen2.5 | " + " | ".join(
        (f"{cells[lv]['top1']:.1f}%" if lv in cells else "-")
        for lv in (0.0, 0.2, 0.4)) + " | " +
        (f"{st.mean([r['kb_per_doc'] for r in rows]):.1f}" if rows else "-") + " | " +
        (f"{st.mean([r['p50_ms'] for r in rows]):.0f}" if rows else "-") + " |")
    body = ("# 阶段 C：ColQwen2.5 在主网格上的 top-1（块级命中率，口径与主表一致）\n\n"
            + "\n".join(lines) + "\n\n"
            "把这一行并入主结果表即可（与 ColPali / ColQwen2 / DSE 同一批 task、同一指标）。\n")
    (out / "colqwen25.md").write_text(body, encoding="utf-8")
    (out / "colqwen25.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    with (out / "ColQwen2.5-main.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("\n" + body)
    print(f"写出 {out}/colqwen25.md（A 段脚本的汇总表可读 ColQwen2.5-main.jsonl）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
