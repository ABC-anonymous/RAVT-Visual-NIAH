"""阶段 B：OCR 后端对照 —— "识别是瓶颈"这个论点不能只挂在 tesseract 上。

同一批 task、同一套检索器，只换 OCR 后端：

    tesseract 2x（论文用的，缓存已存在，几乎免费）
    rapidocr  4x（服务器已装，是现代 PP-OCR 系列的 onnx 版，作为"更强的识别器"对照）

每个 (后端, 噪声档) 报告：块级 top-1 命中率与 OCR 的平均字符错误率（CER，以渲染文本为参照）。
这样即使 tesseract 换成更强的识别器，也能直接看出结论是否稳健：预期是整条文本路线**上移**，
但退化档的排序不变。

    python code/experiments/ocr_backend_comparison.py --root . \
        --levels 0.0 0.2 0.4 --tasks 60

冒烟： --tasks 3 --levels 0.4 --backends rapidocr
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--bench", default="data/test_bench_tmm_rh42")
    ap.add_argument("--levels", nargs="+", type=float, default=[0.0, 0.2, 0.4])
    ap.add_argument("--tasks", type=int, default=60)
    ap.add_argument("--backends", nargs="+", default=["tesseract", "rapidocr"])
    ap.add_argument("--bm25", default="word,char3")
    ap.add_argument("--bge", action="store_true", default=True)
    ap.add_argument("--cache", default="results/ocr_cache")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    for p in (root, root / "code" / "ravt"):
        sys.path.insert(0, str(p))

    import numpy as np
    from datasets_eval import list_tasks
    from scorers import Bm25Scorer, OcrEngine, OcrTextScorer, TextEncoderScorer, ocr_ready

    out = root / a.out
    out.mkdir(parents=True, exist_ok=True)
    avail = {}
    for bk in a.backends:
        ok, why = ocr_ready(bk)
        avail[bk] = ok
        print(f"OCR 后端 {bk}: {'可用' if ok else '不可用 —— ' + why}")
    a.backends = [bk for bk in a.backends if avail[bk]]
    if not a.backends:
        print("❌ 没有可用后端")
        return 1

    results = []
    for lv in a.levels:
        tasks = list_tasks(root / a.bench, noise_levels=[lv], max_tasks=a.tasks)
        print(f"\n=== noise={lv}：{len(tasks)} 个 task ===")
        if not tasks:
            continue
        for bk in a.backends:
            t0 = time.time()
            engine = OcrEngine(cache_dir=str(root / a.cache), backend=bk)
            scorers = []
            for tok in [t.strip() for t in a.bm25.split(",") if t.strip()]:
                scorers.append(OcrTextScorer(f"BM25-{tok}+OCR[{bk}]",
                                             base=Bm25Scorer(f"BM25-{tok}", tokenizer=tok),
                                             engine=engine))
            if a.bge:
                bge_path = root / "models_local" / "bge-m3"
                if bge_path.exists():
                    scorers.append(OcrTextScorer(
                        f"BGE-M3+OCR[{bk}]",
                        base=TextEncoderScorer(f"BGE-M3[{bk}]", model_id=str(bge_path),
                                               pooling="cls", batch_size=64),
                        engine=engine))
            for sc in scorers:
                sc.prepare()
                hit = n = tied = 0
                cers = []
                for task in tasks:
                    docs = sc.encode_documents(task)
                    scores, extra = sc.score_query(task.query, docs, task=task)
                    scores = np.asarray(scores, dtype=np.float64)
                    if scores.size and float(np.nanstd(scores)) == 0.0:
                        tied += 1            # 全 0 分 = 查询词一个都没命中，保守记为未命中
                    elif int(np.argmax(scores)) == int(task.gt_block):
                        hit += 1
                    n += 1
                    cer = (extra or {}).get("ocr_cer")
                    if cer is not None and cer == cer:
                        cers.append(float(cer))
                row = {"level": lv, "backend": bk, "method": sc.name, "n": n,
                       "top1": 100.0 * hit / max(n, 1), "ties": tied,
                       "cer": (sum(cers) / len(cers)) if cers else None,
                       "seconds": round(time.time() - t0, 1)}
                results.append(row)
                print(f"   {sc.name:<26} top-1={row['top1']:5.1f}%  "
                      f"CER={row['cer'] if row['cer'] is None else round(row['cer'], 4)}  "
                      f"全 0 查询 {tied}  用时 {row['seconds']}s")
                sc.close()

    # 汇总：每个 (档位, 方法) 一行，列出各后端
    backends = sorted({r["backend"] for r in results})
    methods = sorted({r["method"].split("[")[0] for r in results})
    lines = ["| 噪声档 | 方法 | " + " | ".join(f"{b} top-1" for b in backends) + " |",
             "|" + "---|" * (len(backends) + 2)]
    for lv in a.levels:
        for m in methods:
            cells = []
            for b in backends:
                v = next((r for r in results if r["level"] == lv and r["backend"] == b
                          and r["method"].split("[")[0] == m), None)
                cells.append(f"{v['top1']:.1f}%" if v else "-")
            if any(c != "-" for c in cells):
                lines.append(f"| {lv} | {m} | " + " | ".join(cells) + " |")
    cer_lines = ["| 噪声档 | " + " | ".join(f"{b} CER" for b in backends) + " |",
                 "|" + "---|" * (len(backends) + 1)]
    for lv in a.levels:
        cells = []
        for b in backends:
            v = next((r for r in results if r["level"] == lv and r["backend"] == b
                      and r["cer"] is not None), None)
            cells.append(f"{v['cer']:.3f}" if v else "-")
        cer_lines.append(f"| {lv} | " + " | ".join(cells) + " |")
    body = ("# 阶段 B：OCR 后端对照（同一批 task、同一套检索器）\n\n"
            f"每个档位 {a.tasks} 个 task（与主网格同一种抽样）。\n\n"
            + "\n".join(lines) + "\n\n## OCR 字符错误率（以渲染文本为参照）\n\n"
            + "\n".join(cer_lines) +
            "\n\n读法：更强的识别器会把整条文本路线上移（尤其低噪声档），"
            "如果退化档的排序随之改变，说明原来的结论依赖 tesseract；"
            "如果排序不变，说明结论对识别器选择稳健。\n")
    (out / "ocr_backends.md").write_text(body, encoding="utf-8")
    (out / "ocr_backends.json").write_text(json.dumps(results, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    print("\n" + body)
    print(f"写出 {out}/ocr_backends.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
