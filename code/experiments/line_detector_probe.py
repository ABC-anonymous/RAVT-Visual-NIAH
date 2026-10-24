"""实验 4：行检测的开销、失败统计，以及"换一个检测器会怎样"。

三件事，全部 CPU/GPU 都很轻：
 1) 计时：`detect_lines` 在 clean 页上的每页耗时（中位/95 分位）；
 2) 失败统计：各档位每页行数、tile 数分布，整页 0 行的页数（数据来自已有的
    detect_stats.json，若有）；
 3) 检测器替换敏感性：用"等距网格"这种明显更差的检测器重建 tile 并评测，
    与投影轮廓检测器对比——差异就是整套系统对上游检测质量的敏感度。

    python code/experiments/line_detector_probe.py --root . --pages 120
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def grid_boxes(img, row_h: int):
    """等距网格：把页面按固定行高切成等距条带（一个故意更粗糙的"检测器"）。"""
    return [(0, y, img.width, min(y + row_h, img.height))
            for y in range(0, img.height - row_h, row_h)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--ckpt", default="checkpoints/full_p14_rh42_s0.pth")
    ap.add_argument("--clip", default="clip_local_weights_224")
    ap.add_argument("--row-height", type=int, default=42)
    ap.add_argument("--pages", type=int, default=120, help="计时的页数")
    ap.add_argument("--eval-pages", type=int, default=150, help="对比检测器时评测的页数")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    for p in (root, root / "code" / "ravt"):
        sys.path.insert(0, str(p))

    import numpy as np
    import torch
    from PIL import Image
    import evaluate_real_pages as rde

    out = root / a.out
    out.mkdir(parents=True, exist_ok=True)
    rde.ROW_H = a.row_height
    docs_dir = root / "vidore_docvqa"
    pages = [json.loads(l) for l in (docs_dir / "pages.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    queries = [json.loads(l) for l in (docs_dir / "queries.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]

    # ---- 1) 计时 ----
    times, lines_per_page = [], []
    for rec in pages[:a.pages]:
        with Image.open(docs_dir / "pages" / rec["file"]) as im:
            t0 = time.perf_counter()
            boxes = rde.detect_lines(im)
            times.append((time.perf_counter() - t0) * 1000)
            lines_per_page.append(len(boxes))
    times.sort()
    timing = {"pages": len(times),
              "ms_per_page_median": times[len(times) // 2],
              "ms_per_page_p95": times[int(0.95 * len(times)) - 1],
              "lines_median": sorted(lines_per_page)[len(lines_per_page) // 2],
              "pages_with_zero_lines": sum(1 for x in lines_per_page if x == 0)}
    print(f"检测耗时：中位 {timing['ms_per_page_median']:.1f} ms/页，"
          f"p95 {timing['ms_per_page_p95']:.1f} ms/页；每页行数中位 "
          f"{timing['lines_median']}；整页 0 行 {timing['pages_with_zero_lines']}/{len(times)}")

    # ---- 2) 已有档位的失败统计 ----
    stats = {}
    for grade, sub in (("clean", ""), ("0.2", "deg02"), ("0.4", "deg04")):
        f = root / "results" / "vidore_docvqa" / sub / "detect_stats.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        rows = sorted(x["lines"] for x in d)
        tiles = sorted(x["tiles"] for x in d)
        stats[grade] = {"pages": len(d), "lines_median": rows[len(rows) // 2],
                        "tiles_total": sum(tiles), "tiles_median": tiles[len(tiles) // 2],
                        "pages_zero_lines": sum(1 for x in rows if x == 0)}
        print(f"[{grade}] 每页行数中位 {stats[grade]['lines_median']}，tile 总数 "
              f"{stats[grade]['tiles_total']}，整页 0 行 {stats[grade]['pages_zero_lines']} 页")

    # ---- 3) 检测器替换：等距网格 ----
    from scorers import RavtScorer
    sc = RavtScorer("RAVT-det", str(root / a.clip), str(root / a.ckpt),
                    row_height=a.row_height)
    sc.prepare()

    def tiles_for(box_fn, tag):
        feats, ids = [], []
        for rec in pages[:a.eval_pages]:
            with Image.open(docs_dir / "pages" / rec["file"]) as im:
                boxes = box_fn(im)
                ms = rde.mosaics(rde.row_images(im, boxes))
            if ms:
                with torch.no_grad():
                    px = sc._processor(images=ms, return_tensors="pt")["pixel_values"]
                    f = sc._model.encode_vision(px.to(sc.device)).cpu().numpy()
                feats.append(f)
                ids.append(rec["page_id"])
        return feats, ids

    variants = {"projection": tiles_for(lambda im: rde.detect_lines(im), "projection"),
                "fixed_grid": tiles_for(lambda im: grid_boxes(im, a.row_height), "grid")}
    table = {}
    for tag, (feats, ids) in variants.items():
        n5 = r1 = n = 0
        for q in queries:
            gold = set(q["relevant_pages"])
            if not (gold & set(ids)):
                continue
            enc = sc._tokenizer(q["query"], padding="max_length", max_length=77,
                                truncation=True, return_tensors="pt").to(sc.device)
            with torch.no_grad():
                t = sc._model.encode_text(enc["input_ids"], enc["attention_mask"])[0]
            best = {}
            for f, pid in zip(feats, ids):
                r = torch.from_numpy(f).to(sc.device)
                s = torch.einsum("bnd,d->bn", r, t).max(dim=-1).values.max().item()
                best[pid] = max(best.get(pid, -1e9), s)
            order = sorted(best, key=lambda p: -best[p])[:5]
            rel = [1 if p in gold else 0 for p in order]
            n5 += sum(r / math.log2(i + 2) for i, r in enumerate(rel))
            r1 += 1.0 if (order and order[0] in gold) else 0.0
            n += 1
        table[tag] = {"nDCG@5": 100 * n5 / max(n, 1), "R@1": 100 * r1 / max(n, 1),
                      "queries": n, "tiles": sum(len(f) for f in feats)}
        print(f"[{tag:<11}] tiles={table[tag]['tiles']:6d}  "
              f"nDCG@5={table[tag]['nDCG@5']:5.1f}  R@1={table[tag]['R@1']:5.1f}")

    body = ("# 实验 4：行检测的开销、失败统计与替换敏感性\n\n"
            f"- 检测耗时：中位 **{timing['ms_per_page_median']:.1f} ms/页**，"
            f"p95 {timing['ms_per_page_p95']:.1f} ms/页（{timing['pages']} 页，clean）\n"
            f"- 整页 0 行：clean {timing['pages_with_zero_lines']}/{timing['pages']}\n\n"
            "| 档位 | 页数 | 每页行数中位 | tile 总数 | 整页 0 行 |\n|---|---|---|---|---|\n"
            + "\n".join(f"| {g} | {v['pages']} | {v['lines_median']} | {v['tiles_total']} | "
                        f"{v['pages_zero_lines']} |" for g, v in stats.items())
            + "\n\n| 检测器 | tile 数 | nDCG@5 | R@1 |\n|---|---|---|---|\n"
            + "\n".join(f"| {k} | {v['tiles']} | {v['nDCG@5']:.1f} | {v['R@1']:.1f} |"
                        for k, v in table.items())
            + "\n")
    (out / "detector.md").write_text(body, encoding="utf-8")
    (out / "detector.json").write_text(json.dumps(
        {"timing": timing, "grades": stats, "detectors": table},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n写出 {out}/detector.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
