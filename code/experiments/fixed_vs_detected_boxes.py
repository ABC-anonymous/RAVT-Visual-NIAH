"""实验 3：几何畸变下，"行检测错位"与"编码器读不了形变字形"各占多少。

做法与光度档相同、只是把算子换成几何的（rotation / perspective / fold）：把*干净页*上
检出的行框当作"完美但已错位"的几何，用它去裁**形变后**的像素；同时用形变页自己的行检测
结果再裁一份。两次评测只差"行框从哪来"，于是差距就是行检测在几何退化下的贡献，
剩下的就是编码器面对形变字形的损失（这正是审稿人要求的受控实验）。

    python code/experiments/fixed_vs_detected_boxes.py --root . \
        --ops rotation,perspective,fold --severity 0.2 --limit-pages 150
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--ckpt", default="checkpoints/full_p14_rh42_s0.pth")
    ap.add_argument("--clip", default="clip_local_weights_224")
    ap.add_argument("--row-height", type=int, default=42)
    ap.add_argument("--ops", default="rotation,perspective,fold")
    ap.add_argument("--severity", type=float, default=0.2)
    ap.add_argument("--limit-pages", type=int, default=150)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    for p in (root, root / "code" / "ravt",
              root / "code" / "ravt"):
        sys.path.insert(0, str(p))

    import numpy as np
    import torch
    from PIL import Image
    from model_defs import CorruptionEngine
    import evaluate_real_pages as rde
    from scorers import RavtScorer

    out = root / a.out
    out.mkdir(parents=True, exist_ok=True)
    docs_dir = root / "vidore_docvqa"
    pages = [json.loads(l) for l in (docs_dir / "pages.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    queries = [json.loads(l) for l in (docs_dir / "queries.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    if a.limit_pages:
        pages = pages[:a.limit_pages]
    ops = [o.strip() for o in a.ops.split(",") if o.strip()]

    engine = CorruptionEngine(random.Random(20261015))
    sc = RavtScorer("RAVT-geo", str(root / a.clip), str(root / a.ckpt),
                    row_height=a.row_height)
    sc.prepare()
    rde.ROW_H = a.row_height

    def encode_pages(page_images):
        feats = []
        with torch.no_grad():
            for i in range(0, len(page_images), 16):
                batch = page_images[i:i + 16]
                flat = [im for t in batch for im in t]
                px = sc._processor(images=flat, return_tensors="pt")["pixel_values"].to(sc.device)
                f = sc._model.encode_vision(px).cpu().numpy()
                k = 0
                for t in batch:
                    feats.append(f[k:k + len(t)])
                    k += len(t)
        return feats

    def page_scores(query, feats_list, page_ids):
        enc = sc._tokenizer(query, padding="max_length", max_length=77, truncation=True,
                            return_tensors="pt").to(sc.device)
        with torch.no_grad():
            t = sc._model.encode_text(enc["input_ids"], enc["attention_mask"])[0]
        best = {}
        for feats, pid in zip(feats_list, page_ids):
            if feats is None or len(feats) == 0:
                continue
            r = torch.from_numpy(feats).to(sc.device)
            s = torch.einsum("bnd,d->bn", r, t).max(dim=-1).values.max().item()
            best[pid] = max(best.get(pid, -1e9), s)
        return best

    results = []
    for op in ops:
        print(f"\n=== [{op}] severity={a.severity}，{len(pages)} 页 ===")
        deg_dir = out / "geo" / op
        (deg_dir / "pages").mkdir(parents=True, exist_ok=True)
        clean_boxes, deg_boxes, ids = {}, {}, []
        for rec in pages:
            src = docs_dir / "pages" / rec["file"]
            dst = deg_dir / "pages" / rec["file"]
            if not dst.exists():
                with Image.open(src) as im:
                    # ????? rotation/perspective ? fillcolor ??? RGB?
                    # ????? TypeError?????? RGB ????????????
                    warped = engine.apply_image(im.convert("RGB"), a.severity, [op],
                                                coverage=1.0)
                warped.save(dst)
            with Image.open(src) as im:
                clean_boxes[rec["page_id"]] = rde.detect_lines(im)
            with Image.open(dst) as im:
                deg_boxes[rec["page_id"]] = rde.detect_lines(im.convert("L"))
            ids.append(rec["page_id"])
        n_clean = sum(len(v) for v in clean_boxes.values())
        n_deg = sum(len(v) for v in deg_boxes.values())
        empty = sum(1 for v in deg_boxes.values() if not v)
        print(f"  检出框：干净页 {n_clean}，形变页 {n_deg}（整页 0 框 {empty} 页）")

        variants = {}
        for tag, boxes in (("detected", deg_boxes), ("clean_boxes", clean_boxes)):
            tiles, tile_ids = [], []
            file_of = {r["page_id"]: r["file"] for r in pages}
            for pid in ids:
                with Image.open(deg_dir / "pages" / file_of[pid]) as im:
                    rows = rde.row_images(im.convert("L"), boxes[pid])
                ms = rde.mosaics(rows)
                if ms:
                    tiles.append(ms)
                    tile_ids.append(pid)
            feats = encode_pages(tiles)
            variants[tag] = (feats, tile_ids)
            print(f"  [{tag}] {sum(len(t) for t in tiles)} 个 tile")

        row = {"op": op, "severity": a.severity, "pages": len(ids),
               "boxes_clean": n_clean, "boxes_warped": n_deg, "empty_pages": empty}
        for tag in ("detected", "clean_boxes"):
            feats, tile_ids = variants[tag]
            n5 = r1 = 0.0
            n = 0
            for q in queries:
                gold = set(q["relevant_pages"])
                if not (gold & set(tile_ids)):
                    continue
                best = page_scores(q["query"], feats, tile_ids)
                if not best:
                    continue
                order = sorted(best, key=lambda p: -best[p])[:5]
                rel = [1 if p in gold else 0 for p in order]
                dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
                n5 += dcg
                r1 += 1.0 if (order and order[0] in gold) else 0.0
                n += 1
            row[tag] = {"ndcg@5": 100 * n5 / max(n, 1), "recall@1": 100 * r1 / max(n, 1),
                        "queries": n}
            print(f"   [{tag}] nDCG@5={row[tag]['ndcg@5']:.1f}  R@1={row[tag]['recall@1']:.1f}"
                  f"  (n={n})")
        results.append(row)

    lines = ["| 算子 | 形变页检出框 | 整页 0 框 | 自检行框 nDCG@5 | 干净页行框 nDCG@5 | 差值 |",
             "|---|---|---|---|---|---|"]
    for r in results:
        d, c = r["detected"]["ndcg@5"], r["clean_boxes"]["ndcg@5"]
        lines.append(f"| {r['op']} | {r['boxes_warped']} | {r['empty_pages']} | {d:.1f} | "
                     f"{c:.1f} | {c - d:+.1f} |")
    body = ("# 实验 3：几何畸变下的受控拆分\n\n"
            f"severity={a.severity}，每档 {len(pages)} 页；两行只差'行框从哪来'。\n\n"
            + "\n".join(lines) +
            "\n\n'干净页行框'是把完美几何用在形变像素上（已错位），因此它测的是编码器面对"
            "形变字形本身的损失；两者之差即行检测在此算子下的贡献。\n")
    (out / "geo_split.md").write_text(body, encoding="utf-8")
    (out / "geo_split.json").write_text(json.dumps(results, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    print("\n" + body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
