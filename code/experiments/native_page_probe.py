"""实验 2：把"原生任务（段落检索）"从 40 页扩到 500 页真实扫描件。

构造方式完全不需要人工标注：ViDoRe 的 500 张真实页面已经切成 tile 并被 OCR（tile 旁边
的 .txt 就是 OCR 文本）。于是：

    查询 = 某个 tile 的 OCR 文本（一段真实正文）
    语料 = 同一档位下的全部 tile
    金标 = 该 tile 所在的**页**（页级检索，与论文里 ViDoRe 表的口径一致）

    ?? RAVT???????????????????????????? OCR ???
    ?? n-gram ?????????????????????
这既扩大了真实证据的体量，也把评测换回了模型的**原生任务**（段落→段落），
而不需要再依赖"问题式页面检索"。

    python code/experiments/native_page_probe.py --root . --queries 300
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class TileCorpus:
    """让服务器的 scorer 认得的最小 task 抽象（与 evaluate_real_pages.TileTask 同形）。"""

    def __init__(self, paths, page_of, texts, cache_dir):
        self.image_paths = paths
        self.page_ids = page_of
        self.texts = texts
        self.path = cache_dir
        self.meta = {"block_count": len(paths)}
        self.query = ""

    @property
    def blocks(self):
        return len(self.image_paths)

    def load_images(self):
        from PIL import Image
        out = []
        for p in self.image_paths:
            with Image.open(p) as im:
                out.append(im.convert("RGB"))
        return out

    def load_texts(self):
        return self.texts


def ndcg_at_k(rel: list[int], k: int) -> float:
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel[:k]))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, sum(1 for r in rel if r))))
    return dcg / ideal if ideal else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--vidore", default="results/vidore_docvqa")
    ap.add_argument("--ckpt", default="checkpoints/full_p14_rh42_s0.pth")
    ap.add_argument("--clip", default="clip_local_weights_224")
    ap.add_argument("--row-height", type=int, default=42)
    ap.add_argument("--queries", type=int, default=300, help="抽多少个查询（固定种子）")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    for p in (root, root / "code" / "ravt",
              root / "code" / "ravt"):
        sys.path.insert(0, str(p))

    import numpy as np
    import torch
    from scorers import RavtScorer

    out = root / a.out
    out.mkdir(parents=True, exist_ok=True)
    grades = {"clean": "", "0.2": "deg02", "0.4": "deg04"}

    ravt = RavtScorer("RAVT-native", str(root / a.clip), str(root / a.ckpt),
                      row_height=a.row_height)
    ravt.prepare()
    rng = random.Random(20261015)
    table, detail = [], {}

    for grade, sub in grades.items():
        d = root / a.vidore / sub
        tiles = sorted((d / "tiles").glob("*.png"))
        if len(tiles) < 100:
            print(f"[{grade}] tile 太少（{len(tiles)}），跳过")
            continue
            index = []
        for t in tiles:
            tp = t.with_suffix(".txt")
            if not tp.exists():
                continue
            txt = tp.read_text(encoding="utf-8", errors="replace").strip()
            if len(txt) < 20:
                continue
            index.append((t, int(t.stem.split("_")[0]), txt))
        if not index:
            print(f"[{grade}] 没有可用的 tile 文本（缺 .txt？），跳过")
            continue
        paths = [x[0] for x in index]
        pages = [x[1] for x in index]
        texts = [x[2] for x in index]
        qpool = list(range(len(index)))
        rng.shuffle(qpool)
        qsel = qpool[:min(a.queries, len(qpool))]
        task = TileCorpus(paths, pages, texts, d / "native_corpus")
        print(f"[{grade}] 语料 {len(paths)} 个 tile / {len(set(pages))} 页，查询 {len(qsel)} 条")

        # --- RAVT：一次编码，逐查询打分 ---
        docs = ravt.encode_documents(task)                      # (B, R, D)
        scores_r = np.zeros((len(qsel), len(paths)), dtype=np.float32)
        with torch.no_grad():
            rows = torch.from_numpy(docs).to(ravt.device)
            for i, qi in enumerate(qsel):
                enc = ravt._tokenizer(texts[qi], padding="max_length", max_length=77,
                                      truncation=True, return_tensors="pt").to(ravt.device)
                t = ravt._model.encode_text(enc["input_ids"], enc["attention_mask"])[0]
                scores_r[i] = torch.einsum("bnd,d->bn", rows, t).max(dim=-1).values.cpu().numpy()

        def page_level(scores):
            """页分 = 该页所有 tile 的最大分（与论文口径一致）。"""
            order_pages = sorted(set(pages))
            idx = {p: k for k, p in enumerate(order_pages)}
            out_ = np.full((scores.shape[0], len(order_pages)), -1e9, dtype=np.float32)
            for j, p in enumerate(pages):
                np.maximum(out_[:, idx[p]], scores[:, j], out=out_[:, idx[p]])
            return out_, order_pages

        row = {"grade": grade, "tiles": len(paths), "pages": len(set(pages)),
               "queries": len(qsel)}
        for name, sc in (("RAVT", scores_r),):
            P, order_pages = page_level(sc)
            gold_idx = {p: k for k, p in enumerate(order_pages)}
            r5 = r1 = n5 = 0.0
            for i, qi in enumerate(qsel):
                g = gold_idx[pages[qi]]
                rank = int(np.where(np.argsort(-P[i]) == g)[0][0]) + 1
                rel = [1 if k == g else 0 for k in np.argsort(-P[i])[:5]]
                n5 += ndcg_at_k(rel, 5)
                r1 += 1.0 if rank == 1 else 0.0
                r5 += 1.0 if rank <= 5 else 0.0
            n = max(len(qsel), 1)
            row[name] = {"ndcg@5": 100 * n5 / n, "recall@1": 100 * r1 / n,
                         "recall@5": 100 * r5 / n}
            print(f"   {name:<16} nDCG@5={row[name]['ndcg@5']:5.1f}  "
                  f"R@1={row[name]['recall@1']:5.1f}  R@5={row[name]['recall@5']:5.1f}")
        table.append(row)
        detail[grade] = row

    lines = ["| ?? | tile ? | ?? | ??? | RAVT nDCG@5 | R@1 |",
             "|---|---|---|---|---|---|"]
    for r in table:
        lines.append(f"| {r['grade']} | {r['tiles']} | {r['pages']} | {r['queries']} | "
                     f"{r['RAVT']['ndcg@5']:.1f} | {r['RAVT']['recall@1']:.1f} |")
    body = ("# 实验 2：500 页真实扫描件上的原生任务（段落→页）\n\n"
            "查询取自 tile 自身的 OCR 文本，语料为同档位全部 tile，金标为 tile 所在页。\n\n"
            + "\n".join(lines) + "\n")
    (out / "native500.md").write_text(body, encoding="utf-8")
    (out / "native500.json").write_text(json.dumps(detail, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\n写出 {out}/native500.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
