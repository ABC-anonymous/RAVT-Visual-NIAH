"""阶段 A：500 页真实扫描件上的原生任务，文本基线 vs RAVT（同一套查询、同一套 qrels）。

为什么不能沿用第三批的构造：那一版把"某个 tile 自己的 OCR 文本"当查询，而这条文本同时
也在语料里，于是任何文本检索器都能精确命中自己 —— BM25 必然是 100%，对照没有意义。
这一版改成真实部署场景：

    查询 = 干净扫描件上某个 tile 的 OCR 文本（可截断成 snippet）
    语料 = 同一档位（clean / 0.2 / 0.4）的全部 tile：文本侧用该档位的 OCR 文本，视觉侧用图像
    金标 = 该 tile 所在的**页**（与论文 ViDoRe 表一致：页分 = 该页所有 tile 的最大分）

于是三条路线看到的是同一批查询、同一批页面，而文本路线必须在**识别退化后的**文本上检索，
这正是审稿人要求的"规模化真实退化评测"。两种查询模式：

    full     —— 原始 OCR 文本（在 clean 档上与语料文本逐字相同，作为 exact-match 对照，会接近 100%）
    snippet  —— 取前若干词（默认 70% 词、上限 40 词），任何档位都不会与语料文本完全相同

    python code/experiments/native_text_baselines.py --root . \
        --queries 300 --seeds 0 1 2 3 4

冒烟： --queries 20 --limit-tiles 2000 --seeds 0
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

#: 档位 -> 相对 --vidore 的子目录
GRADES = (("clean", ""), ("0.2", "deg02"), ("0.4", "deg04"))
QUERY_SEED = 20261016


class TileCorpus:
    """给服务器 scorer 用的最小 task 抽象（与 evaluate_real_pages.TileTask 同形）。"""

    def __init__(self, paths, pages, texts, cache_dir):
        self.image_paths = list(paths)
        self.page_ids = list(pages)
        self.texts = list(texts)
        self.path = Path(cache_dir)
        self.meta = {"block_count": len(self.image_paths)}
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
        return list(self.texts)


def ndcg_at_k(order: list[int], gold: int, k: int) -> float:
    dcg = 0.0
    for rank, idx in enumerate(order[:k]):
        if idx == gold:
            dcg = 1.0 / math.log2(rank + 2)
            break
    return dcg  # ideal DCG is 1.0 (single relevant page)


def page_matrix(scores, pages):
    """tile 分 -> 页分（同页取最大），返回 (页分矩阵, 页顺序)。"""
    import numpy as np

    order_pages = sorted(set(pages))
    col = {p: k for k, p in enumerate(order_pages)}
    out = np.full((scores.shape[0], len(order_pages)), -1e9, dtype=np.float32)
    for j, p in enumerate(pages):
        np.maximum(out[:, col[p]], scores[:, j], out=out[:, col[p]])
    return out, order_pages


def tile_index(tiles_dir: Path, tag: str) -> list[tuple[Path, int, str]]:
    """返回 [(png, page_id, ocr_text)]，跳过没有文本或文本过短的 tile。"""
    out = []
    for png in sorted(tiles_dir.glob("*.png")):
        txt_path = png.with_suffix(".txt")
        if not txt_path.exists():
            continue
        text = txt_path.read_text(encoding="utf-8", errors="replace").strip()
        if len(text) < 20:
            continue
        try:
            page = int(png.stem.split("_")[0])
        except ValueError:
            continue
        out.append((png, page, text))
    print(f"  [{tag}] 可用 tile {len(out)} 个（{len({x[1] for x in out})} 页）")
    return out


def snippet(text: str, frac: float, cap: int) -> str:
    words = text.split()
    keep = max(8, min(cap, int(round(len(words) * frac))))
    return " ".join(words[:keep])


def evaluate(scores, pages, gold_pages):
    """页级指标。"""
    import numpy as np

    P, order_pages = page_matrix(scores, pages)
    col = {p: k for k, p in enumerate(order_pages)}
    r1 = r5 = n5 = 0.0
    for i, gp in enumerate(gold_pages):
        if i >= P.shape[0]:
            break
        g = col.get(gp)
        if g is None:
            continue
        order = list(np.argsort(-P[i]))
        rank = order.index(g) + 1
        r1 += 1.0 if rank == 1 else 0.0
        r5 += 1.0 if rank <= 5 else 0.0
        n5 += ndcg_at_k(order, g, 5)
    n = len(gold_pages)
    return {"n": n, "ndcg@5": 100 * n5 / n, "recall@1": 100 * r1 / n, "recall@5": 100 * r5 / n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--vidore", default="results/vidore_docvqa")
    ap.add_argument("--clip", default="clip_local_weights_224")
    ap.add_argument("--row-height", type=int, default=42)
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--with-deg", action="store_true", help="同时评退化训练版（慢一倍）")
    ap.add_argument("--snippet-frac", type=float, default=0.7)
    ap.add_argument("--snippet-cap", type=int, default=40)
    ap.add_argument("--limit-tiles", type=int, default=0, help="每档最多用多少 tile（冒烟）")
    ap.add_argument("--bm25", default="word,char3")
    ap.add_argument("--bge", action="store_true", default=True)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    for p in (root, root / "code" / "ravt"):
        sys.path.insert(0, str(p))

    import numpy as np
    from scorers import Bm25Scorer, RavtScorer, TextEncoderScorer

    vidore = Path(a.vidore)
    if not vidore.is_absolute():
        vidore = root / vidore
    out = root / a.out
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- 查询：取干净档的 tile 文本 ----------------
    clean_index = tile_index(vidore / "tiles", "clean")
    if not clean_index:
        print("❌ 干净档没有可用 tile，退出")
        return 1
    rng = random.Random(QUERY_SEED)
    pool = list(range(len(clean_index)))
    rng.shuffle(pool)
    qsel = pool[:min(a.queries, len(pool))]
    queries = []
    for i in qsel:
        png, page, text = clean_index[i]
        queries.append({"stem": png.stem, "page": page, "text": text,
                        "snippet": snippet(text, a.snippet_frac, a.snippet_cap)})
    print(f"查询 {len(queries)} 条（种子 {QUERY_SEED}，来自干净档 tile 自身文本）")

    # ---------------- 文本基线 ----------------
    text_scorers = []
    for tok in [t.strip() for t in a.bm25.split(",") if t.strip()]:
        text_scorers.append((f"BM25-{tok}+OCR", Bm25Scorer(f"BM25-{tok}+OCR", tokenizer=tok)))
    if a.bge:
        bge_path = root / "models_local" / "bge-m3"
        if bge_path.exists():
            text_scorers.append(("BGE-M3+OCR",
                                 TextEncoderScorer("BGE-M3+OCR", model_id=str(bge_path),
                                                   pooling="cls", batch_size=64)))
        else:
            print(f"⚠️ 找不到 {bge_path}，跳过 BGE-M3")
    for _, sc in text_scorers:
        if hasattr(sc, "prepare"):
            sc.prepare()

    # ---------------- RAVT 权重 ----------------
    ravt_ckpts = []
    for s in a.seeds:
        p = root / "checkpoints" / f"full_p14_rh42_s{s}.pth"
        if p.exists():
            ravt_ckpts.append((f"RAVT-s{s}", p))
        else:
            print(f"⚠️ 缺 {p.name}，跳过")
    if a.with_deg:
        for s in a.seeds:
            p = root / "checkpoints" / f"deg_rh42_p14_s{s}.pth"
            if p.exists():
                ravt_ckpts.append((f"RAVT-DEG-s{s}", p))

    table, detail = [], {}
    for grade, sub in GRADES:
        d = vidore / sub
        index = tile_index(d / "tiles", grade) if (d / "tiles").is_dir() else []
        if not index:
            print(f"[{grade}] 没有 tile，跳过这一档")
            continue
        if a.limit_tiles:
            index = index[:a.limit_tiles]
        pages_avail = {x[1] for x in index}
        qs = [q for q in queries if q["page"] in pages_avail]
        dropped = len(queries) - len(qs)
        if dropped and a.limit_tiles:
            print(f"   提示：冒烟的 --limit-tiles 只保留前 {a.limit_tiles} 个 tile"
                  f"（≈前 {len(pages_avail)} 页），所以只有这些页的查询可用；"
                  "全量不加该参数时不会丢查询")
        elif len(qs) < 0.5 * len(queries):
            print(f"⚠️ [{grade}] 只有 {len(qs)}/{len(queries)} 条查询的页在这一档存在 —— "
                  "检查该档 tile 的文件名是否与干净档同规范（页号在文件名下划线前），"
                  "以及该页在退化档是否还有被检测到的 tile")
        paths = [x[0] for x in index]
        pages = [x[1] for x in index]
        texts = [x[2] for x in index]
        task = TileCorpus(paths, pages, texts, d / "native_text_corpus")
        gold = [q["page"] for q in qs]
        print(f"\n[{grade}] 语料 {len(paths)} tile / {len(pages_avail)} 页，"
              f"可用查询 {len(qs)}（丢弃 {dropped}，金标页在这一档不存在）")
        row = {"grade": grade, "tiles": len(paths), "pages": len(pages_avail),
               "queries": len(qs), "dropped_queries": dropped, "methods": {}}

        for mode in ("full", "snippet"):
            qtexts = [q["text"] if mode == "full" else q["snippet"] for q in qs]
            for name, sc in text_scorers:
                docs = sc.encode_documents(task)
                scores = np.zeros((len(qs), len(paths)), dtype=np.float32)
                for i, qt in enumerate(qtexts):
                    s, _ = sc.score_query(qt, docs, task=task)
                    scores[i] = np.asarray(s, dtype=np.float32)
                m = evaluate(scores, pages, gold)
                if m:
                    row["methods"][f"{name}|{mode}"] = m
                    print(f"   {name:<16} {mode:<8} nDCG@5={m['ndcg@5']:5.1f}  "
                          f"R@1={m['recall@1']:5.1f}  R@5={m['recall@5']:5.1f}  n={m['n']}")

        for name, ck in ravt_ckpts:
            t0 = time.time()
            sc = RavtScorer(name, str(root / a.clip), str(ck), row_height=a.row_height)
            sc.prepare()
            docs = sc.encode_documents(task)                      # (tiles, 5, D)
            import torch
            with torch.no_grad():
                rows = torch.from_numpy(docs).to(sc.device)
                for mode in ("full", "snippet"):
                    qtexts = [q["text"] if mode == "full" else q["snippet"] for q in qs]
                    scores = np.zeros((len(qs), len(paths)), dtype=np.float32)
                    for i, qt in enumerate(qtexts):
                        enc = sc._tokenizer(qt, padding="max_length", max_length=77,
                                            truncation=True, return_tensors="pt").to(sc.device)
                        t = sc._model.encode_text(enc["input_ids"], enc["attention_mask"])[0]
                        scores[i] = torch.einsum("bnd,d->bn", rows, t).max(dim=-1).values.cpu().numpy()
                    m = evaluate(scores, pages, gold)
                    if m:
                        row["methods"][f"{name}|{mode}"] = m
                        print(f"   {name:<16} {mode:<8} nDCG@5={m['ndcg@5']:5.1f}  "
                              f"R@1={m['recall@1']:5.1f}  R@5={m['recall@5']:5.1f}  n={m['n']}")
            sc.close()
            print(f"   （{name} 用时 {time.time() - t0:.0f}s）")
        table.append(row)
        detail[grade] = row

    # ---------------- 汇总（按方法：RAVT 取种子均值） ----------------
    lines = ["| 方法 | 查询 | clean | 0.2 | 0.4 |", "|---|---|---|---|---|"]
    #: (标签, 键前缀, 是否多种子)——键形如 "RAVT-s0|full"、`"BM25-char3+OCR|full"`，
    #: 所以匹配要用 startswith 而不是 startswith(prefix + "|")，否则 RAVT 行会被漏掉。
    labels = [("RAVT (5 seeds)", "RAVT-s", True), ("RAVT-DEG", "RAVT-DEG-s", True),
              ("BM25-word+OCR", "BM25-word+OCR", False),
              ("BM25-char3+OCR", "BM25-char3+OCR", False),
              ("BGE-M3+OCR", "BGE-M3+OCR", False)]
    for label, prefix, multi in labels:
        for mode in ("full", "snippet"):
            cells = []
            for grade, _ in GRADES:
                r = detail.get(grade)
                if not r:
                    cells.append("-")
                    continue
                vals = [v["ndcg@5"] for k, v in r["methods"].items()
                        if k.startswith(prefix) and k.endswith("|" + mode)
                        and (multi or k.count("|") == 1)]
                if not vals:
                    cells.append("-")
                elif multi and len(vals) > 1:
                    m = sum(vals) / len(vals)
                    sd = (sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5
                    cells.append(f"{m:.1f} ± {sd:.1f}")
                else:
                    cells.append(f"{vals[0]:.1f}")
            if any(c != "-" for c in cells):
                lines.append(f"| {label} | {mode} | " + " | ".join(cells) + " |")
    body = ("# 阶段 A：500 页真实扫描件上的原生任务（页级 nDCG@5）\n\n"
            f"查询 {len(queries)} 条，取自干净档 tile 的 OCR 文本；语料为同档位全部 tile，"
            "金标是该 tile 所在页（页分 = 该页 tile 的最大分）。整行数字为 nDCG@5。\n\n"
            "`full` 模式在 clean 档上查询与语料文本逐字相同，文本基线会接近 100%，那是 "
            "exact-match 对照；`snippet` 模式取前若干词，任何档位都不与语料文本完全相同，"
            "是给文本基线用的公平口径。RAVT 行为多种子均值。\n\n"
            + "\n".join(lines) + "\n")
    (out / "native_text.md").write_text(body, encoding="utf-8")
    (out / "native_text.json").write_text(
        json.dumps({"queries": len(queries), "query_seed": QUERY_SEED,
                    "snippet_frac": a.snippet_frac, "grades": detail},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n" + body)
    print(f"写出 {out}/native_text.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
