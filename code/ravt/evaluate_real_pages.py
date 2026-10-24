"""真实扫描文档上的外部验证：不使用任何合成退化，直接在 DocVQA 页面上评测。

解决"我们的模型是行切分的，真实页面怎么找行"这个问题，方案是：

  ① 行检测（CPU，本脚本自带）：二值化后用**水平投影剖面**找文本行带，
     再在该带内用垂直投影裁掉左右空白，得到每行的包围盒。
     这是文档分析里的经典做法，对印刷体扫描件足够稳；
     多栏/表格页面会过度切分，脚本会打印统计让你看到有多少页属于这种情况。
  ② 归一化到训练尺度：把检测到的行缩放到"文字高 ≈11px、行高 44px"，
     与训练渲染完全一致；过长的行按宽度切成多个"行"（保持字形尺度不变）。
  ③ 拼 mosaic：连续 5 行拼成 224×224（底部 4px 白边，与训练相同）。

评测协议（页级检索，与 DocVQA BEIR 的 qrels 对齐）：
  查询 -> 每张 tile 打分 -> 取该页所有 tile 的最大分 -> 页级排序
  -> Recall@1/5/10、nDCG@10。
文本基线用同一套 tile（先 OCR 再检索），所以两边面对的是同一批真实像素。

用法:
    python code/ravt/evaluate_real_pages.py --docs data/vidore_docvqa --stage tiles
    python code/ravt/evaluate_real_pages.py --docs data/vidore_docvqa --models BM25,BGE
    python code/ravt/evaluate_real_pages.py --docs data/vidore_docvqa --models all \
        --clip-path clip_local_weights_224 \
        --ckpt checkpoints/full_p14_s0.pth checkpoints/deg_adapt_p14_s0.pth
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
from PIL import Image

ROW_W = 224
#: 行高：与训练渲染一致（font 11）。42 = 3×patch14，用于行对齐实验。
ROW_H = int(os.environ.get("RAVT_BLOCK_HEIGHT", 44))
TEXT_H = 11                      # 训练里文字实际高度（像素）
N_ROWS = 5                       # 每个 mosaic 5 行


# --------------------------------------------------------------------------- #
# ① 行检测
# --------------------------------------------------------------------------- #
def otsu_threshold(arr: np.ndarray) -> int:
    hist, _ = np.histogram(arr, bins=256, range=(0, 256))
    total = arr.size
    sum_all = np.dot(np.arange(256), hist)
    sum_b = 0.0
    w_b = 0.0
    best, thr = -1.0, 128
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best:
            best, thr = var, t
    return thr


def detect_lines(img: Image.Image, ink_frac: float = 0.08,
                 min_h: int = 5, merge_gap: int = 2,
                 thin_frac: float = 0.35) -> list[tuple[int, int, int, int]]:
    """返回 [(x0, y0, x1, y1), ...]，按 y 排序。"""
    arr = np.asarray(img.convert("L"))
    thr = otsu_threshold(arr)
    ink = arr < thr
    proj = ink.sum(axis=1).astype(np.float32)
    if proj.max() <= 0:
        return []
    # 阈值取"平滑投影剖面的 99 分位 × ink_frac"：用分位数而不是最大值，
    # 避免页面边框/黑块这类离群行把阈值抬飞；ink_frac=0.08 是在 DocVQA 页面上
    # 实测出来的（每页能切出 20~45 条带，与页面的实际行数量级一致）。
    sm = np.convolve(proj, np.ones(3, dtype=np.float32) / 3.0, mode="same")
    cut = max(1.0, float(np.percentile(sm, 99)) * ink_frac)
    bands: list[list[int]] = []
    inside = False
    for y, v in enumerate(proj):
        if v > cut and not inside:
            inside, start = True, y
        elif v <= cut and inside:
            inside = False
            bands.append([start, y])
    if inside:
        bands.append([start, len(proj)])
    # 合并过近的行带（同一个字的上/下部件）
    merged: list[list[int]] = []
    for b in bands:
        if merged and b[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = b[1]
        else:
            merged.append(b)
    heights = [b[1] - b[0] for b in merged if b[1] - b[0] >= min_h]
    med_h = float(np.median(heights)) if heights else 20.0
    merged = [b for b in merged if b[1] - b[0] >= max(min_h, med_h * thin_frac)]
    heights = [b[1] - b[0] for b in merged]
    med_h = float(np.median(heights)) if heights else 20.0
    max_h = max(min_h + 1, int(round(med_h * 1.8)))
    out = []
    for y0, y1 in merged:
        if y1 - y0 < min_h:
            continue
        # 过高的带（例如整段无空行）按 max_h 切开
        if y1 - y0 > max_h:
            n = int(math.ceil((y1 - y0) / max_h))
            step = (y1 - y0) / n
            chunks = [(int(round(y0 + k * step)), int(round(y0 + (k + 1) * step)))
                      for k in range(n)]
        else:
            chunks = [(y0, y1)]
        for cy0, cy1 in chunks:
            col = ink[cy0:cy1].sum(axis=0)
            nz = np.nonzero(col > 0)[0]
            if len(nz) < 8:
                continue
            out.append((int(nz[0]), int(cy0), int(nz[-1] + 1), int(cy1)))
    return out


# --------------------------------------------------------------------------- #
# ② 归一化 + ③ 拼 mosaic
# --------------------------------------------------------------------------- #
def row_images(img: Image.Image, boxes: list[tuple[int, int, int, int]],
               pad: int = 2) -> list[Image.Image]:
    """把每个行包围盒缩放到训练尺度，贴到 224x44 的白底行里。"""
    arr = np.asarray(img.convert("L"))
    rows: list[Image.Image] = []
    for (x0, y0, x1, y1) in boxes:
        h = max(1, y1 - y0)
        scale = TEXT_H / h
        # 过长的一行按宽度切成多个"行"（保持字形大小不变）
        max_src_w = int((ROW_W - 2 * pad) / scale)
        n_chunks = max(1, int(math.ceil((x1 - x0) / max_src_w)))
        step = max(1, int(math.ceil((x1 - x0) / n_chunks)))
        for k in range(0, x1 - x0, step):
            cx0, cx1 = x0 + k, min(x1, x0 + k + step)
            if cx1 - cx0 < 4 or (cx1 - cx0) * scale < 6:
                continue
            crop = arr[max(0, y0 - pad):min(arr.shape[0], y1 + pad), cx0:cx1]
            im = Image.fromarray(crop)
            new_w = max(4, int(round(im.width * scale)))
            new_h = max(4, int(round(im.height * scale)))
            im = im.resize((new_w, new_h), Image.LANCZOS)
            canvas = Image.new("L", (ROW_W, ROW_H), 255)
            canvas.paste(im, (pad, max(0, (ROW_H - new_h) // 2)))
            rows.append(canvas)
    return rows


def mosaics(rows: list[Image.Image]) -> list[Image.Image]:
    out = []
    for i in range(0, len(rows) - N_ROWS + 1, N_ROWS):
        canvas = Image.new("L", (ROW_W, 224), 255)
        for j, r in enumerate(rows[i:i + N_ROWS]):
            canvas.paste(r, (0, j * ROW_H))
        out.append(canvas)
    return out


def build_tiles(docs: Path, out: Path, limit_pages: int | None, debug: bool) -> dict:
    pages = [json.loads(l) for l in (docs / "pages.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    if limit_pages:
        pages = pages[:limit_pages]
    (out / "tiles").mkdir(parents=True, exist_ok=True)
    (out / "rows").mkdir(parents=True, exist_ok=True)
    index = []
    stats = []
    for rec in pages:
        img = Image.open(docs / "pages" / rec["file"])
        boxes = detect_lines(img)
        rows = row_images(img, boxes)
        tiles = mosaics(rows)
        for j, r in enumerate(rows):
            r.save(out / "rows" / f"{rec['page_id']:04d}_{j:03d}.png")
        for j, t in enumerate(tiles):
            name = f"{rec['page_id']:04d}_{j:03d}.png"
            t.convert("RGB").save(out / "tiles" / name)
            index.append({"tile": name, "page_id": rec["page_id"],
                          "rows": [j * N_ROWS, j * N_ROWS + N_ROWS - 1]})
        stats.append({"page_id": rec["page_id"], "lines": len(boxes), "rows": len(rows),
                      "tiles": len(tiles), "size": [img.width, img.height]})
        if debug:
            print(f"  page {rec['page_id']:4d}: 检测行 {len(boxes):3d} -> 行图 {len(rows):3d}"
                  f" -> tile {len(tiles):3d}")
    (out / "tiles.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in index) + "\n", encoding="utf-8")
    (out / "detect_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    n_lines = [s["lines"] for s in stats]
    n_tiles = [s["tiles"] for s in stats]
    print(f"行检测: {len(stats)} 页 | 每页行数 中位 {int(np.median(n_lines))} "
          f"(min {min(n_lines)}, max {max(n_lines)})")
    print(f"       每页 tile 中位 {int(np.median(n_tiles))} (min {min(n_tiles)}, max {max(n_tiles)})"
          f" | tile 总数 {sum(n_tiles)}")
    print(f"       行数为 0 的页: {sum(1 for x in n_lines if x == 0)}"
          f"（>0 说明该页检测失败：整页无文字或纯图）")
    return {"stats": stats, "tiles": index}


# --------------------------------------------------------------------------- #
# ④ 评测
# --------------------------------------------------------------------------- #
def load_scorers():
    """Load the scorer modules packaged alongside this entry point."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import scorers

    return scorers


def main() -> int:
    global ROW_H
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="real_docs")
    ap.add_argument("--work", default="results/real_doc")
    ap.add_argument("--stage", default="all", choices=["tiles", "eval", "all"])
    ap.add_argument("--models", default="BM25")
    ap.add_argument("--row-height", type=int, default=None,
                    help="行高（像素）；默认取环境变量 RAVT_BLOCK_HEIGHT，未设则 44。"
                         "拼 tile 与建模型必须用同一个值")
    ap.add_argument("--clip-path", default="clip_local_weights_224")
    ap.add_argument("--ckpt", nargs="+", default=None,
                    help="一个或多个 RAVT checkpoint（可把原版与适配版一起报，"
                         "例如 --ckpt checkpoints/full_p14_s0.pth checkpoints/deg_adapt_p14_s0.pth）")
    ap.add_argument("--patch-size", type=int, default=14)
    ap.add_argument("--ocr-backend", default="tesseract")
    ap.add_argument("--degrade", type=float, default=0.0,
                    help="先给页面图施加扫描类退化再检测行（0=干净）。用的是与训练/合成轴"
                         "同一套图像算子（gaussian_blur/jpeg/rotation/low_resolution/shadow），"
                         "覆盖率固定 1.0（每一页都被退化），强度=该值")
    ap.add_argument("--limit-pages", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if args.row_height:
        ROW_H = int(args.row_height)
        print(f"行高: {ROW_H}px（mosaic 224×{N_ROWS * ROW_H}）")

    docs = Path(args.docs)
    work = Path(args.work)

    # 页面级退化（可选）：把 pages/ 过一遍退化引擎再写回 work/docs_deg/，
    # 之后行检测、OCR、评测都基于退化后的页面。这样 RAVT / OCR 流水线 / VLM
    # 三条路径看到的是同一批退化像素，和合成轴用的是同一套算子。
    if args.degrade and args.degrade > 0:
        import random as _random
        import shutil as _shutil
        from model_defs import CorruptionEngine
        # 只取**光度**算子：rotation/perspective 这类几何算子会把行检测本身打坏
        # （实测：加 rotation 后 0.2 档的 OCR 漂移就有 50%），那是另一条已知失效路径，
        # 不该混进"扫描质量变差"这条轴。
        page_ops = ["gaussian_blur", "jpeg", "low_resolution", "shadow"]
        deg_root = work / "docs_deg"
        (deg_root / "pages").mkdir(parents=True, exist_ok=True)
        engine = CorruptionEngine(_random.Random(20261013))
        meta = [json.loads(l) for l in (docs / "pages.jsonl").read_text(
            encoding="utf-8").splitlines() if l.strip()]
        if args.limit_pages:
            meta = meta[:args.limit_pages]
        for r in meta:
            dst = deg_root / "pages" / r["file"]
            if not dst.exists():
                with Image.open(docs / "pages" / r["file"]) as im:
                    out = engine.apply_image(im.convert("L"), float(args.degrade),
                                             page_ops, coverage=1.0)
                out.save(dst)
        (deg_root / "pages.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in meta) + "\n",
            encoding="utf-8")
        _shutil.copyfile(docs / "queries.jsonl", deg_root / "queries.jsonl")
        print(f"页面退化 severity={args.degrade:g}（覆盖率 1.0，算子 {page_ops}）"
              f" -> {deg_root}/pages（{len(meta)} 张）")
        docs = deg_root

    if args.stage in ("tiles", "all"):
        print(f"=== ① 行检测 + 拼 mosaic（{docs}）===")
        build_tiles(docs, work, args.limit_pages, args.debug)
        if args.stage == "tiles":
            return 0

    tiles = [json.loads(l) for l in (work / "tiles.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    queries = [json.loads(l) for l in (docs / "queries.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    pages = sorted({t["page_id"] for t in tiles})
    print(f"=== ② 评测：{len(queries)} 个真实查询 × {len(pages)} 张真实页面"
          f"（{len(tiles)} 个 tile）===")

    scorers = load_scorers()

    class TileTask:
        """把**全部 tile** 当成一个 task 的块（全局索引），页级分数 = 该页 tile 的最大分。

        这样文本基线的 idf 是在全局语料上算的（按页分别建索引会因空文档崩溃，
        也不符合真实检索的设定），而 RAVT 与文本基线面对的是同一批 tile。
        """

        def __init__(self, paths: list[Path], page_ids: list[int], cache_name: str):
            self.image_paths = paths
            self.page_ids = page_ids
            self.meta = {"task_id": 0, "block_count": len(paths)}
            self.query = ""
            self.path = work / cache_name

        @property
        def blocks(self):
            return len(self.image_paths)

        def load_images(self):
            out = []
            for p in self.image_paths:
                with Image.open(p) as im:
                    out.append(im.convert("RGB"))
            return out

        def load_texts(self):
            out = []
            for p in self.image_paths:
                f = p.with_suffix(".txt")
                out.append(f.read_text(encoding="utf-8") if f.exists() else "")
            return out

    tile_list = sorted(tiles, key=lambda t: (t["page_id"], t["tile"]))
    paths = [work / "tiles" / t["tile"] for t in tile_list]
    page_of = [int(t["page_id"]) for t in tile_list]
    task = TileTask(paths, page_of, "all_tiles")
    page_set = sorted(set(page_of))

    wanted = [m.strip().upper() for m in args.models.split(",")]
    want = lambda k: ("ALL" in wanted) or (k in wanted)  # noqa: E731
    results = {}

    def evaluate(name, scorer):
        scorer.prepare()
        docs_enc = scorer.encode_documents(task)
        if isinstance(docs_enc, list) and len(docs_enc) == 1 and isinstance(docs_enc[0], list):
            docs_enc = docs_enc[0]
        per_query = []
        for q in queries:
            s, _ = scorer.score_query(q["query"], docs_enc, task)
            s = np.asarray(s, dtype=np.float64).ravel()
            scores = {}
            for p in page_set:
                vals = s[[i for i, pp in enumerate(page_of) if pp == p]]
                scores[p] = float(np.max(vals)) if vals.size else -1e9
            ranked = sorted(scores, key=lambda p: -scores[p])
            rel = set(q["relevant_pages"])
            hit = [1 if p in rel else 0 for p in ranked]
            rec = {f"recall@{k}": float(any(hit[:k])) for k in (1, 5, 10)}
            for k in (5, 10):
                dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hit[:k]))
                idcg = sum(1 / math.log2(i + 2) for i in range(min(len(rel), k)))
                rec[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0
            rec["rank_of_first_relevant"] = next((i + 1 for i, h in enumerate(hit) if h), 0)
            per_query.append(rec)
        agg = {k: float(np.mean([r[k] for r in per_query])) * 100
               for k in ("recall@1", "recall@5", "recall@10", "ndcg@5", "ndcg@10")}
        print(f"\n{name:<28}" + "".join(f"{k}={v:5.1f}%  " for k, v in agg.items()))
        results[name] = {"agg": agg, "per_query": per_query}
        return scorer

    if want("RAVT") and args.ckpt:
        from types import SimpleNamespace
        for ck in args.ckpt:
            if not Path(ck).exists():
                print(f"（跳过 {ck}：文件不存在）")
                continue
            label = "RAVT-" + Path(ck).stem
            spec = SimpleNamespace(name=label, kind="ravt", checkpoint=ck,
                                   patch_size=args.patch_size, n_rows=5, row_height=ROW_H,
                                   aggregation="max_row", batch_size=64, model_id="",
                                   pooling="cls", tokenizer="word", dtype="float32",
                                   needs_gpu=True,
                                   extra={"clip_path": args.clip_path},
                                   to_dict=lambda: {})
            try:
                evaluate(label, scorers.build_scorer(spec, device=args.device))
            except Exception as exc:
                print(f"（{label} 失败：{type(exc).__name__}: {exc}）")

    # ---- 页面级 VLM 检索器：直接吃整页图（它们本来就是为整页设计的），
    #      与 RAVT 的"行切分 -> tile 打分 -> 页面取 max"是两条不同路径，
    #      但面对的是同一批页面、同一批查询、同一份 qrels。
    if want("COLPALI") or want("COLQWEN2"):
        from types import SimpleNamespace
        meta = [json.loads(l) for l in (docs / "pages.jsonl").read_text(
            encoding="utf-8").splitlines() if l.strip()]
        vpaths = [docs / "pages" / r["file"] for r in meta]
        vtask = TileTask(vpaths, [int(r["page_id"]) for r in meta], "pages")
        specs = []
        if want("COLPALI"):
            base = Path(os.environ.get("COLPALI_BASE", "colpali_offline_weights/base_model"))
            adapter = Path(os.environ.get("COLPALI_ADAPTER",
                                          "colpali_offline_weights/adapter_model"))
            if base.exists() and adapter.exists():
                specs.append(SimpleNamespace(
                    name="ColPali", kind="image_encoder", model_id="vidore/colpali-v1.2",
                    batch_size=4, dtype="bfloat16",
                    extra={"base_model_dir": str(base), "adapter_dir": str(adapter)},
                    checkpoint=None, patch_size=args.patch_size, n_rows=5, row_height=ROW_H,
                    pooling="cls", aggregation="max_row", tokenizer="word", needs_gpu=True,
                    to_dict=lambda: {}))
            else:
                print(f"（跳过 ColPali：找不到 {base} / {adapter}）")
        if want("COLQWEN2"):
            try:
                from ravt_paths import find_weight_dir
                cb, ca = find_weight_dir("colqwen2-base"), find_weight_dir("colqwen2")
            except Exception:
                cb, ca = Path("models_local/colqwen2-base"), Path("models_local/colqwen2")
            if cb.exists() and ca.exists():
                specs.append(SimpleNamespace(
                    name="ColQwen2", kind="image_encoder", model_id="vidore/colqwen2-v1.0",
                    batch_size=1, dtype="bfloat16",
                    extra={"family": "colqwen2", "local_dir": str(cb),
                           "adapter_dir": str(ca), "attn": "sdpa"},
                    checkpoint=None, patch_size=args.patch_size, n_rows=5, row_height=ROW_H,
                    pooling="cls", aggregation="max_row", tokenizer="word", needs_gpu=True,
                    to_dict=lambda: {}))
            else:
                print(f"（跳过 ColQwen2：找不到 {cb} / {ca}）")
        for spec in specs:
            try:
                evaluate(spec.name, scorers.build_scorer(spec, device=args.device))
            except Exception as exc:
                print(f"（{spec.name} 失败：{type(exc).__name__}: {exc}）")

    if want("BM25") or want("BGE"):
        engine = scorers.OcrEngine(cache_dir=str(work / "ocr_cache"), backend=args.ocr_backend)
        texts, cer = engine.texts_for_task(task)
        for path, txt in zip(paths, texts):
            path.with_suffix(".txt").write_text(txt, encoding="utf-8")
        print(f"\n（已把 {len(tiles)} 个 tile 的 OCR 文本写到 {work}/tiles/*.txt）")
        if want("BM25"):
            for tok in ("word", "char3"):
                evaluate(f"BM25-{tok}+OCR",
                         scorers.Bm25Scorer(f"BM25-{tok}+OCR", tokenizer=tok))
        if want("BGE"):
            from types import SimpleNamespace
            # 本地权重优先（服务器通常已下好 models_local/bge-m3，离线也能跑）
            try:
                from ravt_paths import find_weight_dir
                bge_dir = find_weight_dir("bge-m3")
            except Exception:
                bge_dir = Path("models_local/bge-m3")
            bge_id = str(bge_dir) if bge_dir.exists() and any(bge_dir.iterdir()) \
                else "BAAI/bge-m3"
            print(f"（BGE-M3 权重：{bge_id}）")
            spec = SimpleNamespace(name="BGE-M3+OCR", kind="text_encoder", model_id=bge_id,
                                   checkpoint=None, pooling="cls", batch_size=32,
                                   extra={"encoder_class": "auto"}, to_dict=lambda: {})
            try:
                evaluate("BGE-M3+OCR", scorers.build_scorer(spec, device=args.device))
            except Exception as exc:
                print(f"（BGE-M3 加载失败，跳过：{type(exc).__name__}: {exc}）")

    out_json = work / "real_doc_results.json"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果 -> {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
