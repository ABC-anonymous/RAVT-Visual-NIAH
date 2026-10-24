"""实验 1+2：行几何 —— 精确对齐有多重要、需要几个行向量。

两个问题一次算完，全部在推理端，不训练：

  A. 行偏移探针：把同一块 mosaic 整体上/下平移 3 或 7 像素（行窗口因此切错位置），
     看检索掉多少 —— 回答“对齐收益是不是只对固定模板成立”。
  B. 行向量数 k：把同一个 checkpoint 载入 n_rows = 1/2/3/5/10 的模型
     （只有 patch→token 的分组粒度变了，权重全部兼容），行高设为 210/k，
     让 k 个窗口正好铺满 5x42 = 210 px 的文本区，给出准确率—索引—延迟权衡
     —— 回答“五个行向量是不是必要”。

    python code/experiments/row_geometry.py --root . \
        --noise 0.4 --limit 60

冒烟（一两分钟）：
    python code/experiments/row_geometry.py --root ... --limit 2 --max-blocks 40 \
        --variants "as-is,k=1"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

#: (标签, 纵向平移像素, 行向量数, 行高像素)
#: 行高 = 210 / k，k 个窗口正好铺满 5x42=210px 的文本区（画布 224px，底部 14px 白边）。
VARIANTS = [
    ("as-is (n=5, 42px)", 0, 5, 42),
    ("offset +3px", 3, 5, 42),
    ("offset -3px", -3, 5, 42),
    ("offset +7px", 7, 5, 42),
    ("offset -7px", -7, 5, 42),
    ("k=3 (70px)", 0, 3, 70),
    ("k=2 (105px)", 0, 2, 105),
    ("k=1 (210px)", 0, 1, 210),
    ("k=10 (21px)", 0, 10, 21),
]


def shift(im, dy: int):
    """把图像内容整体平移 dy 像素（空出来的部分填白），模拟行窗口切错位置。"""
    band = len(im.getbands())
    canvas = im.copy()
    canvas.paste((255,) * band, (0, 0, im.width, im.height))
    if dy >= 0:
        canvas.paste(im.crop((0, 0, im.width, im.height - dy)), (0, dy))
    else:
        canvas.paste(im.crop((0, -dy, im.width, im.height)), (0, 0))
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--bench", default="data/test_bench_tmm_rh42")
    ap.add_argument("--ckpt", default="checkpoints/full_p14_rh42_s0.pth")
    ap.add_argument("--clip", default="clip_local_weights_224")
    ap.add_argument("--patch-size", type=int, default=14)
    ap.add_argument("--noise", type=float, default=0.4)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--max-blocks", type=int, default=0,
                    help="每个 task 最多保留多少块（0=不限；冒烟用）")
    ap.add_argument("--variants", default="",
                    help="只跑标签以这些前缀开头的变体（逗号分隔），例如 'as-is,k=1'")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    for p in (root, root / "code" / "ravt"):
        sys.path.insert(0, str(p))

    import numpy as np
    import torch
    from datasets_eval import list_tasks
    from model_defs import RAVTConfig, RAVTModel
    from transformers import CLIPProcessor, CLIPTokenizer

    out = root / a.out
    out.mkdir(parents=True, exist_ok=True)
    tasks = list_tasks(root / a.bench, noise_levels=[a.noise], max_tasks=a.limit)
    if not tasks:
        print("没有 task：检查 --bench / --noise")
        return 1
    print(f"task 数 {len(tasks)}（noise={a.noise}，max_blocks={a.max_blocks or '不限'}）")

    # 图片只读一次，所有变体共用；需要截断时保留目标块与干扰块，并重排索引
    prepared = []
    for t in tasks:
        imgs = t.load_images()
        gt = int(t.gt_block)
        dis = int(t.distractor_block) if t.has_distractor else -1
        if a.max_blocks and len(imgs) > a.max_blocks:
            idx = sorted(set(range(a.max_blocks)) | {gt} | ({dis} if dis >= 0 else set()))
            pos = {o: i for i, o in enumerate(idx)}
            imgs = [imgs[i] for i in idx]
            gt, dis = pos[gt], pos.get(dis, -1)
        prepared.append((t.query, imgs, gt, dis))
    n_blocks = sum(len(p[1]) for p in prepared)
    print(f"共 {n_blocks} 块图")

    clip_dir = str(root / a.clip)
    ckpt = str(root / a.ckpt)
    proc = CLIPProcessor.from_pretrained(clip_dir)
    tok = CLIPTokenizer.from_pretrained(clip_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    slot: dict = {"key": None, "model": None}

    def get_model(n_rows: int, row_height: int):
        key = (n_rows, row_height)
        if slot["key"] != key:
            if slot["model"] is not None:                 # 只留一个模型，省显存
                del slot["model"]
                slot["model"] = None
                if device == "cuda":
                    torch.cuda.empty_cache()
            m = RAVTModel(RAVTConfig(clip_path=clip_dir, patch_size=a.patch_size,
                                     n_rows=n_rows, row_height=row_height))
            m = m.to(device).eval()
            state = torch.load(ckpt, map_location=device)
            state = {k.replace("module.", ""): v for k, v in state.items()}
            miss, unexp = m.load_state_dict(state, strict=False)
            print(f"  n_rows={n_rows} 行高={row_height}px: 缺失 {len(miss)} / 多余 {len(unexp)} 参数")
            warn = m.patch_mapping_warning()
            if warn:
                print(f"    {warn}")
            slot["key"], slot["model"] = key, m
        return slot["model"]

    def encode(imgs, model):
        feats, t0 = [], time.perf_counter()
        with torch.no_grad():
            for i in range(0, len(imgs), 32):
                px = proc(images=imgs[i:i + 32], return_tensors="pt")["pixel_values"].to(device)
                feats.append(model.encode_vision(px).cpu().numpy())
        ms = (time.perf_counter() - t0) / max(len(imgs), 1) * 1000.0
        return np.concatenate(feats, axis=0).astype(np.float32), ms

    def qvec(text, model):
        enc = tok(text, padding="max_length", max_length=77, truncation=True,
                  return_tensors="pt").to(device)
        with torch.no_grad():
            return model.encode_text(enc["input_ids"], enc["attention_mask"])[0]

    def block_scores(docs, q):
        if docs.ndim == 2:                       # 全局池化分支
            return np.asarray(docs @ q.detach().cpu().numpy())
        sims = torch.einsum("bnd,d->bn", torch.from_numpy(docs).to(device), q)
        return sims.max(dim=-1).values.cpu().numpy()

    want = [v.strip().lower() for v in a.variants.split(",") if v.strip()]
    sel = [v for v in VARIANTS if not want or any(v[0].lower().startswith(w) for w in want)]
    if not sel:
        print(f"--variants 没匹配到任何变体：{a.variants}")
        return 1

    rows = []
    for tag, dy, n_rows, rh in sel:
        model = get_model(n_rows, rh)
        hit = mrr = dis_err = 0
        ms_list = []
        dim = None
        for query, imgs, gt, dis in prepared:
            imgs_v = [shift(im, dy) for im in imgs] if dy else imgs
            docs, ms = encode(imgs_v, model)
            ms_list.append(ms)
            dim = docs.shape[-1]
            s = block_scores(docs, qvec(query, model))
            order = np.argsort(-s)
            if order[0] == gt:
                hit += 1
            if dis >= 0 and dis != gt and order[0] == dis:
                dis_err += 1
            mrr += 1.0 / (int(np.where(order == gt)[0][0]) + 1)
        n = len(prepared)
        kb = n_rows * dim * 4 / 1024.0 if dim else float("nan")
        row = {"variant": tag, "dy_px": dy, "n_rows": n_rows, "row_height": rh,
               "top1": 100.0 * hit / max(n, 1), "mrr": mrr / max(n, 1),
               "distractor_err": 100.0 * dis_err / max(n, 1),
               "kb_per_block": kb, "encode_ms_per_block": float(np.median(ms_list)),
               "n_tasks": n, "n_blocks": n_blocks}
        rows.append(row)
        print(f"  {tag:<20} top-1 {row['top1']:5.1f}%  干扰误命中 {row['distractor_err']:5.1f}%  "
              f"MRR {row['mrr']:.3f}  {kb:6.1f} KB/块  {np.median(ms_list):5.2f} ms/块")

    align = [r for r in rows if r["n_rows"] == 5]
    krows = [r for r in rows if r["dy_px"] == 0]

    def table(rs):
        lines = ["| 变体 | top-1 | 干扰误命中 | MRR | KB/块 | ms/块 |", "|---|---|---|---|---|---|"]
        lines += [f"| {r['variant']} | {r['top1']:.1f}% | {r['distractor_err']:.1f}% | "
                  f"{r['mrr']:.3f} | {r['kb_per_block']:.1f} | {r['encode_ms_per_block']:.2f} |"
                  for r in rs]
        return lines

    check = ""
    if abs(a.noise - 0.4) < 1e-6 and a.limit == 60 and not a.max_blocks:
        as_is = next((r for r in rows if r["variant"].startswith("as-is")), None)
        if as_is:
            check = (f"\n自检：`as-is` 的 top-1 = {as_is['top1']:.1f}%，"
                     "应与第三批 max_row 的 68.3%（同 60 个 task、noise=0.4）一致。\n")

    body = ("# 行几何：精确对齐与行向量数\n\n"
            f"noise={a.noise}，{len(prepared)} 个 task / {n_blocks} 块，checkpoint={a.ckpt}\n"
            f"{check}\n"
            "## A. 行偏移（n_rows=5, 42px）\n\n" + "\n".join(table(align)) +
            "\n\n把整块内容平移若干像素 = 行窗口切错位置（真实流水线里对应行检测框的垂直误差）。\n"
            "\n## B. 行向量数 k（行高 = 210/k，窗口正好铺满文本区）\n\n" + "\n".join(table(krows)) +
            "\n\nk=1 / k=2 全部沿用同一个 checkpoint，只是 patch→token 的分组粒度不同；\n"
            "k=1 的**训练版**对照是消融表里的 `w/o RAVT (global pooling)`。\n")
    (out / "row_geometry.md").write_text(body, encoding="utf-8")
    (out / "row_geometry.json").write_text(
        json.dumps({"noise": a.noise, "n_tasks": len(prepared), "n_blocks": n_blocks,
                    "checkpoint": a.ckpt, "rows": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print("\n" + body)
    print(f"写出 {out}/row_geometry.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
