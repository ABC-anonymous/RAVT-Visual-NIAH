"""实验 1：max-over-rows 到底失去了什么？——聚合方式替换 + 行排列探针。

不需要训练、不需要改服务器代码：他们的 `RavtScorer.encode_documents()` 直接给出
(blocks, n_rows, D) 的行向量，`RAVTModel.encode_text()` 给出查询向量，于是**一次编码**
就能同时算四种聚合：

    max_row      原论文：块分 = 5 行相似度的最大值
    mean_row     行向量平均后再点积（= 推理端的 global pooling）
    late_softmax 行级 MaxSim 的平滑版：softmax(sims/0.1) 加权求和（他们代码里已有）
    late_top2    只对最高的两行做加权（更接近"保留局部结构"的折中）

再加**行排列探针**：把目标块 mosaic 的 5 行逆序/交换后重新编码，看模型是否把"正确顺序"
排在"打乱顺序"之前（同一段文字、同一组行、只改顺序）。这一项直接回答审稿人的问题：
失效根源是 max 聚合，还是行 token 表示本身不含序列信息。

    python code/experiments/aggregation_probe.py --root . \
        --noise 0.4 --limit 60
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--bench", default="data/test_bench_tmm_rh42")
    ap.add_argument("--ckpt", default="checkpoints/full_p14_rh42_s0.pth")
    ap.add_argument("--clip", default="clip_local_weights_224")
    ap.add_argument("--row-height", type=int, default=42)
    ap.add_argument("--noise", type=float, default=0.4)
    ap.add_argument("--limit", type=int, default=60, help="抽多少个 task（固定种子）")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    for p in (root, root / "code" / "ravt",
              root / "code" / "ravt"):
        sys.path.insert(0, str(p))

    import numpy as np
    import torch
    from PIL import Image
    from datasets_eval import list_tasks
    from scorers import RavtScorer

    out = root / a.out
    out.mkdir(parents=True, exist_ok=True)
    tasks = list_tasks(root / a.bench, noise_levels=[a.noise], max_tasks=a.limit)
    if not tasks:
        print("❌ 该档位没有 task（检查 --bench/--noise）")
        return 1
    print(f"task 数 {len(tasks)}（noise={a.noise}，固定种子抽样）")

    sc = RavtScorer("RAVT-agg", str(root / a.clip), str(root / a.ckpt),
                    row_height=a.row_height)
    sc.prepare()
    rows_h = a.row_height

    def encode_images(imgs):
        feats = []
        with torch.no_grad():
            for i in range(0, len(imgs), 32):
                px = sc._processor(images=imgs[i:i + 32],
                                   return_tensors="pt")["pixel_values"].to(sc.device)
                feats.append(sc._model.encode_vision(px).cpu().numpy())
        return np.concatenate(feats, axis=0).astype(np.float32)

    def query_vec(text: str):
        enc = sc._tokenizer(text, padding="max_length", max_length=77,
                            truncation=True, return_tensors="pt").to(sc.device)
        with torch.no_grad():
            return sc._model.encode_text(enc["input_ids"], enc["attention_mask"])[0]

    def aggregate(sims: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "max_row":
            return sims.max(dim=-1).values
        if mode == "mean_row":
            return sims.mean(dim=-1)
        if mode == "late_softmax":
            w = torch.softmax(sims / 0.1, dim=-1)
            return (w * sims).sum(dim=-1)
        if mode == "late_top2":
            top = sims.topk(2, dim=-1).values
            return top.mean(dim=-1)
        raise ValueError(mode)

    MODES = ("max_row", "mean_row", "late_softmax", "late_top2")
    stat = {m: dict(top1=0, mrr=0.0, gap=0.0, perm_wins=0, swap_wins=0, n=0) for m in MODES}

    warned = False
    for t in tasks:
        imgs = t.load_images()
        if not imgs:
            if not warned:
                sub = [d.name for d in t.path.iterdir()] if t.path.is_dir() else []
                print(f"??  {t.path} ???????????? task??????{sub[:8]}")
                warned = True
            continue
        docs = sc.encode_documents(t)                # (B, R, D)
        gt = t.gt_block
        if gt >= len(imgs):
            continue
        q = query_vec(t.query)
        sims = torch.einsum("bnd,d->bn", torch.from_numpy(docs).to(sc.device), q).cpu()
        # ---- 1a：四种聚合的检索表现 ----
        for m in MODES:
            s = aggregate(sims.to(sc.device), m).cpu().numpy()
            order = np.argsort(-s)
            stat[m]["n"] += 1
            if order[0] == gt:
                stat[m]["top1"] += 1
            rank = int(np.where(order == gt)[0][0]) + 1
            stat[m]["mrr"] += 1.0 / rank
        # ---- 1b：行排列探针（同一段文字、只改 5 行的顺序）----
        base = imgs[gt]
        h = min(rows_h, base.height // 5)
        rows = [base.crop((0, i * h, base.width, (i + 1) * h)) for i in range(5)]
        variants = {"rev": rows[::-1], "swap": [rows[0], rows[3], rows[2], rows[1], rows[4]]}
        for name, order_rows in variants.items():
            canvas = Image.new(base.mode, base.size, (255,) * len(base.getbands()))
            for i, r in enumerate(order_rows):
                canvas.paste(r, (0, i * h))
            vec = encode_images([canvas])[0]
            sim_perm = torch.einsum("nd,d->n", torch.from_numpy(vec).to(sc.device), q).cpu()
            for m in MODES:
                s_true = float(aggregate(sims.to(sc.device), m)[gt].cpu())
                s_perm = float(aggregate(sim_perm.unsqueeze(0).to(sc.device), m)[0].cpu())
                stat[m]["gap"] += s_true - s_perm
                if s_perm > s_true:
                    stat[m][f"perm_wins" if name == "rev" else "swap_wins"] += 1

    lines = ["| 聚合方式 | top-1 命中 | MRR | 顺序间隔(真−逆序) | 逆序反超 | 交换反超 |",
             "|---|---|---|---|---|---|"]
    report = {}
    for m in MODES:
        d = stat[m]
        n = max(d["n"], 1)
        row = dict(top1=100 * d["top1"] / n, mrr=d["mrr"] / n, gap=d["gap"] / n,
                   perm_wins=d["perm_wins"], swap_wins=d["swap_wins"], n=n)
        report[m] = row
        lines.append(f"| {m} | {row['top1']:.1f}% | {row['mrr']:.3f} | "
                     f"{row['gap']:+.4f} | {row['perm_wins']} | {row['swap_wins']} |")
        print(f"[{m}] top1={row['top1']:.1f}%  mrr={row['mrr']:.3f}  "
              f"gap={row['gap']:+.4f}  逆序反超 {row['perm_wins']}/{n}  "
              f"交换反超 {row['swap_wins']}/{n}")
    body = ("# 实验 1：聚合方式与行序\n\n"
            f"noise={a.noise}，{len(tasks)} 个 task，checkpoint={a.ckpt}\n\n"
            + "\n".join(lines) +
            "\n\n读法：'顺序间隔' 是同一段文字在正确行序与逆序下的块分差；间隔≈0 说明该聚合"
            "对行序不敏感（逆序也照样高分），'逆序反超' 计数就是这种情形。\n")
    (out / "aggregation_probe.md").write_text(body, encoding="utf-8")
    (out / "aggregation_probe.json").write_text(
        json.dumps({"noise": a.noise, "tasks": len(tasks), "modes": report},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n写出 {out}/aggregation_probe.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
