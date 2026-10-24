"""实验 4：噪声课程的终点损坏率敏感性 —— 只评测，不训练。

论文的课程调度是 r = 0.30 * e/E。这里已有三个终点（0.15 / 0.30 / 0.45）的训练权重，
直接在同一批 task 上对比，回答审稿人“30% 是普适最优还是针对本测试集调出来的”。

    python code/experiments/evaluate_curriculum_endpoints.py --root . \
        --noise-levels 0.0 0.4 --limit 60
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

#: 终点损坏率 -> 可能的 checkpoint 文件名（按顺序探测）
CKPT_CANDIDATES = {
    "0.30 (reference)": ("full_p14_rh42_s0.pth", "full_p14_rate030_s0.pth",
                         "full_p14_rh42_rate030_s0.pth"),
    "0.15": ("full_p14_rate015_s0.pth", "full_p14_rh42_rate015_s0.pth",
             "rate015_s0.pth"),
    "0.45": ("full_p14_rate045_s0.pth", "full_p14_rh42_rate045_s0.pth",
             "rate045_s0.pth"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--bench", default="data/test_bench_tmm_rh42")
    ap.add_argument("--clip", default="clip_local_weights_224")
    ap.add_argument("--row-height", type=int, default=42)
    ap.add_argument("--noise-levels", nargs="+", type=float, default=[0.0, 0.4])
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    for p in (root, root / "code" / "ravt"):
        sys.path.insert(0, str(p))

    import numpy as np
    from datasets_eval import list_tasks
    from scorers import RavtScorer

    out = root / a.out
    out.mkdir(parents=True, exist_ok=True)
    ckpt_root = root / "checkpoints"
    ckpts, missing = {}, []
    for tag, names in CKPT_CANDIDATES.items():
        hit = next((ckpt_root / n for n in names if (ckpt_root / n).exists()), None)
        ckpts[tag] = hit
        if hit is None:
            missing.append(tag)
        print(f"[{tag}] {hit if hit else '缺 checkpoint'}")
    if missing:
        print(f"⚠️ 缺 {missing} 的权重：这几行会被跳过（结果表里不出现，不影响其他阶段）")
    if not any(ckpts.values()):
        print("没有任何 checkpoint，退出")
        return 1

    levels = sorted(set(round(float(x), 4) for x in a.noise_levels))
    grid = {}
    for lv in levels:
        tasks = list_tasks(root / a.bench, noise_levels=[lv], max_tasks=a.limit)
        print(f"\nnoise={lv}: {len(tasks)} 个 task")
        if not tasks:
            continue
        for tag, ck in ckpts.items():
            if ck is None:
                continue
            sc = RavtScorer(f"RAVT-{tag}", str(root / a.clip), str(ck),
                            row_height=a.row_height)
            sc.prepare()
            hit = n = 0
            for t in tasks:
                docs = sc.encode_documents(t)
                s, _ = sc.score_query(t.query, docs, task=t)
                if int(np.argmax(s)) == int(t.gt_block):
                    hit += 1
                n += 1
            sc.close()
            grid[(tag, lv)] = {"top1": 100.0 * hit / max(n, 1), "n": n,
                               "ckpt": ck.name}
            print(f"  {tag:<18} top-1 = {grid[(tag, lv)]['top1']:5.1f}%   (n={n})")

    tags = [t for t in CKPT_CANDIDATES if ckpts[t] is not None]
    lines = ["| 课程终点损坏率 | " + " | ".join(f"noise={lv}" for lv in levels) + " |",
             "|" + "---|" * (len(levels) + 1)]
    for tag in tags:
        cells = []
        for lv in levels:
            g = grid.get((tag, lv))
            cells.append(f"{g['top1']:.1f}%" if g else "-")
        lines.append(f"| {tag} | " + " | ".join(cells) + " |")
    body = ("# 噪声课程：终点损坏率敏感性\n\n"
            f"同一批 task（每档 {a.limit} 个），只换课程终点 r_max；行高 {a.row_height}px。\n\n"
            + "\n".join(lines) +
            "\n\n读法：若三行在同一档内基本持平，说明结论不依赖 30% 这个具体取值；"
            "若 0.45 在干净档掉、在 0.40 档涨，则存在“以干净精度换退化鲁棒性”的取舍。\n")
    (out / "curriculum.md").write_text(body, encoding="utf-8")
    (out / "curriculum.json").write_text(
        json.dumps([{"curriculum": k[0], "noise": k[1], **v} for k, v in grid.items()],
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n" + body)
    print(f"写出 {out}/curriculum.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
