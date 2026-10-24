"""实验 5：噪声课程的敏感性——终点损坏率 0.15 / 0.30 / 0.45。

0.30 是论文的设置。这里回答"30% 是不是针对 Visual-NIAH 调出来的"：
从同一个已训练 checkpoint 热启动，各跑 5 个 epoch（只改 --max-noise），
然后在 40% 档评测，看排序是否稳定。

    python code/experiments/train_curriculum_endpoints.py --root . \
        --rates 0.15 0.45 --epochs 5 --limit 60

（0.30 的参考模型已经在 checkpoints/ 里，不必重跑。）
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--rates", nargs="+", default=["0.15", "0.45"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--train-blocks", default="data/processed/train_blocks_rh42")
    ap.add_argument("--init-from", default="checkpoints/full_p14_rh42_s0.pth")
    ap.add_argument("--row-height", type=int, default=42)
    ap.add_argument("--noise", type=float, default=0.4)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--eval-only", action="store_true",
                    help="??????????? rate*.pth????????????")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    for p in (root, root / "code" / "ravt",
              root / "code" / "ravt"):
        sys.path.insert(0, str(p))

    out = root / a.out
    out.mkdir(parents=True, exist_ok=True)
    ckpts = {"0.30 (reference)": root / a.init_from}
    for r in a.rates:
        tag = "rate" + r.replace(".", "")
        # train.py ?????? full_p14_<tag>_s0.pth??????????
        dst = next((c for c in (root / "checkpoints" / f"full_p14_{tag}_s0.pth",
                                root / "checkpoints" / f"full_p14_rh42_s0_{tag}.pth")
                    if c.exists()), root / "checkpoints" / f"full_p14_{tag}_s0.pth")
        if not dst.exists() and not a.eval_only:
            cmd = [a.python, "-u", str(root / "code" / "ravt" / "train.py"),
                   "--train-blocks", a.train_blocks, "--ckpt-dir", "checkpoints",
                   "--row-height", str(a.row_height), "--init-from", a.init_from,
                   "--epochs", str(a.epochs), "--max-noise", r, "--tag", tag,
                   "--seeds", "0", "--num-workers", "4"]
            print("$", " ".join(cmd))
            subprocess.run(cmd, cwd=str(root), check=False)
        ckpts[r] = dst if dst.exists() else None
        print(f"[{r}] checkpoint -> {ckpts[r]}")

    import numpy as np
    from datasets_eval import list_tasks
    from scorers import RavtScorer

    tasks = list_tasks(root / "data" / "test_bench_tmm_rh42",
                       noise_levels=[a.noise], max_tasks=a.limit)
    rows = []
    for name, ck in ckpts.items():
        if ck is None:
            continue
        sc = RavtScorer("RAVT-" + name, str(root / "clip_local_weights_224"), str(ck),
                        row_height=a.row_height)
        sc.prepare()
        hit = n = 0
        for t in tasks:
            docs = sc.encode_documents(t)
            s, _ = sc.score_query(t.query, docs, task=t)
            if int(np.argmax(s)) == t.gt_block:
                hit += 1
            n += 1
        rows.append({"curriculum": name, "top1": 100 * hit / max(n, 1), "n": n})
        print(f"{name:<20} top-1 = {rows[-1]['top1']:.1f}%  (n={n}, noise={a.noise})")

    lines = [f"| 课程终点损坏率 | top-1（noise={a.noise}） | n |", "|---|---|---|"]
    lines += [f"| {r['curriculum']} | {r['top1']:.1f}% | {r['n']} |" for r in rows]
    body = ("# 实验 5：噪声课程敏感性\n\n"
            f"从同一 checkpoint 热启动 {a.epochs} 个 epoch，只改 --max-noise。\n\n"
            + "\n".join(lines) + "\n")
    (out / "curriculum.md").write_text(body, encoding="utf-8")
    (out / "curriculum.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    print(f"\n写出 {out}/curriculum.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
