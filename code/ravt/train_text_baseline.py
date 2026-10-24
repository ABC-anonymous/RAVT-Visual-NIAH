"""E2 归因实验：用**和 RAVT 完全相同的噪声课程**微调文本编码器（BGE-M3）。

为什么必须做这一条：现在论文能说明"渲染成图的视觉检索器比**未适配**的文本模型强"，
但审稿人会问——"提升到底来自渲染成图，还是来自加噪训练？"
本脚本把噪声课程原样搬到文本侧：同样的 homoglyph/gibberish/occlusion 课程、
同样的线性升温（0 -> max_noise）、同样的长度分桶负样本，只是不再渲染成图。

产物：checkpoints/text_bge_typo_s{seed}/（8 轮）
      checkpoints/text_bge_clean_s{seed}/（对照组，不加噪）
评测：evaluate.py 会自动扫描 checkpoints/text_* 并加入矩阵，
      名字是 TXT-text_bge_typo_s0（oracle 文本）与 TXT-text_bge_typo_s0+OCR[tesseract]。

用法:
    python train_text_baseline.py --seeds 0            # 两条 baseline（typo + clean）
    python train_text_baseline.py --seeds 0 1 2        # 多种子
    python train_text_baseline.py --plan               # 只打印计划
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import TrainSpec          # noqa: E402
from stage_train import train_one     # noqa: E402


def already_done(log_dir: str, run_id: str) -> bool:
    p = Path(log_dir) / f"{run_id}.jsonl"
    if not p.exists():
        return False
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip() and json.loads(line).get("record") == "run_summary":
                return True
    except Exception:
        return False
    return False


def pick_backbone(explicit: str | None) -> str:
    """优先用本地权重（离线可跑），否则回落到 HF 仓库名。"""
    if explicit:
        return explicit
    for cand in ("models_local/bge-m3", "../models_local/bge-m3"):
        if Path(cand).exists():
            return cand
    return "BAAI/bge-m3"


def main() -> int:
    ap = argparse.ArgumentParser(description="噪声课程微调的文本基线（E2 归因实验）")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--backbone", default=None, help="默认自动用 models_local/bge-m3")
    ap.add_argument("--train-blocks", default="data/processed/train_blocks")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--log-dir", default="results/train")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr-head", type=float, default=2e-5)
    ap.add_argument("--max-noise", type=float, default=0.30)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--max-blocks", type=int, default=None,
                    help="只用前 N 个训练块（冒烟用）")
    ap.add_argument("--device", default=None)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--base-dir", default=None)
    args = ap.parse_args()

    from ravt_paths import setup
    setup(args.base_dir, quiet=True)

    backbone = pick_backbone(args.backbone)
    specs = []
    for seed in args.seeds:
        for tag, use_noise in (("typo", True), ("clean", False)):
            specs.append(TrainSpec(
                name=f"text_bge_{tag}_s{seed}", kind="text_encoder", seed=seed,
                text_backbone=backbone, use_noise=use_noise, use_buckets=True,
                epochs=args.epochs, batch_size=args.batch_size,
                lr_head=args.lr_head, max_noise=args.max_noise,
                extra={"max_blocks": args.max_blocks} if args.max_blocks else {}))

    print(f"文本基线训练计划（{len(specs)} 个 run，backbone={backbone}）")
    print(f"{'run':<24}{'seed':>5}{'加噪课程':>10}{'epochs':>8}{'batch':>7}")
    for s in specs:
        print(f"{s.name:<24}{s.seed:>5}{('是' if s.use_noise else '否'):>10}"
              f"{s.epochs:>8}{s.batch_size:>7}")
    print("\n说明：typo = 加噪声课程（与 RAVT 相同）；clean = 不加噪的对照组。\n"
          "      两者跑完，配合 eval_all 就能回答：到底是渲染在起作用，还是训练在起作用。")
    if args.plan:
        return 0

    done, failed = [], []
    for s in specs:
        if not args.force and already_done(args.log_dir, s.name):
            print(f"⏭️  跳过 {s.name}（已完成）")
            continue
        try:
            out = train_one(s, train_blocks=args.train_blocks, ckpt_dir=args.ckpt_dir,
                            device=args.device, max_steps=args.max_steps,
                            num_workers=args.num_workers, log_dir=args.log_dir,
                            allow_cpu=args.allow_cpu)
            done.append(out)
        except Exception as exc:
            print(f"❌ {s.name} 失败: {type(exc).__name__}: {exc}")
            failed.append(s.name)

    print("\n" + "=" * 70)
    print(f"完成 {len(done)}，失败 {len(failed)}")
    for d in done:
        print(f"  ✅ {d['run_id']:<24} -> {d['checkpoint']}")
    for n in failed:
        print(f"  ❌ {n}")
    if done:
        print("\n下一步：评测（会自动把这些 text_* 目录加进矩阵）")
        print("  python evaluate.py --models all --benches main "
              "--bench-main data/test_bench_tmm --out-dir results/jsonl/eval_tmm")
    print("=" * 70)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
