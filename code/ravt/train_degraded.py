"""退化感知训练（TMM 版主模型的来源）。

与原论文主模型的唯一区别：训练时除了**字符级**课程，还开一条**成像级**课程
（模糊/JPEG/低分辨率/旋转/光照，可选真实扫描件族：透视/折痕/透印/摩尔纹/
复印/运动模糊/椒盐/非均匀光照）。动机是审稿人一定会问的那句：
"你们的训练分布里根本没有主要的测试条件（成像退化）"。

预置的 run（`--runs`）：
    deg       主模型：字符课程 + 温和成像退化（5 族），强度上限 0.35，线性课程
    degreal   真实扫描件版：13 族成像退化，强度上限 0.25
    degcov    覆盖率/强度解耦消融：固定覆盖率 0.5，强度上限 0.4
    degflat   无课程消融：一上来就是满强度
    charonly  对照组：只加字符噪声（等价于原论文主模型，image_noise=False）

用法（服务器）:
    python train_degraded.py --plan                       # 只打印计划与预估耗时
    python train_degraded.py --runs deg --seeds 0 1 2     # 跑 3 个种子的退化感知主模型
    python train_degraded.py --runs all --seeds 0         # 跑全部 5 个 run
    python train_degraded.py --runs deg --seeds 0 --dry-run   # 冒烟（随机小 CLIP）

产物：
    checkpoints/deg_p14_s0.pth / degreal_p14_s0.pth / ...
    日志 results/train/<run>.jsonl（含逐轮 noise_rate / image_severity，方便画课程曲线）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import TrainSpec            # noqa: E402
from stage_train import train_one       # noqa: E402

MILD_IMAGE = ("gaussian_blur", "jpeg", "rotation", "low_resolution", "shadow")
REAL_IMAGE = ("perspective", "fold", "show_through", "moire",
              "photocopy", "motion_blur", "salt_pepper", "uneven_light")
ALL_IMAGE = MILD_IMAGE + REAL_IMAGE

#: run 名 -> (说明, TrainSpec 覆盖项)
RUNS: dict[str, tuple[str, dict]] = {
    "deg": ("字符课程 + 温和成像退化（主模型）", dict(
        image_noise=True, image_kinds=MILD_IMAGE,
        image_max_severity=0.35, image_curriculum="linear")),
    "degreal": ("再加真实扫描件族（透视/折痕/透印/摩尔纹/复印/椒盐/光照）", dict(
        image_noise=True, image_kinds=ALL_IMAGE,
        image_max_severity=0.25, image_curriculum="linear")),
    "degcov": ("覆盖率与强度解耦：固定覆盖率 0.5", dict(
        image_noise=True, image_kinds=MILD_IMAGE,
        image_max_severity=0.40, image_coverage=0.5, image_curriculum="linear")),
    "degflat": ("无成像课程：从第 0 轮就是满强度", dict(
        image_noise=True, image_kinds=MILD_IMAGE,
        image_max_severity=0.35, image_curriculum="constant")),
    "charonly": ("对照组：只有字符噪声（= 原论文主模型）", dict(
        image_noise=False)),
}


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


def build_specs(args) -> list[TrainSpec]:
    names = list(RUNS) if args.runs == "all" else \
        [r.strip() for r in args.runs.split(",") if r.strip()]
    unknown = [r for r in names if r not in RUNS]
    if unknown:
        raise SystemExit(f"未知的 run {unknown}；可选: {list(RUNS)}")

    specs: list[TrainSpec] = []
    for run in names:
        _, overrides = RUNS[run]
        for seed in args.seeds:
            specs.append(TrainSpec(
                name=f"{run}_p{args.patch_size}_s{seed}", seed=seed,
                patch_size=args.patch_size, epochs=args.epochs, row_height=args.row_height,
                batch_size=args.batch_size, lr_backbone=args.lr_backbone,
                lr_head=args.lr_head, max_noise=args.max_noise, **overrides))
    return specs


def main():
    ap = argparse.ArgumentParser(description="退化感知训练（TMM 版主模型）")
    ap.add_argument("--runs", default="deg",
                    help="all 或逗号分隔: " + ",".join(RUNS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--patch-size", type=int, default=14)
    ap.add_argument("--row-height", type=int,
                    default=int(os.environ.get("RAVT_BLOCK_HEIGHT", 44)),
                    help="行高（像素）；44=默认，42=与 patch 14 精确对齐")
    ap.add_argument("--lr-backbone", type=float, default=1e-7)
    ap.add_argument("--lr-head", type=float, default=3e-5)
    ap.add_argument("--max-noise", type=float, default=0.30,
                    help="字符错误率上限（与原论文一致；成像强度另有自己的上限）")
    ap.add_argument("--clip-path", default="clip_local_weights_224")
    ap.add_argument("--train-blocks", default="data/processed/train_blocks")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--log-dir", default="results/train")
    ap.add_argument("--device", default=None)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--max-blocks", type=int, default=None)
    ap.add_argument("--init-from", default=None,
                    help="从已有 checkpoint 热启动（适配协议）：例如 "
                         "checkpoints/full_p14_s0.pth。不传就是从 CLIP backbone 从头训。"
                         "适配版建议 --epochs 5。")
    ap.add_argument("--run-tag", default="",
                    help="run 名里插一个标签（如 adapt -> deg_adapt_p14_s0.pth），"
                         "避免适配版覆盖完整版权重。默认跟随 --init-from 自动设为 adapt。")
    ap.add_argument("--hours-per-epoch", type=float, default=None,
                    help="用于 --plan 的耗时预估（按实测填，例如 4.0 表示每轮 4 小时）")
    ap.add_argument("--plan", action="store_true", help="只打印计划，不训练")
    ap.add_argument("--random-clip", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="随机小 CLIP 冒烟：epochs=1 batch=4 max_steps=3")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--base-dir", default=None)
    args = ap.parse_args()

    from ravt_paths import setup
    setup(args.base_dir)

    if args.dry_run:
        args.random_clip = True
        args.epochs = 1
        args.batch_size = 4
        args.max_steps = args.max_steps or 3
        args.max_blocks = args.max_blocks or 64

    smoke = args.random_clip
    prefix = "dryrun_" if smoke else ""
    if smoke:
        if args.ckpt_dir == "checkpoints":
            args.ckpt_dir = "checkpoints_dryrun"
        if args.log_dir == "results/train":
            args.log_dir = "results/train_dryrun"

    specs = build_specs(args)
    tag_part = args.run_tag or ("adapt" if args.init_from else "")
    if tag_part:
        for s in specs:
            head, _, tail = s.name.partition("_p")
            s.name = f"{head}_{tag_part}_p{tail}"
    if args.init_from:
        for s in specs:
            s.extra["init_from"] = args.init_from
        print(f"⚙️  热启动自 {args.init_from}（退化感知适配协议，建议 --epochs 5）")
    print(f"退化感知训练计划：{len(specs)} 个 run")
    print(f"{'run':<22}{'seed':>5}{'字符上限':>10}{'成像上限':>10}{'覆盖率':>9}{'课程':>10}")
    for s in specs:
        cov = "-" if s.image_coverage is None else f"{s.image_coverage:.2f}"
        print(f"{s.name:<22}{s.seed:>5}{s.max_noise:>10.2f}"
              f"{(s.image_max_severity if s.image_max_severity is not None else (s.max_noise if s.image_noise else 0.0)):>10.2f}"
              f"{cov:>9}{s.image_curriculum if s.image_noise else '-':>10}")
    if args.plan or args.hours_per_epoch:
        per_run = args.hours_per_epoch * args.epochs if args.hours_per_epoch else None
        if per_run:
            print(f"\n预估：每 run {per_run:.1f} 小时（{args.hours_per_epoch} h/epoch × "
                  f"{args.epochs} epochs），合计 {per_run * len(specs):.1f} 小时")
        else:
            print("\n提示：加 --hours-per-epoch 4 可以按你的实测速度预估总耗时")
    if args.plan:
        return

    if args.random_clip:
        debug = dict(random_init=True, max_blocks=args.max_blocks or 64,
                     random_cfg=dict(image_size=224, patch_size=14, hidden_size=64,
                                     intermediate_size=128, num_hidden_layers=2,
                                     num_attention_heads=4, text_hidden_size=32,
                                     text_intermediate_size=64, text_num_hidden_layers=2,
                                     text_num_attention_heads=4, vocab_size=49408))
        for s in specs:
            s.extra.update(debug)
            s.epochs = min(s.epochs, 1)
            s.batch_size = min(s.batch_size, 4)
            s.name = prefix + s.name

    done, failed = [], []
    for s in specs:
        if not args.force and already_done(args.log_dir, s.name):
            print(f"⏭️  跳过 {s.name}（已完成；要重跑加 --force）")
            continue
        try:
            out = train_one(s, train_blocks=args.train_blocks, ckpt_dir=args.ckpt_dir,
                            clip_path=args.clip_path, device=args.device,
                            max_steps=args.max_steps, num_workers=args.num_workers,
                            log_dir=args.log_dir,
                            allow_cpu=bool(args.allow_cpu or args.random_clip))
            done.append(out)
        except Exception as exc:
            print(f"❌ {s.name} 失败: {type(exc).__name__}: {exc}")
            failed.append((s.name, str(exc)[:200]))

    print("\n" + "=" * 74)
    print(f"训练结束：成功 {len(done)}，失败 {len(failed)}")
    for d in done:
        print(f"  ✅ {d['run_id']:<28} -> {d['checkpoint']}")
    for name, err in failed:
        print(f"  ❌ {name:<28} {err}")
    print("=" * 74)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
