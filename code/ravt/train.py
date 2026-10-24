"""训练 RAVT 主模型，并可顺带跑消融实验。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import TrainSpec, build_train_matrix   # noqa: E402
from stage_train import train_one                 # noqa: E402

ABLATIONS = ["no_fusion", "no_slicing", "no_gated_proj", "no_buckets", "no_noise",
             "no_local_loss", "no_margin_loss", "no_local_no_margin"]


def already_done(log_dir: str, run_id: str) -> bool:
    """该 run 是否已经跑完（日志里有 run_summary 记录）。"""
    p = Path(log_dir) / f"{run_id}.jsonl"
    if not p.exists():
        return False
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                if json.loads(line).get("record") == "run_summary":
                    return True
            except json.JSONDecodeError:
                continue
    except Exception:
        return False
    return False


def build_specs(args, prefix: str = "") -> list[TrainSpec]:
    """主模型 + 按需的消融配置。"""
    specs: list[TrainSpec] = []
    for seed in args.seeds:
        specs.append(TrainSpec(
            name=f"{prefix}full_p{args.patch_size}"
                 f"{('_' + args.tag) if args.tag else ''}_s{seed}", seed=seed,
            patch_size=args.patch_size, row_height=args.row_height,
            epochs=args.epochs, batch_size=args.batch_size,
            lr_backbone=args.lr_backbone, lr_head=args.lr_head,
            max_noise=args.max_noise,
            image_noise=args.image_noise,
        ))

    if args.ablation == "none":
        return specs

    wanted = ABLATIONS if args.ablation == "all" else \
        [a.strip() for a in args.ablation.split(",") if a.strip()]
    unknown = [a for a in wanted if a not in ABLATIONS]
    if unknown:
        raise SystemExit(f"未知的消融名 {unknown}；可选: {ABLATIONS}")

    flags = {
        "no_fusion": dict(use_fusion=False),
        "no_slicing": dict(use_slicing=False),
        "no_gated_proj": dict(use_gated_proj=False),
        "no_buckets": dict(use_buckets=False),
        "no_noise": dict(use_noise=False),
        "no_local_loss": dict(use_local_loss=False),      # 论文缺的消融
        "no_margin_loss": dict(use_margin_loss=False),    # 论文缺的消融
        "no_local_no_margin": dict(use_local_loss=False, use_margin_loss=False),
    }
    for name in wanted:
        for seed in args.ablation_seeds:
            specs.append(TrainSpec(
                # ⚠️ 消融名里也必须带 --tag：否则 42px 的消融会覆盖 44px 的同名 checkpoint
                name=f"{prefix}{name}_p{args.patch_size}"
                     f"{('_' + args.tag) if args.tag else ''}_s{seed}", seed=seed,
                patch_size=args.patch_size, row_height=args.row_height, epochs=args.epochs,
                batch_size=args.batch_size,
                image_noise=args.image_noise, **flags[name]))
    return specs


def main():
    ap = argparse.ArgumentParser(description="RAVT 训练（含消融）")
    ap.add_argument("--clip-path", default="clip_local_weights_224")
    ap.add_argument("--train-blocks", default="data/processed/train_blocks")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--log-dir", default="results/train")
    ap.add_argument("--row-height", type=int, default=int(os.environ.get("RAVT_BLOCK_HEIGHT", 44)),
                    help="行高（像素）；44=默认，42=与 patch 14 精确对齐的实验设置")
    ap.add_argument("--tag", default="",
                    help="插入 run 名的标签（例如 rh42 -> full_p14_rh42_s0.pth，"
                         "避免覆盖 44px 的主模型）")
    ap.add_argument("--init-from", default=None,
                    help="从已有 checkpoint 热启动（行高 42px 的实验建议从 44px 主模型热启动）")
    ap.add_argument("--patch-size", type=int, default=14,
                    help="行切割映射用的 patch 像素（ViT-L/14 用 14）")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr-backbone", type=float, default=1e-7)
    ap.add_argument("--lr-head", type=float, default=3e-5)
    ap.add_argument("--max-noise", type=float, default=0.30)
    ap.add_argument("--image-noise", action="store_true",
                    help="训练时加入图像级退化（模糊/JPEG/低分辨率/旋转/光照）。"
                         "走「扫描件」路线时用它把模型重训一遍")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--ablation", default="none",
                    help="none | all | 逗号分隔的消融名")
    ap.add_argument("--ablation-seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--device", default=None)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=None, help="每个 epoch 只跑 N 个 batch")
    ap.add_argument("--max-blocks", type=int, default=None, help="只用前 N 个训练块")
    ap.add_argument("--random-clip", action="store_true",
                    help="用随机初始化的小 CLIP 跑（不需要权重，用于冒烟）")
    ap.add_argument("--dry-run", action="store_true",
                    help="等价于 --random-clip --max-steps 3 --epochs 1 --batch-size 4")
    ap.add_argument("--check-backbone", action="store_true",
                    help="只打印 backbone 的 patch/hidden 配置后退出")
    ap.add_argument("--force", action="store_true",
                    help="已完成的 run 也重跑（默认跳过，便于中断后续跑）")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="允许在 CPU 上训练（会慢 10~30 倍，默认拒绝）")
    ap.add_argument("--base-dir", default=None,
                    help="基准目录（放 dataset/ 与 clip_local_weights_224/ 的地方）。"
                         "缺省时自动在脚本目录及其上 3 层里查找")
    args = ap.parse_args()

    from ravt_paths import setup
    setup(args.base_dir)

    if args.check_backbone:
        from model_defs import RAVTConfig, RAVTModel
        try:
            m = RAVTModel(RAVTConfig(clip_path=args.clip_path, patch_size=args.patch_size))
        except Exception as exc:
            raise SystemExit(
                f"❌ 加载 {args.clip_path} 失败: {type(exc).__name__}: {exc}\n"
                f"   请确认该目录存在、且含 config.json / pytorch_model.bin / text_encoder/。")
        print(f"backbone 实际网格: {m.grid_size}x{m.grid_size}"
              f"（patch={m.true_patch_size()}px），"
              f"行切割假定 patch={args.patch_size}px")
        warn = m.patch_mapping_warning()
        print("⚠️ " + warn if warn else "✅ patch 大小与行切割映射一致")
        return

    if args.dry_run:
        args.random_clip = True
        args.epochs = 1
        args.batch_size = 4
        args.max_steps = args.max_steps or 3
        args.max_blocks = args.max_blocks or 64
        print("🧪 dry-run: 随机初始化小 CLIP，epochs=1 batch=4 "
              f"max_steps={args.max_steps} max_blocks={args.max_blocks}")

    # ⚠️ 冒烟产物必须与正式训练分开命名与存放，
    #    否则随机小模型的权重会和真实权重同名，test.py 加载时报一堆 size mismatch。
    smoke = args.random_clip
    prefix = "dryrun_" if smoke else ""
    if smoke:
        if args.ckpt_dir == "checkpoints":
            args.ckpt_dir = "checkpoints_dryrun"
        if args.log_dir == "results/train":
            args.log_dir = "results/train_dryrun"
        print(f"🧪 冒烟模式：checkpoint 写到 {args.ckpt_dir}/，文件名带 dryrun_ 前缀，"
              "不会与正式训练产物混淆")

    specs = build_specs(args, prefix=prefix)
    print(f"将训练 {len(specs)} 个 run:")
    for s in specs:
        print(f"  {s.name:<28} seed={s.seed} patch={s.patch_size} epochs={s.epochs}")

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

    done, failed = [], []
    if args.init_from:
        for s in specs:
            s.extra["init_from"] = args.init_from
        print(f"⚙️  热启动自 {args.init_from}")
    for s in specs:
        if not args.force and already_done(args.log_dir, s.name):
            print(f"⏭️  跳过 {s.name}（已完成；要重跑加 --force）")
            continue
        try:
            out = train_one(s, train_blocks=args.train_blocks, ckpt_dir=args.ckpt_dir,
                            clip_path=args.clip_path, device=args.device,
                            max_steps=args.max_steps, num_workers=args.num_workers,
                            log_dir=args.log_dir,
                            allow_cpu=bool(args.allow_cpu or args.random_clip
                                           or (args.device or "").startswith("cpu")))
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
    if done:
        hint = f"python test.py --checkpoint {done[0]['checkpoint']} --n-tasks 10"
        if smoke:
            hint += " --random-clip"
            print("\n下一步（冒烟权重必须带 --random-clip，结果是随机的，没有意义）:")
        else:
            print("\n下一步:")
        print("  " + hint)
    print("=" * 74)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
