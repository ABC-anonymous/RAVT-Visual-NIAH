"""ColQwen2 + LoRA，在 RAVT 同一批 50k mosaic 上做域内适配。

训练对与 RAVT 完全一致：query = 目标行的干净文本，document = 5 行 mosaic
（目标行含在同一张图里，其余四行是同长度桶的困难负样本），腐蚀课程也一致。

    python code/adaptation/train_colqwen2_lora.py --root . --dry-run
    python code/adaptation/train_colqwen2_lora.py --root . --max-steps 20 --limit 400
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "code" / "ravt"))


class MosaicPairs(torch.utils.data.Dataset):
    """(mosaic, clean target text)；负样本随机取自同一批训练块。"""

    def __init__(self, root: Path, renderer, engine, n_rows: int = 5,
                 seed: int = 0, limit: int | None = None):
        self.files = sorted(Path(root).glob("*.txt"))
        if limit:
            self.files = self.files[:limit]
        if not self.files:
            raise FileNotFoundError(f"{root} 下没有 .txt 训练块")
        self.texts = [p.read_text(encoding="utf-8").strip() for p in self.files]
        # 与 RAVT 一致：按文件名 L{字符数}_ 分桶，负样本取自同桶
        self.buckets: dict[int, list[int]] = {}
        self.file_bucket: list[int] = []
        for i, p in enumerate(self.files):
            stem = p.stem
            key = len(stem)
            if stem.startswith("L"):
                head = stem.split("_")[0][1:]
                if head.isdigit():
                    key = int(head)
            self.buckets.setdefault(key, []).append(i)
            self.file_bucket.append(key)
        self.renderer = renderer
        self.engine = engine
        self.n_rows = n_rows
        self.seed = seed
        self.noise_rate = 0.0
        self.image_severity = 0.0
        self.image_coverage = None
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        from model_defs import build_mosaic

        rng = random.Random(self.seed * 100003 + idx * 7919 + self.epoch * 104729)
        pos = rng.randrange(self.n_rows)
        pool = self.buckets.get(self.file_bucket[idx]) or list(range(len(self.texts)))
        if len(pool) <= self.n_rows:
            pool = list(range(len(self.texts)))
        negs: list[int] = []
        while len(negs) < self.n_rows - 1:
            j = pool[rng.randrange(len(pool))]
            if j != idx and j not in negs:
                negs.append(j)
        rows, it = [], iter(negs)
        for p in range(self.n_rows):
            rows.append(self.texts[idx] if p == pos else self.texts[next(it)])
        canvas, _ = build_mosaic(
            rows, self.renderer, self.engine,
            noise_rate=self.noise_rate,
            char_families=("homoglyph", "gibberish", "occlusion"),
            image_families=("gaussian_blur", "jpeg", "low_resolution", "shadow"),
            image_severity=self.image_severity,
            image_coverage=self.image_coverage,
        )
        return canvas, self.texts[idx]


def collate(batch):
    return [b[0] for b in batch], [b[1] for b in batch]


def load_model(base: str, device: str, dtype: torch.dtype):
    from colpali_engine.models import ColQwen2, ColQwen2Processor

    model = ColQwen2.from_pretrained(base, dtype=dtype, device_map=device,
                                     local_files_only=True)
    try:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    except Exception:                                            # noqa: BLE001
        pass
    proc = ColQwen2Processor.from_pretrained(base, local_files_only=True)
    return model, proc


def add_lora(model, r: int, alpha: int):
    """返回 (负责保存的 PeftModel, 用于前向的外层 ColQwen2)。"""
    from peft import LoraConfig, get_peft_model

    targets = ["q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj"]
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.05,
                     target_modules=targets, bias="none",
                     task_type="FEATURE_EXTRACTION")
    try:
        peft_model = get_peft_model(model, cfg)
        return peft_model, peft_model
    except Exception as exc:                                     # noqa: BLE001
        print(f"  外层 LoRA 失败（{type(exc).__name__}: {exc}），改挂内层 Qwen2-VL")
        inner = get_peft_model(model.model, cfg)
        model.model = inner
        return inner, model


def encode_images(model, proc, images, device, dtype):
    enc = proc.process_images(images).to(device)
    if "pixel_values" in enc:
        enc["pixel_values"] = enc["pixel_values"].to(dtype)
    return model(**enc)


def encode_queries(model, proc, texts, device):
    enc = proc.process_queries(texts).to(device)
    return model(**enc), enc["attention_mask"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--blocks", default="data/processed/train_blocks_rh42")
    ap.add_argument("--base", default="models_local/colqwen2-base")
    ap.add_argument("--out", default="checkpoints/colqwen2_lora_s0")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--max-noise", type=float, default=0.30)
    ap.add_argument("--max-image", type=float, default=0.35)
    ap.add_argument("--temperature", type=float, default=20.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    root = Path(a.root).resolve()
    for p in (root, root / "code" / "ravt"):
        sys.path.insert(0, str(p))
    from model_defs import BlockRenderer, CorruptionEngine, RendererConfig

    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("❌ 没有 CUDA；LoRA 训练不在 CPU 上跑")
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark = False

    renderer = BlockRenderer(RendererConfig(block_height=42, canvas_size=224))
    engine = CorruptionEngine(random.Random(a.seed))
    ds = MosaicPairs(root / a.blocks, renderer, engine, seed=a.seed, limit=a.limit)
    print(f"训练块 {len(ds)}，渲染 {renderer.describe()}")

    model, proc = load_model(str(root / a.base), a.device, torch.bfloat16)
    saver, outer = add_lora(model, a.lora_r, a.lora_alpha)
    trainable = [p for p in outer.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"LoRA 参数量 {n_train/1e6:.1f}M / 总 {sum(p.numel() for p in outer.parameters())/1e6:.0f}M")

    opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=0.0)
    loader_generator = torch.Generator().manual_seed(a.seed)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
        collate_fn=collate, drop_last=True, pin_memory=True,
        generator=loader_generator)

    out = root / a.out
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "train_log.jsonl").open("a", encoding="utf-8")
    step = 0
    t0 = time.time()

    total_steps = min(a.max_steps or 10**9,
                      max(1, len(loader) // max(a.accum, 1)) * a.epochs)
    for epoch in range(a.epochs):
        ds.epoch = epoch
        ds.noise_rate = a.max_noise * (epoch + 1) / a.epochs
        ds.image_severity = a.max_image * (epoch + 1) / a.epochs
        print(f"\n== epoch {epoch+1}/{a.epochs}  char={ds.noise_rate:.2f} "
              f"image={ds.image_severity:.2f} ==")
        for i, (images, texts) in enumerate(loader):
            if a.dry_run and i >= 1:
                break
            if step == 0:
                print("  dry-run：第一批", len(images), "图 /", len(texts), "query")
            d = encode_images(outer, proc, images, a.device, torch.bfloat16)
            q, qmask = encode_queries(outer, proc, texts, a.device)
            if step == 0:
                print("  doc", tuple(d.shape), "query", tuple(q.shape))
            if a.dry_run:
                break
            d = F.normalize(d.float(), dim=-1)
            q = F.normalize(q.float(), dim=-1)
            sim = torch.einsum("bqd,cpd->bcqp", q, d).max(dim=-1).values
            scores = (sim * qmask[:, None, :].float()).sum(-1)
            loss = F.cross_entropy(scores * a.temperature,
                                   torch.arange(len(images), device=a.device))
            (loss / a.accum).backward()
            if (i + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    lr = a.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total_steps)))
                    for g in opt.param_groups:
                        g["lr"] = max(lr, a.lr * 0.05)
                    print(f"  step {step}  loss {loss.item():.4f}  "
                          f"{time.time()-t0:.0f}s", flush=True)
                    log.write(json.dumps({"step": step, "epoch": epoch,
                                          "loss": float(loss.item()),
                                          "char": ds.noise_rate,
                                          "image": ds.image_severity}) + "\n")
                    log.flush()
                if a.save_every and step % a.save_every == 0:
                    saver.save_pretrained(out / f"step_{step:05d}")
                if a.max_steps and step >= a.max_steps:
                    break
        if a.max_steps and step >= a.max_steps:
            break

    if not a.dry_run:
        saver.save_pretrained(out)
        print(f"\n适配器 -> {out}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
