"""ColPali + LoRA，在 RAVT 同一批 50k mosaic 上做域内适配（第二个页面模型）。

    python code/adaptation/train_colpali_lora.py --root . --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "code" / "ravt"))


class MosaicPairs(torch.utils.data.Dataset):
    """(mosaic, clean target text)；负样本取自同长度桶。"""

    def __init__(self, root: Path, renderer, engine, n_rows: int = 5,
                 seed: int = 0, limit: int | None = None):
        self.files = sorted(Path(root).glob("*.txt"))
        if limit:
            self.files = self.files[:limit]
        if not self.files:
            raise FileNotFoundError(f"{root} 下没有 .txt 训练块")
        self.texts = [p.read_text(encoding="utf-8").strip() for p in self.files]
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
        self.n_rows = int(n_rows)
        if self.n_rows < 2:
            raise ValueError(f"n_rows 必须 >= 2，收到 {n_rows!r}")
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
        cand = [j for j in pool if j != idx]
        if len(cand) < self.n_rows - 1:
            cand = [j for j in range(len(self.texts)) if j != idx]
        negs = rng.sample(cand, self.n_rows - 1)
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--blocks", default="data/processed/train_blocks_rh42")
    ap.add_argument("--base", default="colpali_offline_weights/base_model")
    ap.add_argument("--out", default="checkpoints/colpali_lora_s0")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
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
    for path in (root, root / "code" / "ravt"):
        sys.path.insert(0, str(path))
    from model_defs import BlockRenderer, CorruptionEngine, RendererConfig

    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("❌ 没有 CUDA")
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark = False

    renderer = BlockRenderer(RendererConfig(block_height=42, canvas_size=224))
    engine = CorruptionEngine(random.Random(a.seed))
    ds = MosaicPairs(root / a.blocks, renderer, engine, seed=a.seed, limit=a.limit)
    print(f"训练块 {len(ds)}")
    _img, _txt = ds[0]
    print(f"dataset self-check: {_img.size}, query {len(_txt.split())} words")

    from colpali_engine.models import ColPali, ColPaliProcessor
    from peft import LoraConfig, get_peft_model

    model = ColPali.from_pretrained(str(root / a.base), dtype=torch.bfloat16,
                                    device_map=a.device, local_files_only=True)
    try:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    except Exception:
        pass
    proc = ColPaliProcessor.from_pretrained(str(root / a.base), local_files_only=True)

    # 不用 "all-linear"：它会把 embedding 层也纳入，而这些层在 device_map 下是 meta
    # tensor，save_pretrained 会报 "Cannot copy out of meta tensor"。
    targets = ["q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj"]
    cfg = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.05,
                     target_modules=targets, bias="none",
                     task_type="FEATURE_EXTRACTION")
    saver = get_peft_model(model, cfg)
    outer = saver
    trainable = [p for p in outer.parameters() if p.requires_grad]
    print(f"LoRA 参数 {sum(p.numel() for p in trainable)/1e6:.1f}M")

    opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=0.0)
    loader_generator = torch.Generator().manual_seed(a.seed)
    loader = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True,
                                         num_workers=a.workers, collate_fn=collate,
                                         drop_last=True, pin_memory=True,
                                         generator=loader_generator)
    out = root / a.out
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "train_log.jsonl").open("a", encoding="utf-8")
    total_steps = min(a.max_steps or 10**9, max(1, len(loader) // a.accum) * a.epochs)
    step = 0
    t0 = time.time()
    for epoch in range(a.epochs):
        ds.epoch = epoch
        ds.noise_rate = a.max_noise * (epoch + 1) / a.epochs
        ds.image_severity = a.max_image * (epoch + 1) / a.epochs
        for i, (images, texts) in enumerate(loader):
            enc = proc.process_images(images).to(a.device)
            if "pixel_values" in enc:
                enc["pixel_values"] = enc["pixel_values"].to(torch.bfloat16)
            d = outer(**enc)
            qenc = proc.process_queries(texts).to(a.device)
            q = outer(**qenc)
            if a.dry_run:
                print("doc", tuple(d.shape), "query", tuple(q.shape))
                break
            d = F.normalize(d.float(), dim=-1)
            q = F.normalize(q.float(), dim=-1)
            sim = torch.einsum("bqd,cpd->bcqp", q, d).max(dim=-1).values
            scores = (sim * qenc["attention_mask"][:, None, :].float()).sum(-1)
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
                    print(f"step {step} loss {loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
                    log.write(json.dumps({"step": step, "epoch": epoch,
                                          "loss": float(loss.item())}) + "\n")
                    log.flush()
                if a.save_every and step % a.save_every == 0:
                    saver.save_pretrained(out / f"step_{step:05d}")
                if a.max_steps and step >= a.max_steps:
                    break
        if a.dry_run or (a.max_steps and step >= a.max_steps):
            break
    if not a.dry_run:
        saver.save_pretrained(out)
        print(f"适配器 -> {out}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
