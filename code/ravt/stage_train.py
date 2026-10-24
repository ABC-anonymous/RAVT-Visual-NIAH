"""阶段 2：训练。

两类训练：

* `ravt`        —— 论文主模型（ViT + 多层融合 + RAVT 行切割 + 门控投影），支持全部消融开关
* `text_encoder` —— E2 的关键对照：把同样的噪声课程 + 硬负样本训练**施加到纯文本编码器**上，
                    用来回答"鲁棒性到底来自渲染成图，还是来自噪声训练"

两个训练都写 `results/jsonl/train/<run_id>.jsonl`（逐 epoch 记录），
checkpoint 落在 `checkpoints/`。
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from schemas import SCHEMA_TRAIN
from common import (JsonlWriter, ensure_dir, now_iso, repo_path, seed_everything,
                     worker_init_fn, env_snapshot)
from config import TrainSpec
from model_defs import (BlockRenderer, CorruptionEngine, RandomProcessor, RandomTokenizer,
                         RAVTConfig, RAVTModel, RendererConfig, build_mosaic,
                         build_random_clip_cfg, multi_grained_loss)

DEFAULT_TRAIN_BLOCKS = "data/processed/train_blocks"
DEFAULT_CKPT_DIR = "checkpoints"


def _length_bucket(path: Path) -> int:
    """从文件名 `L{字符数}_{序号}.txt` 取出长度桶编号（与老代码一致）。"""
    stem = path.stem
    if stem.startswith("L"):
        head = stem.split("_")[0][1:]
        if head.isdigit():
            return int(head)
    return len(stem)


def epoch_schedules(spec: "TrainSpec", epoch: int) -> tuple[float, float, float | None]:
    """返回第 epoch 轮的 (字符错误率, 成像退化强度, 成像退化覆盖率)。

    两条轴各有自己的课程：
      * 字符错误率：`max_noise * epoch/epochs`（与老代码逐位一致，保证可比）
      * 成像强度  ：`image_max_severity * epoch/epochs`（linear），
                    或 constant（一上来就用满），或 step（前 1/4 轮不用）
    覆盖率与强度默认同值；显式给 `image_coverage` 就固定覆盖率、只变强度。
    """
    t = epoch / max(spec.epochs, 1)
    char_rate = (spec.max_noise * t) if spec.use_noise else 0.0
    img_max = (spec.image_max_severity if spec.image_max_severity is not None
               else spec.max_noise)
    if not (spec.image_noise and spec.use_noise):
        img = 0.0
    elif spec.image_curriculum == "constant":
        img = img_max
    elif spec.image_curriculum == "step":
        img = 0.0 if t < 0.25 else img_max
    else:
        img = img_max * t
    cov = img if spec.image_coverage is None else spec.image_coverage
    return char_rate, img, cov


def resolve_device(device: str | None, allow_cpu: bool = False) -> str:
    """决定训练设备。

    绝不静默退回 CPU：ViT-L/14 在 CPU 上每步要几十秒（GPU 上是 1~3 秒），
    一个 15 epoch 的 run 会从 2 小时变成 2 天以上。必须显式要求才允许 CPU。
    """
    import torch
    if device:
        if device.startswith("cpu"):
            print("⚠️ 使用 CPU 训练：每步比 GPU 慢 10~30 倍，一个 run 可能要 2 天以上。"
                  "（显式指定 --device cpu，视为有意为之）")
        return device
    if torch.cuda.is_available():
        return "cuda"
    if allow_cpu:
        print("⚠️ CUDA 不可用，按 --allow-cpu 在 CPU 上运行（会慢 10~30 倍）")
        return "cpu"
    raise SystemExit(
        "❌ 检测不到可用的 CUDA 设备，且未指定 --device。\n"
        "   不要用 CPU 训练 ViT-L/14 —— 每步要几十秒，一个 run 要 2 天以上。\n"
        "   排查：\n"
        "     python -c \"import torch;print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)\"\n"
        "     nvidia-smi          # 训练时应该看到 python 进程占着显存\n"
        "   常见原因：装的是 CPU 版 torch、CUDA_VISIBLE_DEVICES 为空、驱动/容器没挂 GPU。\n"
        "   确认就要在 CPU 上跑，加 --device cpu --allow-cpu。")


def describe_device(device: str) -> str:
    if device.startswith("cuda"):
        try:
            import torch
            p = torch.cuda.get_device_properties(torch.cuda.current_device())
            return f"{device} ({p.name}, {p.total_memory / 1024 ** 3:.0f} GB)"
        except Exception:
            return device
    return device


# --------------------------------------------------------------------------- #
# 数据集
# --------------------------------------------------------------------------- #
class MosaicTrainDataset(Dataset):
    """每次取一条目标文本 + 4 条同长度桶内的困难负样本，拼成一个 5 行 mosaic。"""

    def __init__(self, root: Path, spec: TrainSpec, renderer: BlockRenderer,
                 max_blocks: int | None = None):
        self.root = Path(root)
        self.spec = spec
        self.renderer = renderer
        self.corrupter = CorruptionEngine()
        self.files = sorted(p for p in self.root.iterdir() if p.suffix == ".txt")
        if max_blocks:
            self.files = self.files[:max_blocks]
        if not self.files:
            raise FileNotFoundError(f"{self.root} 下没有 .txt 训练块，请先跑 stage_data train-corpus")
        self._text_cache: dict[Path, str] = {}
        self.noise_rate = 0.0
        #: 图像退化的强度与覆盖率（由 epoch 课程分别设置；None -> 与强度同值）
        self.image_severity = 0.0
        self.image_coverage: float | None = None

        # 按字符长度分桶；同时记录每个文件属于哪个桶
        self.buckets: dict[int, list[int]] = {}
        self.file_bucket: list[int] = []
        for idx, p in enumerate(self.files):
            key = _length_bucket(p)
            self.file_bucket.append(key)
            self.buckets.setdefault(key, []).append(idx)
        self.bucket_keys = sorted(self.buckets)
        self._rng = random.Random(spec.seed)
        self._worker_id: int | None = None

    def _rng_for_worker(self) -> random.Random:
        """每个 DataLoader worker 用不同的随机流（否则 8 个 worker 会重复同样的采样序列）。"""
        try:
            from torch.utils.data import get_worker_info
            info = get_worker_info()
            wid = info.id if info is not None else 0
        except Exception:
            wid = 0
        if self._worker_id != wid:
            self._rng = random.Random(self.spec.seed * 100003 + wid)
            self._worker_id = wid
        return self._rng

    def __len__(self):
        return len(self.files)

    def _text(self, idx: int) -> str:
        p = self.files[idx]
        if p not in self._text_cache:
            self._text_cache[p] = p.read_text(encoding="utf-8").strip()
        return self._text_cache[p]

    def __getitem__(self, idx):
        spec = self.spec
        rng = self._rng_for_worker()
        target_pos = rng.randrange(5)

        if spec.use_buckets:
            # 关键：负样本必须来自**目标自己所在的长度桶**（老代码就是这个语义）。
            # 若随机挑桶，负样本长度就与目标无关，长度/行数会变成可利用的捷径，
            # "length bucketing 提供困难负样本" 这一设计就失效了。
            key = self.file_bucket[idx]
            pool = self.buckets.get(key) or list(range(len(self.files)))
        else:
            pool = list(range(len(self.files)))

        # 4 条负样本必须是 idx 之外的样本，避免"目标行与负样本行内容相同"
        others = [c for c in pool if c != idx]
        if len(others) < 4:
            others = [c for c in range(len(self.files)) if c != idx]
        negatives = rng.sample(others, 4)

        rows, target_text = [], ""
        neg_iter = iter(negatives)
        for pos in range(5):
            if pos == target_pos:
                target_text = self._text(idx)
                rows.append(target_text)
            else:
                rows.append(self._text(next(neg_iter)))

        rate = self.noise_rate if spec.use_noise else 0.0
        families = ("homoglyph", "gibberish", "occlusion") if spec.use_noise else ()
        # 图像级退化：让模型见过模糊/JPEG/低分辨率等真实扫描件退化。
        # 现有的显存/时间开销很小（PIL 操作，在 DataLoader 里并行完成）。
        img_families = tuple(spec.image_kinds) if (spec.image_noise and spec.use_noise) else ()
        canvas, _ = build_mosaic(rows, self.renderer, self.corrupter,
                                 noise_rate=rate, char_families=families,
                                 image_families=img_families,
                                 # 强度与覆盖率分开给：老行为是 coverage=severity，
                                 # 现在既支持"两者同升"，也支持固定覆盖率只调强度。
                                 image_severity=self.image_severity if img_families else 0.0,
                                 image_coverage=self.image_coverage)
        return canvas, target_text, target_pos


class TextPairDataset(Dataset):
    """E2 文本侧对照：clean 文本 -> 加噪后的同一段文本，构成正样本对。"""

    def __init__(self, root: Path, spec: TrainSpec, limit: int | None = None):
        self.root = Path(root)
        self.spec = spec
        self.corrupter = CorruptionEngine(random.Random(spec.seed))
        self.files = sorted(p for p in self.root.iterdir() if p.suffix == ".txt")
        if limit:
            self.files = self.files[:limit]
        if not self.files:
            raise FileNotFoundError(f"{self.root} 下没有训练文本")
        self.noise_rate = 0.0

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        text = self.files[idx].read_text(encoding="utf-8").strip()
        rate = self.noise_rate if self.spec.use_noise else 0.0
        families = ["homoglyph", "gibberish", "occlusion"] if self.spec.use_noise else []
        noisy = self.corrupter.apply_char(text, rate, families)
        return text, noisy


def _mosaic_collate(batch):
    """默认 collate 无法处理 PIL.Image / str，这里保持它们为 list 交给训练循环处理。"""
    canvases = [b[0] for b in batch]
    texts = [b[1] for b in batch]
    positions = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return canvases, texts, positions


def _pair_collate(batch):
    return [b[0] for b in batch], [b[1] for b in batch]


# --------------------------------------------------------------------------- #
# RAVT 训练
# --------------------------------------------------------------------------- #
def train_ravt(spec: TrainSpec, train_blocks: str = DEFAULT_TRAIN_BLOCKS,
               ckpt_dir: str = DEFAULT_CKPT_DIR, clip_path: str = "clip_local_weights_224",
               device: str | None = None, max_steps: int | None = None,
               num_workers: int = 8, log_dir: str = "results/jsonl/train",
               allow_cpu: bool = False) -> dict:
    from transformers import CLIPProcessor, CLIPTokenizer
    seed_everything(spec.seed)
    device = resolve_device(device, allow_cpu=allow_cpu or spec.extra.get("random_init", False))
    print(f"使用设备: {describe_device(device)}")
    run_id = spec.name
    log_path = repo_path(log_dir) / f"{run_id}.jsonl"

    model_cfg = RAVTConfig(
        clip_path=clip_path, patch_size=spec.patch_size, row_height=spec.row_height,
        n_rows=spec.n_rows, use_fusion=spec.use_fusion, use_slicing=spec.use_slicing,
        use_gated_proj=spec.use_gated_proj, temperature_init=14.28,
        random_init=bool(spec.extra.get("random_init")),
        random_cfg=dict(spec.extra.get("random_cfg") or {}),
    )
    model = RAVTModel(model_cfg).to(device)
    warn = model.patch_mapping_warning()

    # 可选：从已有 checkpoint 热启动（退化感知"适配"版协议）。
    # 完整重训 15 轮很贵；从字符噪声版主模型出发再跑 5 轮退化课程，
    # 是期刊里常见、且成本可接受的折中（论文里要写明是 warm-start）。
    init_from = spec.extra.get("init_from")
    if init_from:
        from pathlib import Path as _Path
        p = _Path(init_from)
        p = p if p.is_absolute() else repo_path(str(init_from))
        if not p.exists():
            raise FileNotFoundError(f"init_from 指定的权重不存在: {p}")
        sd = torch.load(p, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"  ⚙️  热启动自 {p.name}：缺失 {len(missing)} 个张量，多余 {len(unexpected)} 个")

    if model_cfg.random_init:
        # 冒烟模式：不加载任何权重，只需要张量形状对得上
        processor = RandomProcessor(model_cfg.image_size)
        tokenizer = RandomTokenizer(model_cfg.random_cfg.get("vocab_size", 49408))
    else:
        processor = CLIPProcessor.from_pretrained(clip_path)
        tokenizer = CLIPTokenizer.from_pretrained(clip_path)
    # 渲染器必须读环境变量：行高 42px 的实验要求"训练时每行也按 42px 渲染"，
    # 否则模型的行窗口（由 RAVTConfig.row_height 决定）与画布上的行位置错位。
    from stage_data import renderer_from_env
    renderer = renderer_from_env()
    dataset = MosaicTrainDataset(repo_path(train_blocks), spec, renderer,
                                 max_blocks=spec.extra.get("max_blocks"))
    loader = DataLoader(dataset, batch_size=spec.batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=(device == "cuda"),
                        drop_last=True, collate_fn=_mosaic_collate,
                        worker_init_fn=worker_init_fn(spec.seed))

    groups = [
        {"params": [p for p in model.backbone.parameters() if p.requires_grad],
         "lr": spec.lr_backbone, "name": "backbone"},
        {"params": [p for p in model.text_encoder.parameters() if p.requires_grad],
         "lr": spec.lr_backbone * 5, "name": "text"},
        {"params": model.proj.parameters(), "lr": spec.lr_head, "name": "proj"},
        {"params": [model.temperature], "lr": 1e-3, "name": "temperature"},
    ]
    if spec.use_fusion:
        groups.insert(2, {"params": model.fusion.parameters(), "lr": spec.lr_head, "name": "fusion"})
    optimizer = torch.optim.AdamW(groups, weight_decay=spec.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[g["lr"] for g in groups],
        steps_per_epoch=max(1, len(loader)), epochs=spec.epochs)

    ensure_dir(repo_path(ckpt_dir))
    writer = JsonlWriter(log_path)
    writer.write({"record": "run_start", "schema": SCHEMA_TRAIN, "timestamp": now_iso(),
                  "run_id": run_id, "kind": "ravt", "spec": spec.to_dict(),
                  "model_cfg": model_cfg.to_dict(), "device": device,
                  "grid_size": model.grid_size, "expected_grid": model.expected_grid,
                  "patch_mapping_warning": warn, "env": env_snapshot(),
                  "n_train_blocks": len(dataset), "steps_per_epoch": len(loader)})
    if warn:
        print(f"⚠️  {warn}")

    t_start = time.time()
    ckpt = repo_path(ckpt_dir) / f"{run_id}.pth"   # 先声明，避免 epochs=0 时未定义
    for epoch in range(spec.epochs):
        model.train()
        # 字符课程与成像课程分开推进（见 epoch_schedules 的说明）
        dataset.noise_rate, _img_sev, _img_cov = epoch_schedules(spec, epoch)
        dataset.image_severity = _img_sev
        dataset.image_coverage = _img_cov
        writer.write({"record": "epoch_schedule", "timestamp": now_iso(),
                      "run_id": run_id, "epoch": epoch,
                      "char_noise_rate": dataset.noise_rate,
                      "image_severity": dataset.image_severity,
                      "image_coverage": dataset.image_coverage})
        agg = {"total": 0.0, "global": 0.0, "local": 0.0, "margin": 0.0}
        nb = 0
        t_epoch = time.time()
        for step, (canvas, target_text, target_pos) in enumerate(loader):
            px = processor(images=list(canvas), return_tensors="pt")["pixel_values"].to(device)
            # 注意：用下标取而不是 .to(device)，这样假 tokenizer（冒烟模式）也能用
            tok = tokenizer(list(target_text), padding="max_length", max_length=77,
                            truncation=True, return_tensors="pt")
            input_ids = tok["input_ids"].to(device)
            attention_mask = tok["attention_mask"].to(device)
            target_pos = target_pos.long().to(device)

            optimizer.zero_grad(set_to_none=True)
            v_feats, t_feat, temp = model(px, input_ids, attention_mask)
            losses = multi_grained_loss(
                v_feats, t_feat, target_pos, temp,
                w_local=spec.loss_weight_local, w_margin=spec.loss_weight_margin,
                margin=spec.margin,
                use_local=spec.use_local_loss, use_margin=spec.use_margin_loss)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 0.5)
            optimizer.step()
            scheduler.step()

            for k in ("total", "global", "local", "margin"):
                agg[k] += float(losses[k].item()) if k in losses else 0.0
            nb += 1
            # 每 20 步打印一次速度与 ETA：一眼就能看出设备/数据加载是否正常
            if nb % 20 == 0 or (max_steps and nb == max_steps):
                spb = (time.time() - t_epoch) / nb
                left = (len(loader) - nb) * spb if not max_steps else 0
                print(f"\r[{run_id}] E{epoch + 1} {nb}/{len(loader)} "
                      f"{spb:.2f}s/step  ETA {left / 60:.0f} min   ", end="")
                if spb > 10:
                    print(f"\n  ⚠️ {spb:.1f} 秒/步 —— 异常慢。GPU 上应为 1~3 秒/步。"
                          "检查：nvidia-smi 是否看到 python 进程、日志里 device 是否为 cuda。")
            if max_steps and step + 1 >= max_steps:
                break

        s_per_epoch = time.time() - t_epoch
        record = {"record": "epoch", "schema": SCHEMA_TRAIN, "run_id": run_id,
                  "timestamp": now_iso(), "epoch": epoch, "noise_rate": dataset.noise_rate,
                  "temperature": float(model.temperature.item()),
                  "n_batches": nb, "elapsed_s": round(time.time() - t_start, 1),
                  "s_per_epoch": round(s_per_epoch, 1),
                  "s_per_step": round(s_per_epoch / max(nb, 1), 3),
                  "device": device}
        for k, v in agg.items():
            record[f"loss_{k}"] = v / max(nb, 1)
        writer.write(record)
        print(f"[{run_id}] epoch {epoch + 1}/{spec.epochs} "
              f"noise={dataset.noise_rate:.2f} loss={record['loss_total']:.4f} "
              f"{record['s_per_epoch'] / 60:.1f} min/epoch "
              f"({record['s_per_step']:.2f}s/step)")

        torch.save(model.state_dict(), ckpt)

    writer.write({"record": "run_summary", "schema": SCHEMA_TRAIN, "run_id": run_id,
                  "timestamp": now_iso(), "checkpoint": str(ckpt),
                  "elapsed_s": round(time.time() - t_start, 1)})
    writer.close()
    print(f"[{run_id}] 完成 -> {ckpt}")
    return {"run_id": run_id, "checkpoint": str(ckpt)}


# --------------------------------------------------------------------------- #
# 文本编码器抗噪微调（E2）
# --------------------------------------------------------------------------- #
def train_text_encoder(spec: TrainSpec, train_blocks: str = DEFAULT_TRAIN_BLOCKS,
                       ckpt_dir: str = DEFAULT_CKPT_DIR, device: str | None = None,
                       max_steps: int | None = None, num_workers: int = 4,
                       log_dir: str = "results/jsonl/train",
                       allow_cpu: bool = False) -> dict:
    from transformers import AutoModel, AutoTokenizer

    seed_everything(spec.seed)
    device = resolve_device(device, allow_cpu=allow_cpu)
    print(f"使用设备: {describe_device(device)}")
    run_id = spec.name
    writer = JsonlWriter(repo_path(log_dir) / f"{run_id}.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(spec.text_backbone)
    model = AutoModel.from_pretrained(spec.text_backbone).to(device)
    for p in model.parameters():
        p.requires_grad = True

    dataset = TextPairDataset(repo_path(train_blocks), spec,
                              limit=spec.extra.get("max_blocks"))
    loader = DataLoader(dataset, batch_size=spec.batch_size, shuffle=True, drop_last=True,
                        num_workers=num_workers, collate_fn=_pair_collate,
                        worker_init_fn=worker_init_fn(spec.seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.lr_head,
                                  weight_decay=spec.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=spec.lr_head, steps_per_epoch=max(1, len(loader)), epochs=spec.epochs)

    writer.write({"record": "run_start", "schema": SCHEMA_TRAIN, "timestamp": now_iso(),
                  "run_id": run_id, "kind": "text_encoder", "spec": spec.to_dict(),
                  "device": device, "env": env_snapshot(), "n_pairs": len(dataset)})

    def _pool(out, mask):
        if spec.text_pooling == "mean":
            m = mask.unsqueeze(-1).float()
            return (out.last_hidden_state * m).sum(1) / m.sum(1).clamp(min=1e-6)
        return out.last_hidden_state[:, 0]

    t_start = time.time()
    for epoch in range(spec.epochs):
        model.train()
        dataset.noise_rate, _img_sev, _img_cov = epoch_schedules(spec, epoch)
        dataset.image_severity = _img_sev
        dataset.image_coverage = _img_cov
        total, nb = 0.0, 0
        for step, (clean, noisy) in enumerate(loader):
            q = tokenizer(list(clean), padding=True, truncation=True,
                          max_length=spec.text_max_length, return_tensors="pt").to(device)
            d = tokenizer(list(noisy), padding=True, truncation=True,
                          max_length=spec.text_max_length, return_tensors="pt").to(device)
            optimizer.zero_grad(set_to_none=True)
            q_feat = F.normalize(_pool(model(**q), q["attention_mask"]), dim=1)
            d_feat = F.normalize(_pool(model(**d), d["attention_mask"]), dim=1)
            logits = q_feat @ d_feat.t() / 0.05
            labels = torch.arange(logits.shape[0], device=device)
            loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += float(loss.item())
            nb += 1
            if max_steps and step + 1 >= max_steps:
                break

        writer.write({"record": "epoch", "schema": SCHEMA_TRAIN, "run_id": run_id,
                      "timestamp": now_iso(), "epoch": epoch,
                      "noise_rate": dataset.noise_rate, "loss_total": total / max(nb, 1),
                      "elapsed_s": round(time.time() - t_start, 1)})
        print(f"[{run_id}] epoch {epoch + 1}/{spec.epochs} "
              f"noise={dataset.noise_rate:.2f} loss={total / max(nb, 1):.4f}")

    out_dir = ensure_dir(repo_path(ckpt_dir) / run_id)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    writer.write({"record": "run_summary", "schema": SCHEMA_TRAIN, "run_id": run_id,
                  "timestamp": now_iso(), "checkpoint": str(out_dir),
                  "elapsed_s": round(time.time() - t_start, 1)})
    writer.close()
    print(f"[{run_id}] 完成 -> {out_dir}")
    return {"run_id": run_id, "checkpoint": str(out_dir)}


# --------------------------------------------------------------------------- #
def train_one(spec: TrainSpec, **kwargs) -> dict:
    if spec.kind == "ravt":
        return train_ravt(spec, **kwargs)
    if spec.kind == "text_encoder":
        kwargs.pop("clip_path", None)
        return train_text_encoder(spec, **kwargs)
    raise ValueError(f"未知训练类型: {spec.kind}")
