"""Evaluation and training matrix configuration.

改这里就能改实验规模，不需要动任何脚本。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

# --------------------------------------------------------------------------- #
# 1. Benchmark 网格
# --------------------------------------------------------------------------- #
#: v1_legacy —— 与现有 results/main/*.csv 完全一致的旧网格（用于复现旧结论）
#: v2_fixed  —— 修复两个构造缺陷后的网格（用于修改稿的新结果）
BENCHMARKS: dict[str, dict[str, Any]] = {
    "v1_legacy": {
        "block_counts": [20, 50, 100, 150, 200, 300, 500, 1000],
        "noise_levels": [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
        "depths": [0.1, 0.5, 0.9],
        "repeats": 5,
        "seed": 1234,
        # 旧行为：needle 落在 int(b*5*depth) 处（实测恒为第 0 行）
        "target_position": "depth_formula",
        # 旧行为：has_distractor 每任务独立抛硬币
        "distractor_rule": "random",
    },
    "v2_fixed": {
        "block_counts": [20, 50, 100, 150, 200, 300, 500, 1000],
        "noise_levels": [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
        "depths": [0.1, 0.5, 0.9],
        "repeats": 20,          # E6: 5 -> 20，把每个单元格的样本量提到 480
        "seed": 20260925,
        # 修复 1：needle 在 haystack 内随机取样，target_row 均匀覆盖 0..4
        "target_position": "random",
        # 修复 2：干扰项按 task_id 确定性分配，所有模型分母一致
        "distractor_rule": "deterministic",
    },
    #: v3_tmm —— TMM 版主网格。相对 v2_fixed 的两点变化：
    #:   ① 恢复长上下文：块数重新覆盖到 1000（v2 为了控成本砍到了 100）
    #:   ② 恢复三个 depth 分层，并提高重复数到 10（每个 (噪声, 块数) 点 n=30）
    "v3_tmm": {
        "block_counts": [20, 50, 100, 150, 200, 300, 500, 1000],
        "noise_levels": [0.0, 0.20, 0.40],
        "depths": [0.1, 0.5, 0.9],
        "repeats": 10,
        "seed": 20260929,
        "target_position": "random",
        "distractor_rule": "deterministic",
    },
}

#: TMM 版三条补充测试轴的网格（与主网格共用同一份语料与渲染器）。
#:   * image —— 纯成像退化强度曲线（文本干净，OCR 流水线与视觉路线公平对撞）
#:   * real  —— 真实扫描件退化（透视/折痕/透印/摩尔纹/复印/椒盐/非均匀光照）
#:   * grid  —— 2D 热图：内容错字率 × 成像退化强度，用来定位"视觉反超"的临界区
TMM_GRIDS: dict[str, dict[str, Any]] = {
    "image": {
        "block_counts": [20, 50, 100, 300, 1000],
        "noise_levels": [0.0],
        "image_levels": [0.0, 0.1, 0.2, 0.3, 0.4],
        "depths": [0.5],
        "repeats": 4,
    },
    "real": {
        "block_counts": [20, 50, 100, 300, 1000],
        "noise_levels": [0.0],
        # 实测（tesseract，16 样本）：real_scan 的 CER 在 0.2 档是 0.11、
        # 0.4 档是 0.64（已经接近不可读）。为了画出有区分度的曲线，
        # 把档位收在 0~0.3，避免全部基线一起掉到 0。
        "image_levels": [0.0, 0.1, 0.2, 0.3],
        "depths": [0.5],
        "repeats": 4,
    },
    "char": {
        "block_counts": [20, 50, 100, 300, 1000],
        "noise_levels": [0.0, 0.2, 0.4],
        "depths": [0.5],
        "repeats": 4,
    },
    "hard": {
        "block_counts": [20, 50, 100, 300, 1000],
        "noise_levels": [0.0, 0.2, 0.4],
        "depths": [0.5],
        "repeats": 4,
    },
    "grid": {
        "block_counts": [50, 100],
        "noise_levels": [0.0, 0.2, 0.4],
        "image_levels": [0.0, 0.2, 0.4],
        "depths": [0.5],
        "repeats": 5,
    },
}

#: 快速冒烟模式（用于本地/服务器上先验证流水线能跑通）
SMOKE_OVERRIDE: dict[str, Any] = {
    "block_counts": [20, 50],
    "noise_levels": [0.0, 0.40],
    "depths": [0.5],
    "repeats": 1,
}

#: 端到端冒烟专用的极小网格：只需要 20*5 = 100 条语料就能跑
E2E_OVERRIDE: dict[str, Any] = {
    "block_counts": [20],
    "noise_levels": [0.0, 0.40],
    "depths": [0.5],
    "repeats": 1,
}


# --------------------------------------------------------------------------- #
# 2. 噪声家族（E8：把"训练见过的噪声"与"没见过的真实退化"分开）
# --------------------------------------------------------------------------- #
#: seen   —— 与训练分布一致的字符级合成噪声
#: unseen —— 训练从未出现的扰动，用来测泛化（审稿人 1 第 2 条、审稿人 2 第 3 条）
CORRUPTION_FAMILIES: dict[str, list[str]] = {
    "clean": [],
    "seen": ["homoglyph", "gibberish", "occlusion"],
    # TMM 版新增：训练里从未出现的"语义保持"内容噪声（同义替换/丢词）
    "paraphrase": ["synonym_swap", "word_drop"],
    "unseen_char": ["keyboard_typo", "deletion", "transposition", "insertion"],
    "unseen_image": ["gaussian_blur", "jpeg", "rotation", "low_resolution", "shadow"],
    # TMM 版新增：真实扫描件退化（透视/折痕/透印/摩尔纹/复印/运动模糊/椒盐/非均匀光照）
    "real_scan": ["perspective", "fold", "show_through", "moire",
                  "photocopy", "motion_blur", "salt_pepper", "uneven_light"],
    # 全部成像退化（温和 + 真实）
    "scan_all": ["gaussian_blur", "jpeg", "rotation", "low_resolution", "shadow",
                 "perspective", "fold", "show_through", "moire",
                 "photocopy", "motion_blur", "salt_pepper", "uneven_light"],
    "seen_char_unseen": ["homoglyph", "gibberish", "occlusion",
                         "keyboard_typo", "deletion", "transposition", "insertion"],
    # 2D 热图专用：内容轴（训练见过的字符替换）× 成像轴（温和退化）
    "seen_image": ["homoglyph", "gibberish", "occlusion",
                   "gaussian_blur", "jpeg", "rotation", "low_resolution", "shadow"],
    "unseen_all": [
        "keyboard_typo", "deletion", "transposition", "insertion",
        "gaussian_blur", "jpeg", "rotation", "low_resolution", "shadow",
    ],
    # 论文主实验用的最狠一档：内容错字 + 全部成像退化
    "everything": [
        "homoglyph", "gibberish", "occlusion",
        "keyboard_typo", "deletion", "transposition", "insertion",
        "synonym_swap", "word_drop",
        "gaussian_blur", "jpeg", "rotation", "low_resolution", "shadow",
        "perspective", "fold", "show_through", "moire",
        "photocopy", "motion_blur", "salt_pepper", "uneven_light",
    ],
}


# --------------------------------------------------------------------------- #
# 3. 训练任务
# --------------------------------------------------------------------------- #
@dataclass
class TrainSpec:
    """一个训练 run 的全部超参。"""

    name: str
    kind: str = "ravt"                 # ravt | text_encoder
    seed: int = 0
    # --- RAVT 结构 ---
    patch_size: int = 14               # E0b: ViT-L/14 应为 14（旧代码写死 16）
    row_height: int = 44
    n_rows: int = 5
    use_fusion: bool = True
    use_gated_proj: bool = True
    use_slicing: bool = True
    use_buckets: bool = True
    use_noise: bool = True
    use_local_loss: bool = True        # E3: 单独消融 L_seq
    use_margin_loss: bool = True       # E3: 单独消融 L_margin
    #: 训练时是否加入**图像级**退化（模糊/JPEG/低分辨率/旋转/光照）。
    #: 现有模型只见过字符替换噪声 + 干净渲染，因此在图像退化测试集上属于配置不匹配。
    #: 走"扫描件"路线时必须打开（重训）。
    image_noise: bool = False
    image_kinds: tuple = ("gaussian_blur", "jpeg", "rotation",
                          "low_resolution", "shadow")
    #: 成像退化强度的上限；None 表示沿用 max_noise（老行为）。
    #: 单独给一个值时，字符错误率与成像强度就是两条独立的课程。
    image_max_severity: float | None = None
    #: 成像退化覆盖率；None 表示覆盖率与强度同值（老行为）。
    #: 给 0.5 表示"一半的块被退化，但退化本身不减轻"。
    image_coverage: float | None = None
    #: 成像课程形状：linear（随 epoch 线性升）| constant | step（前 1/4 轮不退化）
    image_curriculum: str = "linear"
    loss_weight_local: float = 1.0
    loss_weight_margin: float = 0.5
    margin: float = 0.15
    # --- 优化 ---
    epochs: int = 15
    batch_size: int = 128
    lr_backbone: float = 1e-7
    lr_head: float = 3e-5
    weight_decay: float = 0.05
    max_noise: float = 0.30
    # --- text_encoder（E2 归因实验）---
    text_backbone: str = "BAAI/bge-m3"
    text_pooling: str = "cls"
    text_max_length: int = 512
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def build_train_matrix(seeds: list[int], patch_sizes: list[int]) -> list[TrainSpec]:
    """完整训练矩阵：主模型 + 8 项消融（含 L_seq / L_margin）+ 两种 patch 映射对照。"""
    specs: list[TrainSpec] = []

    # (1) 主模型：每个种子一个
    for s in seeds:
        for ps in patch_sizes:
            specs.append(TrainSpec(name=f"full_p{ps}_s{s}", seed=s, patch_size=ps))

    # (2) 结构消融（E0b / E3 / 原 Table 4）
    ablation_flags = {
        "no_fusion": dict(use_fusion=False),
        "no_slicing": dict(use_slicing=False),
        "no_gated_proj": dict(use_gated_proj=False),
        "no_buckets": dict(use_buckets=False),
        "no_noise": dict(use_noise=False),
        "no_local_loss": dict(use_local_loss=False),      # 原论文缺失
        "no_margin_loss": dict(use_margin_loss=False),    # 原论文缺失
        "no_local_no_margin": dict(use_local_loss=False, use_margin_loss=False),
    }
    for ps in patch_sizes:
        for tag, flags in ablation_flags.items():
            for s in seeds[:2]:  # 消融跑 2 个种子即可
                specs.append(TrainSpec(name=f"{tag}_p{ps}_s{s}", seed=s, patch_size=ps, **flags))

    # (3) 文本编码器的抗噪微调（E2 的关键对照：抽掉"渲染成图"，只保留噪声课程）
    for s in seeds[:2]:
        for tag, noise_on, buckets in [("typo", True, True), ("clean", False, True)]:
            specs.append(TrainSpec(
                name=f"text_bge_{tag}_s{s}", kind="text_encoder", seed=s,
                text_backbone="BAAI/bge-m3",
                use_noise=noise_on, use_buckets=buckets,
                epochs=8, batch_size=64, lr_head=2e-5, max_noise=0.30,
            ))
    return specs


# --------------------------------------------------------------------------- #
# 4. 评测任务
# --------------------------------------------------------------------------- #
@dataclass
class EvalSpec:
    """一个待评测的检索器。"""

    name: str
    kind: str                       # ravt | bm25 | text_encoder | image_encoder
    checkpoint: str | None = None
    run_id: str | None = None       # 对应训练 run，用于追溯
    patch_size: int = 14
    n_rows: int = 5
    row_height: int = 44
    aggregation: str = "max_row"    # max_row | late_interaction（E3 的修复尝试）
    tfidf: bool = False             # bm25 变体的 tf-idf
    tokenizer: str = "word"         # word | char3 | char5 | denoise
    model_id: str = ""              # HF 模型名
    pooling: str = "cls"
    batch_size: int = 64
    dtype: str = "float32"
    needs_gpu: bool = True
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def build_eval_matrix(ravt_checkpoints: dict[str, str]) -> list[EvalSpec]:
    """评测矩阵。

    ravt_checkpoints: {训练 run 名 -> checkpoint 路径}，由 stage_train 产出后回填。
    """
    specs: list[EvalSpec] = []

    # 区分两类 checkpoint：RAVT 的是 .pth，文本编码器的是一个目录（含 config.json）
    ravt_runs = {k: v for k, v in ravt_checkpoints.items() if not k.startswith("text_")}
    text_runs = {k: v for k, v in ravt_checkpoints.items() if k.startswith("text_")}

    # --- 我们的方法（多个种子 / 多个 patch 映射 / 两种聚合方式）---
    #     只有 full_* 进这一组；消融配置统一走下面的 ABL-* 命名，避免重复评测
    for run_name, ckpt in ravt_runs.items():
        if not run_name.startswith("full_"):
            continue
        ps = _parse_patch_size(run_name)
        short = run_name.replace("full_", "")
        specs.append(EvalSpec(name=f"RAVT-{short}", kind="ravt", checkpoint=ckpt,
                              run_id=run_name, patch_size=ps, aggregation="max_row"))
        specs.append(EvalSpec(name=f"RAVT-LI-{short}", kind="ravt", checkpoint=ckpt,
                              run_id=run_name, patch_size=ps, aggregation="late_interaction"))

    # --- 结构消融的评测（复用同一套评测代码）---
    for run_name, ckpt in ravt_runs.items():
        if run_name.startswith("full_"):
            continue
        ps = _parse_patch_size(run_name)
        specs.append(EvalSpec(name=f"ABL-{run_name}", kind="ravt", checkpoint=ckpt,
                              run_id=run_name, patch_size=ps, aggregation="max_row"))

    # --- 文本基线 ---
    specs += [
        EvalSpec(name="BM25-word", kind="bm25", tokenizer="word", needs_gpu=False),
        EvalSpec(name="BM25-char3", kind="bm25", tokenizer="char3", needs_gpu=False),
        EvalSpec(name="BM25-char5", kind="bm25", tokenizer="char5", needs_gpu=False),
        EvalSpec(name="BM25-denoise", kind="bm25", tokenizer="denoise", needs_gpu=False),
        EvalSpec(name="BGE-M3", kind="text_encoder", model_id="BAAI/bge-m3", pooling="cls",
                 batch_size=64),
    ]

    # --- 视觉基线 ---
    specs += [
        EvalSpec(name="ColPali", kind="image_encoder", model_id="vidore/colpali-v1.2",
                 batch_size=4, dtype="bfloat16"),
        EvalSpec(name="JinaCLIP-v2", kind="image_encoder", model_id="jinaai/jina-clip-v2",
                 batch_size=32),
    ]

    # --- E2 的关键对照：带噪声课程微调的文本编码器（由 stage_train 产出后追加）---
    for run_name, ckpt in text_runs.items():
        specs.append(EvalSpec(name=f"TXT-{run_name}", kind="text_encoder",
                              checkpoint=ckpt, run_id=run_name,
                              model_id="BAAI/bge-m3", pooling="cls", batch_size=64,
                              needs_gpu=True))
    return specs


def _parse_patch_size(run_name: str) -> int:
    for token in run_name.split("_"):
        if token.startswith("p") and token[1:].isdigit():
            return int(token[1:])
    return 14


QUICK_FILTER: dict[str, Any] = {
    "max_tasks": 6,
    "block_counts": [20, 50],
    "noise_levels": [0.0, 0.40],
    "models": ["BM25-word", "BM25-char3", "BM25-denoise"],
}
