"""Visual-NIAH 任务读取与筛选。"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass
class NiahTask:
    task_id: int
    path: Path
    meta: dict
    image_paths: list[Path]
    text_paths: list[Path]

    # ---- 便捷访问 ----
    @property
    def query(self) -> str:
        return self.meta["needle_query"]

    @property
    def gt_block(self) -> int:
        return int(self.meta.get("target_block_idx", 0))

    @property
    def gt_row(self) -> int:
        return int(self.meta.get("target_row_idx", -1))

    @property
    def has_distractor(self) -> bool:
        return bool(self.meta.get("has_distractor", False))

    @property
    def distractor_block(self) -> int:
        return int(self.meta.get("distractor_block_idx", -1))

    @property
    def blocks(self) -> int:
        return int(self.meta.get("block_count", len(self.image_paths)))

    @property
    def noise(self) -> float:
        return float(self.meta.get("noise_level", -1.0))

    @property
    def image_level(self) -> float:
        """成像退化强度。没有独立成像轴的测试集 -> 退回 noise_level（覆盖率=强度）。

        注意：字段存在但值为 None（新代码对"没有成像轴"的测试集就是这么写的）
        也要退回，否则 float(None) 会直接抛异常。
        """
        v = self.meta.get("image_level")
        if v is None:
            v = self.meta.get("noise_level", -1.0)
        return float(v)

    @property
    def depth(self) -> float:
        return float(self.meta.get("depth", -1.0))

    def load_images(self) -> list[Image.Image]:
        imgs = []
        for p in self.image_paths:
            with Image.open(p) as im:
                imgs.append(im.convert("RGB"))
        return imgs

    def load_texts(self) -> list[str]:
        out = []
        for p in self.text_paths:
            try:
                out.append(p.read_text(encoding="utf-8"))
            except Exception:
                out.append("")
        return out

    def full_context(self) -> str:
        path = self.path / "full_context.txt"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return "\n".join(self.load_texts())


def load_task(task_dir: Path) -> NiahTask:
    task_dir = Path(task_dir)
    with open(task_dir / "metadata.json", "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    img_dir = task_dir / "visual_haystack"
    txt_dir = task_dir / "text_haystack"
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() == ".png") if img_dir.exists() else []
    texts = sorted(p for p in txt_dir.iterdir() if p.suffix.lower() == ".txt") if txt_dir.exists() else []
    return NiahTask(
        task_id=int(meta.get("task_id", -1)),
        path=task_dir,
        meta=meta,
        image_paths=images,
        text_paths=texts,
    )


def list_tasks(root: Path, *, block_counts=None, noise_levels=None, depths=None,
               image_levels=None,
               max_tasks: int | None = None, seed: int = 0,
               strict: bool = True) -> list[NiahTask]:
    """列出任务，可按网格维度筛选。

    strict=True 时会立刻读取 metadata 并校验维度，便于早失败；
    max_tasks 用固定种子抽样，保证多次运行选到同一批任务。
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"找不到评测集目录 {root}。请先从 data/archives/niah_robust_test.zip 解压，"
            "或用 stage_data 重新生成。"
        )
    dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not strict:
        return [NiahTask(-1, d, {}, [], []) for d in dirs]

    tasks = []
    for d in dirs:
        if not (d / "metadata.json").exists():
            continue
        t = load_task(d)
        if block_counts and t.blocks not in block_counts:
            continue
        if noise_levels and round(t.noise, 4) not in {round(n, 4) for n in noise_levels}:
            continue
        if image_levels and round(t.image_level, 4) not in {round(n, 4) for n in image_levels}:
            continue
        if depths and round(t.depth, 4) not in {round(x, 4) for x in depths}:
            continue
        tasks.append(t)

    if max_tasks is not None and len(tasks) > max_tasks:
        rng = random.Random(seed)
        # 分层抽样：先按 (blocks, noise) 分组各取一点，保证覆盖度
        groups: dict[tuple, list[NiahTask]] = {}
        for t in tasks:
            groups.setdefault((t.blocks, t.noise, t.image_level), []).append(t)
        picked: list[NiahTask] = []
        keys = sorted(groups)
        i = 0
        while len(picked) < max_tasks and keys:
            key = keys[i % len(keys)]
            pool = groups[key]
            if pool:
                picked.append(pool.pop(rng.randrange(len(pool))))
            i += 1
            if i > 100000:
                break
        tasks = sorted(picked, key=lambda t: t.task_id)[:max_tasks]
    return tasks
