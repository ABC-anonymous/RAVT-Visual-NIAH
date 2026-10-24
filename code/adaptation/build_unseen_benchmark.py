"""构造 leave-one-corruption-family-out 测试集。

训练只见过 homoglyph / gibberish / occlusion；这里用未见过的 transposition /
keyboard_typo / deletion 重新渲染 5 行 mosaic，query 仍是干净原文。
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--source", default="data/test_bench_tmm_rh42")
    ap.add_argument("--out", default="data/test_bench_unseen240")
    ap.add_argument("--tasks", type=int, default=240, help="每个档位抽多少 task")
    ap.add_argument("--levels", type=float, nargs="+", default=[0.2, 0.4])
    ap.add_argument("--families", nargs="+",
                    default=["transposition", "keyboard_typo", "deletion"])
    ap.add_argument("--max-blocks", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261021)
    a = ap.parse_args()

    root = Path(a.root).resolve()
    sys.path.insert(0, str(root / "code" / "ravt"))
    from datasets_eval import list_tasks
    from model_defs import BlockRenderer, CorruptionEngine, RendererConfig, build_mosaic

    out = root / a.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    bench = root / a.source
    clean = [t for t in list_tasks(bench, strict=True)
             if round(t.noise, 2) == 0.0 and t.blocks <= a.max_blocks]
    rng = random.Random(a.seed)
    rng.shuffle(clean)
    if len(clean) < a.tasks:
        raise SystemExit(
            f"干净模板不足：需要每档 {a.tasks} 个，只有 {len(clean)} 个 "
            f"(max_blocks={a.max_blocks})。增大 --max-blocks 或减小 --tasks。"
        )
    # 各档使用完全相同的底层 task，只改变腐蚀 realization，保证逐 task 可配对。
    pool = clean[:a.tasks]
    renderer = BlockRenderer(RendererConfig(block_height=42, canvas_size=224))
    engine = CorruptionEngine(random.Random(a.seed))

    made = 0
    per_level = {lv: 0 for lv in a.levels}
    for lv in a.levels:
        for t in pool:
            if per_level[lv] >= a.tasks:
                break
            # eval_all.ensure_bench 只识别 Task_* 目录；命名必须与主 benchmark 一致。
            name = f"Task_{made:04d}_UF_N{int(lv*100):02d}"
            d = out / name
            (d / "visual_haystack").mkdir(parents=True)
            (d / "text_haystack").mkdir(parents=True)
            meta = dict(t.meta)
            meta["task_id"] = made
            meta["source_task_id"] = int(t.task_id)
            meta["source_task_dir"] = t.path.name
            meta["noise_level"] = lv
            meta["unseen_families"] = list(a.families)
            meta["corruptions"] = "unseen_content"
            meta["benchmark"] = "v3_tmm_unseen240"
            for j, (png, txt) in enumerate(zip(t.image_paths, t.text_paths)):
                lines = [x for x in txt.read_text(encoding="utf-8").splitlines() if x.strip()]
                while len(lines) < 5:
                    lines.append("")
                canvas, rendered = build_mosaic(
                    lines[:5], renderer, engine, noise_rate=lv,
                    char_families=tuple(a.families))
                stem = f"block_{j:04d}"
                canvas.save(d / "visual_haystack" / f"{stem}.png")
                (d / "text_haystack" / f"{stem}.txt").write_text(
                    "\n".join(rendered), encoding="utf-8")
            meta["block_count"] = len(t.image_paths)
            (d / "metadata.json").write_text(
                json.dumps(meta, ensure_ascii=False), encoding="utf-8")
            made += 1
            per_level[lv] += 1
        print(f"level {lv:g}: {per_level[lv]} tasks")

    print(f"共生成 {made} 个 task -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
