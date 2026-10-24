"""阶段 1：数据与评测集的准备。

三件事：

1. `extract`  —— 解压既有的 Visual-NIAH（复现旧结果时必须用这一份，见下）
2. `benchmark` —— 生成新评测集。支持：
       - v1_legacy：复现旧网格（needle 恒在第 0 行、干扰项随机分配）
       - v2_fixed ：修复两个构造缺陷（needle 全行覆盖、干扰项确定性分配、repeats=20）
       - --corruptions unseen_*：换成训练没见过的噪声家族（E8 的跨域泛化）
3. `train-corpus` —— 渲染 wikitext-103 训练块

为什么既有评测集不能"重新生成一遍就好"：
    原 `build_niah_benchmark.py` 里的 random 调用**没有固定种子**，
    重新生成会得到不同的任务内容，而 `results/main/*.csv` 是按 Task_ID 对齐的，
    两者一旦错位，所有历史对比都会失效。所以复现旧结果必须用 zip 里那一份。
"""

from __future__ import annotations

import json
import multiprocessing as mp
import random
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from schemas import SCHEMA_DATA
from common import JsonlWriter, ensure_dir, now_iso, repo_path, seed_everything
from config import BENCHMARKS, CORRUPTION_FAMILIES, SMOKE_OVERRIDE
from corpus import CorpusSpec, iter_texts, load_texts, print_report
from model_defs import BlockRenderer, CorruptionEngine, RendererConfig, build_mosaic

DEFAULT_ZIP = "data/archives/niah_robust_test.zip"
DEFAULT_BENCH_ROOT = "data/niah_robust_test"
DEFAULT_TRAIN_BLOCKS = "data/processed/train_blocks"

#: 渲染参数可以通过环境变量统一控制（训练语料与测试集必须用同一套参数）
#:   RAVT_FONT       : 指定字体文件；不设则按候选列表自动查找
#:   RAVT_WRAP_MODE  : fast（默认，快 8.7 倍）| exact（与老代码逐像素一致）
#:   RAVT_BLOCK_HEIGHT: 行高（默认 44）。设成 42（= 3 × patch 14）可让每个行窗口
#:                    与行边界**逐像素对齐**，消除 44px 下"窗口混入 27% 邻行"的近似；
#:                    训练/评测必须用同一个值（模型的行窗口由它决定）。
#:   RAVT_FONT_SIZE  : 字号（默认 11），只影响观感，训练评测不必一致。
def renderer_from_env() -> BlockRenderer:
    import os
    cfg = RendererConfig(font_path=os.environ.get("RAVT_FONT") or None,
                         wrap_mode=os.environ.get("RAVT_WRAP_MODE", "fast"),
                         block_height=int(os.environ.get("RAVT_BLOCK_HEIGHT", 44)),
                         font_size=int(os.environ.get("RAVT_FONT_SIZE", 11)))
    return BlockRenderer(cfg)


def _renderer_record() -> dict:
    """把渲染参数写进 JSONL，便于事后确认"训练语料与测试集用的是同一套渲染"。

    字体是关键实验变量：老代码在找不到 liberation 时会退回 PIL 默认位图字体，
    与使用 DejaVu/微软雅黑的效果差别很大，必须留痕。
    """
    r = renderer_from_env()
    return {"font": r.font_path, "wrap_mode": r.cfg.wrap_mode, "font_size": r.cfg.font_size,
            "block": [r.cfg.block_width, r.cfg.block_height],
            "canvas_size": r.cfg.canvas_size,
            "margins": [r.cfg.margin_x, r.cfg.margin_y]}

#: 默认语料 = 服务器上的 arrow 数据集目录（dataset/data-00000-of-00010.arrow 等）
#: 训练默认只取**分片 0**，分片 1 留给测试集，避免训练/测试重叠（论文里的 held-out 才成立）
DEFAULT_CORPUS = CorpusSpec(path="dataset", min_len=15, shard_ids=[0])
DEFAULT_TEST_CORPUS = CorpusSpec(path="dataset", min_len=15, shard_ids=[1])


def as_spec(corpus) -> CorpusSpec:
    """允许传 CorpusSpec / 字符串路径 / None（用默认）。"""
    if corpus is None:
        return DEFAULT_CORPUS
    if isinstance(corpus, CorpusSpec):
        return corpus
    return CorpusSpec(path=str(corpus), min_len=DEFAULT_CORPUS.min_len,
                      shard_ids=DEFAULT_CORPUS.shard_ids)

#: 深度分层：用三分位而不是原来的固定点，这样 needle 的行号能覆盖 0~4
DEPTH_BANDS = {"0.1": (0.00, 1 / 3), "0.5": (1 / 3, 2 / 3), "0.9": (2 / 3, 1.00)}

#: 采样池上限（与原脚本一致，避免把 180 万行全塞进内存）
POOL_LIMIT = 200_000

#: 多进程渲染：每个 worker 只建一次渲染器
_WORKER_STATE: dict = {}


def _worker_renderer() -> BlockRenderer:
    if "renderer" not in _WORKER_STATE:
        _WORKER_STATE["renderer"] = renderer_from_env()
    return _WORKER_STATE["renderer"]


def _render_task(job: dict) -> dict:
    """渲染一个 task 的全部块并写盘，返回元数据。

    会被多进程并发调用，因此所有随机性都必须来自 job['rng_seed']，
    不能碰全局 random 状态——否则串行与并行的结果会不一致。
    """
    renderer = _worker_renderer()
    engine = CorruptionEngine(random.Random(job["rng_seed"]))
    task_dir = Path(job["task_dir"])
    img_dir = ensure_dir(task_dir / "visual_haystack")
    txt_dir = ensure_dir(task_dir / "text_haystack")

    full_context = []
    for b_idx, rows in enumerate(job["block_rows"]):
        mosaic, rendered_rows = build_mosaic(
            rows, renderer, engine,
            noise_rate=job["noise"],
            char_families=tuple(job["families"]),
            word_families=tuple(job["families"]),
            image_families=tuple(job["families"]),
            # 成像轴与内容轴分开：image_level 是成像退化强度，
            # image_scale 用于整体放大/缩小退化（--hard-scales）
            image_severity=job.get("image_level", job["noise"]) * job.get("image_scale", 1.0),
            image_coverage=job.get("image_coverage"),
        )
        mosaic.save(img_dir / f"block_{b_idx:04d}.png")
        (txt_dir / f"block_{b_idx:04d}.txt").write_text("\n".join(rendered_rows), encoding="utf-8")
        full_context.extend(rendered_rows)

    (task_dir / "full_context.txt").write_text("\n".join(full_context), encoding="utf-8")
    (task_dir / "metadata.json").write_text(
        json.dumps(job["meta"], indent=4, ensure_ascii=False), encoding="utf-8")
    return job["meta"]



# --------------------------------------------------------------------------- #
def extract_benchmark(zip_path: str = DEFAULT_ZIP, out_dir: str = "data") -> Path:
    zip_path = repo_path(zip_path) if not Path(zip_path).is_absolute() else Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"找不到 {zip_path}")
    out = repo_path(out_dir) if not Path(out_dir).is_absolute() else Path(out_dir)
    ensure_dir(out)
    print(f"解压 {zip_path} -> {out}（约 3.5 GB，请耐心等待）")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out)
    target = Path(out) / "niah_robust_test"
    n = len([d for d in target.iterdir() if d.is_dir()]) if target.exists() else 0
    print(f"解压完成，共 {n} 个 task")
    return target


# --------------------------------------------------------------------------- #
def build_benchmark(bench_name: str = "v2_fixed", out_root: str = DEFAULT_BENCH_ROOT,
                    corpus=None, corruptions: str = "seen", quick: bool = False,
                    workers: int = 1, grid_override: dict | None = None,
                    pool_limit: int = POOL_LIMIT,
                    overwrite: bool = False,
                    image_scale: float = 1.0,
                    log_path: str = "results/jsonl/data/benchmark_build.jsonl") -> Path:
    """生成 Visual-NIAH 评测集。"""
    spec = as_spec(corpus)
    grid = dict(BENCHMARKS[bench_name])
    if quick:
        grid.update(SMOKE_OVERRIDE)
    if grid_override:
        grid.update(grid_override)

    seed = int(grid["seed"])
    seed_everything(seed)
    rng = random.Random(seed)

    families = CORRUPTION_FAMILIES.get(corruptions, [])
    out_root = repo_path(out_root) if not Path(out_root).is_absolute() else Path(out_root)
    ensure_dir(out_root)

    # 目录里已有 task 时先处理掉：任务目录名只含 B/N/D 而不含重复序号，
    # 用不同的 repeats 重建会让一部分旧任务残留下来，评测集就混了两次构建的内容。
    import shutil
    existing = [d for d in out_root.iterdir()
                if d.is_dir() and (d / "metadata.json").exists()]
    if existing:
        if not overwrite:
            raise RuntimeError(
                f"{out_root} 里已有 {len(existing)} 个 task，直接重建会与旧任务混在一起"
                "（任务目录名不含重复序号，用不同 repeats 重建时旧任务不会被覆盖）。\n"
                "请二选一：\n"
                f"  ① 先清空：rm -rf {out_root}\n"
                "  ② 或者显式加 --overwrite（会自动删除已有 task 后重建）")
        for d in existing:
            shutil.rmtree(d)
        print(f"已清空 {out_root} 下 {len(existing)} 个已有 task（--overwrite）")

    print("读取语料 ...")
    texts = load_texts(spec, limit=pool_limit)
    print(f"  采样池: {len(texts)} 条文本")
    if len(texts) < max(grid["block_counts"]) * 5:
        raise RuntimeError(
            f"语料太少（{len(texts)} 条），不足以构造 {max(grid['block_counts'])} 块的 haystack。"
            "请检查 --corpus 路径与分片是否完整。")

    # 内容轴（noise_levels = 字符错误率）与成像轴（image_levels = 成像退化强度）
    # 是两条独立的轴：给了 image_levels 就做笛卡尔积，用来建 2D 热图网格。
    image_levels = list(grid.get("image_levels") or [])
    if image_levels:
        combos = [(b, n, i, d) for b in grid["block_counts"] for n in grid["noise_levels"]
                  for i in image_levels for d in grid["depths"]]
    else:
        combos = [(b, n, n, d) for b in grid["block_counts"] for n in grid["noise_levels"]
                  for d in grid["depths"]]
    image_coverage = grid.get("image_coverage")
    total = len(combos) * int(grid["repeats"])
    max_blocks = max(grid["block_counts"])
    pool_needed = max_blocks * 5 * 4
    if len(texts) < pool_needed:
        texts = texts * (pool_needed // len(texts) + 1)

    print(f"生成评测集 {bench_name} | 组合 {len(combos)} x 重复 {grid['repeats']} = {total} 个 task")
    print(f"  needle 定位: {grid['target_position']} | 干扰项: {grid['distractor_rule']} "
          f"| 噪声家族: {corruptions} {families}")
    if image_levels:
        print(f"  内容轴 noise_levels={grid['noise_levels']} × 成像轴 image_levels={image_levels}"
              f" | 成像覆盖率 coverage={image_coverage if image_coverage is not None else '=severity'}")
    print(f"  并发进程: {workers}")

    task_id = 0
    created = []

    def job_iter():
        """惰性生成 job（内存里只保留一个 task 的文本，避免 1.25M 字符串常驻）。"""
        nonlocal task_id
        for b_count, noise, image_level, depth in combos:
            for _ in range(int(grid["repeats"])):
                haystack = rng.sample(texts, b_count * 5)
                total_rows = b_count * 5

                if grid["target_position"] == "random":
                    lo, hi = DEPTH_BANDS[str(depth)]
                    lo_i = int(lo * total_rows)
                    hi_i = max(lo_i + 1, int(hi * total_rows))
                    target_idx = rng.randrange(lo_i, min(hi_i, total_rows))
                else:  # depth_formula：原论文行为（实测 target_row 恒为 0）
                    target_idx = int(total_rows * depth)
                target_block = target_idx // 5
                target_row = target_idx % 5
                raw_needle = haystack[target_idx]
                needle_query = renderer_from_env().render(raw_needle)[1]

                if grid["distractor_rule"] == "deterministic":
                    has_distractor = (task_id % 2 == 0) and b_count > 1
                else:
                    has_distractor = rng.choice([True, False]) and b_count > 1

                dist_block, dist_row, distractor_text = -1, -1, ""
                if has_distractor:
                    cands = [i for i in range(b_count) if i != target_block]
                    dist_block = rng.choice(cands)
                    dist_row = rng.randrange(5)
                    words = raw_needle.split()
                    rng.shuffle(words)
                    distractor_text = " ".join(words)

                block_rows = []
                for b_idx in range(b_count):
                    rows = []
                    for r_idx in range(5):
                        flat = b_idx * 5 + r_idx
                        if has_distractor and b_idx == dist_block and r_idx == dist_row:
                            rows.append(distractor_text)
                        else:
                            rows.append(haystack[flat])
                    block_rows.append(rows)

                tag = f"_C{corruptions}" if corruptions != "seen" else ""
                img_tag = (f"_I{int(round(image_level * 100))}" if image_levels else "")
                task_dir = out_root / (f"Task_{task_id:04d}_B{b_count}"
                                       f"_N{int(round(noise * 100))}{img_tag}"
                                       f"_D{int(depth * 100)}{tag}")
                meta = {
                    "task_id": task_id, "block_count": b_count, "noise_level": noise,
                    # 只有这个网格**真的有**成像轴时才写 image_level；
                    # 否则写成 None，免得下游把 noise 误当成独立的成像强度
                    # （那会让汇总表多出一张和噪声档一模一样的表）。
                    "image_level": (image_level if image_levels else None),
                    "image_coverage": (image_coverage if image_coverage is not None
                                       else image_level),
                    "depth": depth,                    # 名义分层，保持与旧表兼容
                    "depth_actual": target_idx / total_rows,
                    "target_idx": target_idx,
                    "target_block_idx": target_block,
                    "target_row_idx": target_row,
                    "needle_query": needle_query,
                    "has_distractor": has_distractor,
                    "distractor_block_idx": dist_block,
                    "distractor_row_idx": dist_row,
                    "corruptions": corruptions,
                    "corruption_families": families,
                    "benchmark": bench_name, "seed": seed,
                    "render_config": "aligned_natural_v2_distractor_with_text",
                }
                yield {"task_dir": str(task_dir), "block_rows": block_rows,
                       "noise": noise, "image_level": image_level,
                       "image_coverage": (image_coverage if image_coverage is not None
                                          else image_level),
                       "families": families, "meta": meta,
                       "image_scale": image_scale,
                       "rng_seed": seed * 1_000_003 + task_id}
                task_id += 1

    # ---- 串行或并行渲染 ----
    if workers and workers > 1:
        with mp.Pool(processes=workers) as pool:
            for i, meta in enumerate(pool.imap_unordered(_render_task, job_iter(), chunksize=1), 1):
                created.append(meta)
                if i % 100 == 0:
                    print(f"  已完成 {i}/{total}")
    else:
        for i, job in enumerate(job_iter(), 1):
            created.append(_render_task(job))
            if i % 100 == 0:
                print(f"  已完成 {i}/{total}")

    created.sort(key=lambda m: m["task_id"])

    # ---- 记录构建清单（分析脚本据此确认数据来源）----
    with JsonlWriter(repo_path(log_path)) as w:
        w.write({
            "record": "benchmark_build",
            "schema": SCHEMA_DATA,
            "timestamp": now_iso(),
            "benchmark": bench_name, "corruptions": corruptions, "seed": seed,
            "image_scale": image_scale,
            "corpus": spec.to_dict(),          # 记录用了哪些分片（provenance）
            "grid": {k: grid[k] for k in ("block_counts", "noise_levels", "depths", "repeats")},
            "image_levels": image_levels or None,
            "image_coverage": image_coverage,
            "n_tasks": len(created),
            "out_root": str(out_root),
            "renderer": _renderer_record(),
            "target_row_hist": {str(r): sum(1 for c in created if c["target_row_idx"] == r)
                                for r in range(5)},
            "n_blocks_total": int(sum(c["block_count"] for c in created)),
        })

    rows = [c["target_row_idx"] for c in created]
    hist = {r: rows.count(r) for r in range(5)}
    print(f"完成：{len(created)} 个 task -> {out_root}")
    print(f"  needle 行号分布: {hist}")
    if len([v for v in hist.values() if v > 0]) == 1:
        print("  ⚠️ needle 只落在一行上——这是旧版 benchmark 的已知缺陷")
    return out_root


# --------------------------------------------------------------------------- #
def build_train_corpus(out_dir: str = DEFAULT_TRAIN_BLOCKS, corpus=None,
                       max_records: int = 50000,
                       log_path: str = "results/jsonl/data/train_corpus.jsonl") -> Path:
    """把 wikitext-103 渲染成 224x44 的训练块（png + txt）。"""
    spec = as_spec(corpus)
    out = repo_path(out_dir) if not Path(out_dir).is_absolute() else Path(out_dir)
    ensure_dir(out)
    renderer = renderer_from_env()
    print(f"渲染参数: {renderer.describe()}")
    written = 0
    lengths = []
    for text in iter_texts(spec):
        if written >= max_records:
            break
        img, actual = renderer.render(text)
        if len(actual.strip()) < spec.min_len:
            continue
        name = f"L{len(actual)}_{written:06d}"
        img.save(out / f"{name}.png")
        (out / f"{name}.txt").write_text(actual, encoding="utf-8")
        lengths.append(len(actual))
        written += 1
        if written % 5000 == 0:
            print(f"  已生成 {written} 对")

    with JsonlWriter(repo_path(log_path)) as w:
        w.write({"record": "train_corpus", "schema": SCHEMA_DATA, "timestamp": now_iso(),
                 "out_dir": str(out), "n_blocks": written,
                 "corpus": spec.to_dict(),      # 记录用了哪些分片（provenance）
                 "renderer": _renderer_record(),
                 "length_mean": float(np.mean(lengths)) if lengths else 0.0,
                 "length_min": int(min(lengths)) if lengths else 0,
                 "length_max": int(max(lengths)) if lengths else 0})
    print(f"训练块生成完成：{written} 对 -> {out}")
    return out
