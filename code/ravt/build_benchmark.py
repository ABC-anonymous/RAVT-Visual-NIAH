"""从 arrow 语料生成训练块与测试集（分片 0 训练 / 分片 1 测试）。"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from corpus import (CorpusSpec, decide_split_mode, load_texts,  # noqa: E402
                    print_report, split_rows)
from config import TMM_GRIDS                                   # noqa: E402
from stage_data import build_benchmark, build_train_corpus  # noqa: E402

#: TMM 版主测试集网格（= config.BENCHMARKS["v3_tmm"]）：
#:   8 个块数 × 3 个噪声档 × 3 个深度 × 10 次重复 = 720 个 task
#: 相对上一版的两个变化：① 块数重新覆盖到 1000（长上下文曲线）；
#:                        ② 恢复三个 depth 分层并提高重复数（每个单元格 n=30）。
TEST_GRID = {"block_counts": [20, 50, 100, 150, 200, 300, 500, 1000],
             "noise_levels": [0.0, 0.20, 0.40],
             "depths": [0.1, 0.5, 0.9],
             "repeats": 10}

#: 老评测集风格（needle 恒在第 0 行、干扰项随机分配），
#: 只用于和旧论文的 73.33%@0% 噪声做"同口径"对照。
LEGACY_GRID = {"block_counts": [20, 50, 100, 150, 200, 300, 500, 1000],
               "noise_levels": [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
               "depths": [0.1, 0.5, 0.9],
               "repeats": 5}

#: 补充测试轴（TMM 版新增，与主测试集共用同一份测试语料与渲染器）：
#:   image —— 纯成像退化 5 档（文本干净，OCR 流水线与视觉路线公平对撞）
#:   real  —— 真实扫描件退化（透视/折痕/透印/摩尔纹/复印/椒盐/非均匀光照）
#:   char  —— 训练没见过的字符错误（键盘错字/删除/换位/插入）
#:   hard  —— 内容错字 + 成像退化叠加
#:   grid  —— 2D 热图：内容错字率 × 成像退化强度
EXTRA_BENCHES = {
    "image": ("unseen_image", "data/test_image"),
    "real": ("real_scan", "data/test_real"),
    "char": ("unseen_char", "data/test_char"),
    "hard": ("unseen_all", "data/test_hard"),
    "grid": ("seen_image", "data/test_grid"),
}


def _label(spec) -> str:
    """打印用的切分说明。"""
    if spec.shard_ids:
        return f"分片 {spec.shard_ids}"
    if spec.split:
        return f"行级切分 {spec.split}（每 {spec.test_every} 行取 1）"
    return "全部"


def main():
    ap = argparse.ArgumentParser(description="生成训练块与测试集")
    ap.add_argument("--corpus", default=None,
                    help="语料路径（目录 / *.parquet / *.arrow / *.jsonl）。"
                         "缺省时自动找：dataset/ 目录 > 基准目录下的 *.parquet > *.arrow")
    ap.add_argument("--test-corpus", default=None,
                    help="测试语料单独指定（例如官方 wikitext-103 的 test split）。"
                         "缺省时测试集从 --corpus 里切分出来")
    ap.add_argument("--text-column", default=None, help="文本列名（默认自动识别）")
    ap.add_argument("--train-shards", type=int, nargs="+", default=[0])
    ap.add_argument("--test-shards", type=int, nargs="+", default=[1],
                    help="分片切分模式下的测试分片（默认 [1]）")
    ap.add_argument("--split-mode", default="auto", choices=["auto", "shards", "rows"],
                    help="auto=有多个分片就按分片切，否则按行切（默认）")
    ap.add_argument("--test-every", type=int, default=10,
                    help="行级切分时每 N 行取 1 行作测试（默认 10，即 10%%）")
    ap.add_argument("--min-len", type=int, default=15)
    ap.add_argument("--records", type=int, default=20000, help="渲染多少个训练块")
    ap.add_argument("--train-blocks", default="data/processed/train_blocks")
    ap.add_argument("--test-bench", default="data/test_bench")
    ap.add_argument("--test-tasks", type=int, default=None,
                    help="限制测试集任务数（默认用 TEST_GRID 的 18 个）")
    ap.add_argument("--grid", default="tmm", choices=["tmm", "fixed", "legacy"],
                    help="tmm/fixed=needle 覆盖 0~4 行、块数 20~1000（默认）；"
                         "legacy=needle 恒在第 0 行，用于与旧论文结果同口径对照")
    ap.add_argument("--extra-benches", default="",
                    help="顺带生成的补充测试轴（逗号分隔）：image,real,char,hard,grid。"
                         "留空只生成主测试集。")
    ap.add_argument("--no-cache", action="store_true", help="不复用 arrow->jsonl 缓存")
    ap.add_argument("--overwrite", action="store_true",
                    help="测试集目录已存在时先清空再重建（否则直接报错，"
                         "避免两次构建的任务混在一起）")
    ap.add_argument("--base-dir", default=None,
                    help="基准目录（放 dataset/ 与 clip_local_weights_224/ 的地方）。"
                         "缺省时自动在脚本目录及其上 3 层里查找")
    args = ap.parse_args()

    from ravt_paths import find_corpus, setup
    base = setup(args.base_dir)
    args.corpus = find_corpus(base, args.corpus)
    if args.test_corpus:
        args.test_corpus = find_corpus(base, args.test_corpus)

    mode = args.split_mode
    if mode == "auto":
        mode = decide_split_mode(args.corpus)
    common = dict(path=args.corpus, text_column=args.text_column,
                  min_len=args.min_len, use_cache=not args.no_cache)
    if args.test_corpus:
        # 训练/测试用**不同文件**（最干净：如 wikitext train 训练、官方 test split 评测）
        train_spec = CorpusSpec(**common)
        test_spec = CorpusSpec(**{**common, "path": args.test_corpus})
        print(f"切分方式: 独立文件 —— 训练 {args.corpus} / 测试 {args.test_corpus}")
    elif mode == "shards":
        train_spec = CorpusSpec(shard_ids=args.train_shards, **common)
        test_spec = CorpusSpec(shard_ids=args.test_shards, **common)
        print(f"切分方式: 按分片 —— 训练 {args.train_shards} / 测试 {args.test_shards}")
    else:
        train_spec, test_spec = split_rows(test_every=args.test_every, **common)
        print(f"切分方式: 按行 —— 每 {args.test_every} 行取 1 行作测试"
              f"（训练 {1 - 1 / args.test_every:.0%} / 测试 {1 / args.test_every:.0%}），"
              "交错取样，两边不相交")

    print("=" * 74)
    print(f"训练语料（{_label(train_spec)}）")
    print("=" * 74)
    print_report(train_spec)

    print("\n" + "=" * 74)
    print(f"测试语料（{_label(test_spec)}）")
    print("=" * 74)
    print_report(test_spec)

    print("\n" + "=" * 74)
    print(f"渲染 {args.records} 个训练块")
    print("=" * 74)
    build_train_corpus(out_dir=args.train_blocks, corpus=train_spec,
                       max_records=args.records,
                       log_path="results/train_corpus.jsonl")

    print("\n" + "=" * 74)
    print("生成测试集（只用测试分片）")
    print("=" * 74)
    grid = dict(LEGACY_GRID if args.grid == "legacy" else TEST_GRID)
    bench_name = "v1_legacy" if args.grid == "legacy" else "v3_tmm"
    if args.grid == "legacy":
        print("⚠️ legacy 网格：needle 恒在 mosaic 第 0 行 —— 只用于与旧结果对照，"
              "不要作为修改稿的主结果")
    # 测试集的每个 task 需要 block_count*5 条文本；语料不够时自动收缩块数网格
    pool = load_texts(test_spec, limit=200000)
    max_blocks = max(1, len(pool) // 5)
    keep = [b for b in grid["block_counts"] if b <= max_blocks]
    if not keep:
        keep = [max_blocks]
    if keep != grid["block_counts"]:
        print(f"⚠️ 测试语料只有 {len(pool)} 条，块数网格 {grid['block_counts']} "
              f"自动收缩为 {keep}")
        if args.grid == "legacy" and max(keep) < 20:
            print("⚠️ legacy 网格被收缩到 <20 块后，"
                  "'needle 恒在第 0 行' 这个性质就不成立了"
                  "（原性质依赖块数是 10 的倍数）。")
            print("   此时的结果不能与旧论文的 73.33%@0% 噪声对照。"
                  "请用更多测试分片或 --test-shards 指定更大的分片。")
    grid["block_counts"] = keep
    if args.test_tasks:
        per_cell = max(1, args.test_tasks // (len(grid["block_counts"]) *
                                              len(grid["noise_levels"]) * len(grid["depths"])))
        grid["repeats"] = per_cell
    build_benchmark(bench_name, out_root=args.test_bench, corpus=test_spec,
                    quick=False, workers=4, grid_override=grid, pool_limit=200000,
                    overwrite=args.overwrite,
                    log_path="results/test_bench_build.jsonl")

    wanted = [b.strip() for b in str(args.extra_benches).split(",") if b.strip()]
    unknown = [b for b in wanted if b not in EXTRA_BENCHES]
    if unknown:
        raise SystemExit(f"未知的补充测试轴 {unknown}；可选: {list(EXTRA_BENCHES)}")
    for name in wanted:
        corruptions, out_root = EXTRA_BENCHES[name]
        grid_extra = dict(TMM_GRIDS[name])
        grid_extra["block_counts"] = keep
        if args.test_tasks and "image_levels" not in grid_extra:
            grid_extra["repeats"] = max(1, args.test_tasks // max(1, (
                len(grid_extra["block_counts"]) * len(grid_extra["noise_levels"])
                * len(grid_extra["depths"]))))
        print("\n" + "=" * 74)
        print(f"生成补充测试轴 {name}（噪声族 {corruptions}）-> {out_root}")
        print("=" * 74)
        build_benchmark("v2_fixed", out_root=out_root, corpus=test_spec,
                        corruptions=corruptions, quick=False, workers=4,
                        grid_override=grid_extra, pool_limit=200000,
                        overwrite=args.overwrite,
                        log_path=f"results/{name}_bench_build.jsonl")

    print("\n✅ 数据准备完成：")
    print(f"   训练块: {args.train_blocks}")
    print(f"   测试集: {args.test_bench}")
    if wanted:
        print(f"   补充测试轴: {', '.join(wanted)}")
    print("   下一步: python train.py --dry-run   （先冒烟）")


if __name__ == "__main__":
    main()
