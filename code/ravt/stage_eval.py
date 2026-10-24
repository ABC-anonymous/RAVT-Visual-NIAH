"""阶段 3：评测 —— 唯一的数据出口，所有结果都写成 JSONL。

每个 task 一行记录（schema = ravt.eval.v1），字段设计对齐审稿人的每一条要求：

    correct / hit_distractor    -> Table 1/2/3
    noise / blocks / depth      -> 分档聚合
    target_row / pred_row       -> 检查"行定位"是否真的被考察（E6 的修复效果）
    latency_ms.{index,query}    -> E4 的延迟协议（end2end vs precomputed）
    index_bytes                 -> E11 的存储开销
    n_candidates                -> 分母，直接写进表格 caption

断点续跑：已存在于 JSONL 中的 (run_id, task_id) 会被跳过；失败的任务会写
failure 记录并允许下次重试（原代码的 `except: pass` 就是 31 个任务消失的原因）。
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path

import numpy as np

from schemas import SCHEMA_EVAL, SCHEMA_FAILURE, SCHEMA_RUN
from common import JsonlWriter, ensure_dir, env_snapshot, load_done_keys, now_iso, read_jsonl, repo_path
from datasets_eval import list_tasks
from scorers import build_scorer

DEFAULT_BENCH = "data/niah_robust_test"
DEFAULT_OUT = "results/jsonl/eval"
DEFAULT_CACHE = "results/cache"


def _slug(name: str) -> str:
    keep = [c if (c.isalnum() or c in "._-") else "-" for c in name]
    return "".join(keep).strip("-")


def _has_run_header(path: Path) -> bool:
    return any(r.get("record") == "run_start" for r in read_jsonl(path))


def _read_run_header(path: Path) -> dict | None:
    """读结果文件里的 run_start 头（用于判断这个文件是不是同一份测试集写的）。"""
    for r in read_jsonl(path):
        if r.get("record") == "run_start":
            return r
    return None


def _bench_mismatch_warning(path: Path, bench_root: str, n_tasks: int) -> str | None:
    """结果文件已经存在、但记录的是**另一份测试集**时给出告警。

    两份测试集只要 tag 相同（都叫 main）、task_id 又从 0 开始编号，
    断点续跑就会把新测试集的前 N 个任务当成"已完成"直接跳过，
    汇总表里还会把两份集子的成绩混在一起 —— 必须显式提醒。
    """
    if not path.exists():
        return None
    hdr = _read_run_header(path)
    if not hdr:
        return None
    old_root = str(hdr.get("bench_root", ""))
    old_n = hdr.get("n_tasks_total")
    if old_root == str(bench_root) and (old_n in (None, n_tasks)):
        return None
    return (f"结果文件 {path} 之前记录的是另一份测试集"
            f"（bench={old_root}，{old_n} 个 task），本次是 {bench_root}（{n_tasks} 个 task）。\n"
            "   两份集子的 task_id 会重叠，继续写会让汇总表混在一起。建议：\n"
            f"     ① 换输出目录：--out-dir results/jsonl/eval_<新版标签>\n"
            f"     ② 或删掉这个文件后重跑：rm {path}")


def _cache_tag(spec) -> str:
    """文档端 embedding 的缓存键。

    注意：文档编码与**聚合方式无关**，所以 RAVT 与 RAVT-LI（late interaction）
    必须共用同一份缓存，否则 precomputed 协议下会白算一遍索引。
    """
    if spec.kind == "ravt":
        stem = Path(spec.checkpoint).stem if spec.checkpoint else "raw"
        return f"ravt_{stem}_p{spec.patch_size}_r{spec.n_rows}"
    if spec.kind == "text_encoder":
        stem = Path(spec.checkpoint).name if spec.checkpoint else spec.model_id.replace("/", "_")
        return f"txt_{stem}_{spec.pooling}"
    if spec.kind == "bm25":
        return f"bm25_{spec.tokenizer}"
    return f"{spec.kind}_{spec.model_id.replace('/', '_')}"


def _cache_path(cache_root: Path, tag: str, task_id: int) -> Path:
    return cache_root / tag / f"task_{task_id:05d}.npy"


def _docs_identical(docs) -> bool:
    """文档表示是否完全相同。

    用于区分两种情况：
      * 文档本身相同（如"去同形字"基线在高噪声下把每个块都还原成空串）
        → 分数全相同是**基线的真实行为**，应如实记录，不能当成错误丢掉；
      * 文档不同但分数全相同 → 模型/预处理有错（旧版 Jina 基线全 0 就是这种），
        必须拒绝写入表格。
    """
    if docs is None:
        return False
    if isinstance(docs, np.ndarray):
        if docs.shape[0] < 2:
            return True
        return bool(np.array_equal(docs[0], docs[1]) and np.allclose(docs, docs[0]))
    if isinstance(docs, (list, tuple)) and len(docs) >= 2:
        first = docs[0]
        return all(d == first for d in docs)
    return False


def evaluate_run(spec, bench_root: str = DEFAULT_BENCH, protocol: str = "end2end",
                 out_dir: str = DEFAULT_OUT, cache_dir: str = DEFAULT_CACHE,
                 device: str | None = None, block_counts=None, noise_levels=None,
                 depths=None, image_levels=None, max_tasks: int | None = None,
                 use_cache: bool = True,
                 skip_prepare: bool = False) -> dict:
    """跑完一个 EvalSpec，返回统计摘要。"""
    run_id = _slug(spec.name)
    cache_tag = _cache_tag(spec)
    out_path = repo_path(out_dir) / f"{run_id}.jsonl"
    ensure_dir(out_path.parent)
    cache_root = repo_path(cache_dir)

    tasks = list_tasks(repo_path(bench_root), block_counts=block_counts,
                       noise_levels=noise_levels, depths=depths, image_levels=image_levels,
                       max_tasks=max_tasks,
                       seed=spec.extra.get("task_seed", 0) if isinstance(spec.extra, dict) else 0)
    done = load_done_keys(out_path, ("run_id", "task_id"))
    todo = [t for t in tasks if (run_id, t.task_id) not in done]

    print(f"\n=== 评测 {spec.name} ({spec.kind}) ===")
    print(f"  task 总数 {len(tasks)}，已完成 {len(tasks) - len(todo)}，本次待跑 {len(todo)}")
    print(f"  协议 {protocol} | 结果文件 {out_path}")
    warn = _bench_mismatch_warning(out_path, str(bench_root), len(tasks))
    if warn:
        print("  ⚠️ " + warn)
    if not todo:
        return {"run_id": run_id, "n_done": len(tasks), "n_new": 0, "path": str(out_path)}

    scorer = build_scorer(spec, device=device)
    writer = JsonlWriter(out_path)
    if not _has_run_header(out_path):
        writer.write({"record": "run_start", "schema": SCHEMA_RUN, "timestamp": now_iso(),
                      "run_id": run_id, "spec": spec.to_dict(), "protocol": protocol,
                      "bench_root": str(bench_root), "n_tasks_total": len(tasks),
                      "env": env_snapshot()})
    if not skip_prepare:
        t0 = time.time()
        scorer.prepare()
        print(f"  模型加载耗时 {time.time() - t0:.1f}s")

    n_ok, n_fail = 0, 0
    t_run = time.time()
    for i, task in enumerate(todo):
        try:
            lat_index = lat_query = 0.0
            docs = None

            if protocol == "precomputed" and scorer.supports_precompute:
                cp = _cache_path(cache_root, cache_tag, task.task_id)
                if use_cache and cp.exists():
                    docs = np.load(cp)
                else:
                    t0 = time.perf_counter()
                    docs = scorer.encode_documents(task)
                    lat_index = (time.perf_counter() - t0) * 1000.0
                    if use_cache and isinstance(docs, np.ndarray):
                        ensure_dir(cp.parent)
                        np.save(cp, docs)

                t0 = time.perf_counter()
                scores, extra = scorer.score_query(task.query, docs, task)
                lat_query = (time.perf_counter() - t0) * 1000.0
            else:
                t0 = time.perf_counter()
                docs = scorer.encode_documents(task)
                scores, extra = scorer.score_query(task.query, docs, task)
                lat_query = (time.perf_counter() - t0) * 1000.0
                lat_index = 0.0

            scores = np.asarray(scores, dtype=np.float64)
            if scores.size != len(task.image_paths) and scores.size != task.blocks:
                raise ValueError(f"得分数量 {scores.size} 与块数 {task.blocks} 不符")
            if not np.isfinite(scores).all():
                raise ValueError("得分包含 NaN/Inf")
            degenerate = False
            tie_all = False
            if scores.size > 1 and float(np.nanstd(scores)) == 0.0:
                if _docs_identical(docs):
                    # 文档本身一样（例如去噪基线把整块还原成空串）：
                    # 这是基线的真实失败，如实记录，不丢弃（否则会高估它的成绩）
                    degenerate = True
                elif getattr(scorer, "allows_score_ties", False):
                    # 稀疏检索器（BM25 系）"查询词一个都没命中" -> 全 0 分。
                    # 这是真实结果（保守记为未命中），不是模型/配置错误。
                    tie_all = True
                else:
                    raise ValueError(
                        "所有块的得分完全相同，但文档表示并不相同"
                        "——模型/预处理配置有误，拒绝写入结果")

            pred_block = int(np.argmax(scores))
            if tie_all:
                # 平局时 argmax 恒为 0，命中与否纯属运气 —— 保守判为未命中，
                # 并在 JSONL 里打标记，方便做敏感性分析时单独剔除/复算。
                pred_block = -1
            row_best = extra.get("row_scores_best")
            pred_row = int(row_best[pred_block]) if (row_best and pred_block >= 0) else -1
            if not tie_all and task.has_distractor and pred_block >= 0 \
                    and pred_block == task.distractor_block \
                    and task.distractor_block != task.gt_block:
                hit_distractor = True
            else:
                hit_distractor = False

            idx_bytes = scorer.index_bytes(docs) if docs is not None else 0
            writer.write({
                "record": "task_eval", "schema": SCHEMA_EVAL, "timestamp": now_iso(),
                "run_id": run_id, "model": spec.name, "kind": spec.kind,
                "protocol": protocol,
                "task_id": task.task_id, "blocks": task.blocks, "noise": task.noise,
                # 注意：这里写**原始 metadata 值**，不要回退成 noise ——
                # 老测试集没有成像轴，回退会让汇总表多出一张与噪声档完全重复的表。
                "image_level": task.meta.get("image_level"),
                "image_coverage": task.meta.get("image_coverage"),
                "depth": task.depth, "depth_actual": task.meta.get("depth_actual"),
                "benchmark": task.meta.get("benchmark"), "corruptions": task.meta.get("corruptions"),
                "target_block": task.gt_block, "target_row": task.gt_row,
                "has_distractor": task.has_distractor,
                "distractor_block": task.distractor_block,
                "pred_block": pred_block, "pred_row": pred_row,
                "correct": int(pred_block == task.gt_block),
                "hit_distractor": int(hit_distractor),
                "n_candidates": len(scores),
                "index_bytes": int(idx_bytes),
                "bytes_per_doc": int(idx_bytes / max(len(scores), 1)),
                "latency_ms": {"index": round(lat_index, 3), "query": round(lat_query, 3),
                               "total": round(lat_index + lat_query, 3)},
                "top1_score": float(np.max(scores)),
                "score_std": float(np.std(scores)),
                # OCR 流水线专用：把识别错误率一并记下来（其他模型为 None）
                "ocr_cer": extra.get("ocr_cer"),
                # 基线自身退化（如去噪后全变空串）——聚合时要单独说明
                "degenerate_input": int(degenerate),
                "score_all_tied": int(tie_all),
            })
            n_ok += 1
        except Exception as exc:
            n_fail += 1
            writer.write({
                "record": "failure", "schema": SCHEMA_FAILURE, "timestamp": now_iso(),
                "run_id": run_id, "task_id": task.task_id,
                "error_type": type(exc).__name__, "error_msg": str(exc)[:500],
                "traceback": traceback.format_exc()[-1500:],
            })
            print(f"  [失败] task {task.task_id}: {type(exc).__name__}: {exc}")

        if (i + 1) % 20 == 0 or (i + 1) == len(todo):
            rate = (i + 1) / max(time.time() - t_run, 1e-6)
            eta = (len(todo) - i - 1) / max(rate, 1e-9)
            print(f"  {i + 1}/{len(todo)}  成功 {n_ok} 失败 {n_fail}  "
                  f"{rate:.2f} task/s  ETA {eta / 60:.1f} min", end="\r")

    print()
    writer.write({"record": "run_end", "schema": SCHEMA_RUN, "timestamp": now_iso(),
                  "run_id": run_id, "n_new": n_ok, "n_failed": n_fail,
                  "elapsed_s": round(time.time() - t_run, 1)})
    writer.close()
    scorer.close()
    return {"run_id": run_id, "n_new": n_ok, "n_failed": n_fail, "path": str(out_path)}
