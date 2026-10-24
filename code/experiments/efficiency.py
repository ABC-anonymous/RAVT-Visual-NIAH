"""实验 3：效率的统一口径 —— 端到端 / 离线建库 / 在线查询，以及峰值显存。

三块，对应审稿人 Major 4（“效率口径要重新审计”）：
  A. 端到端口径：直接读主评测的逐任务 JSONL（results/jsonl/eval_tmm_rh42），
     给出每任务 P50/P95 时间、KB/doc、QPS —— 这就是论文表 II 的“端到端查询延迟”。
  B. 预计算口径：读 results/jsonl/latency_rh42（protocol=precomputed），
     把“文档编码/建库”与“查询”分开，说明表 II 的 2 秒里有多少是每次重编码 haystack。
  C. --measure：同机、同批块图的自测：文档编码 ms/块、查询 ms、权重占用与推理峰值显存。
     模型按各自签名构造，装不上的跳过并打印原因。

    python code/experiments/efficiency.py --root . --measure
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

#: 表 A 里出现的模型（与论文表 II 对齐）
MAIN_MODELS = ("RAVT-rh42-s0@main", "BM25-char3@main", "BM25-char3+OCR[tesseract]@main",
               "BGE-M3@main", "BGE-M3+OCR[tesseract]@main", "ColQwen2@main",
               "ColPali@main", "DSE-Phi3@main")


class _Sub:
    """给 scorer 用的最小 task 壳：只暴露 load_images / load_texts。"""

    def __init__(self, imgs, texts):
        self._imgs = imgs
        self._texts = texts
        self.image_paths = []
        self.path = Path(".")

    def load_images(self):
        return list(self._imgs)

    def load_texts(self):
        return list(self._texts)


def read_jsonl_dir(path: Path):
    per = defaultdict(lambda: {"index": [], "query": [], "total": [], "bytes": [],
                               "blocks": [], "protocol": set()})
    n_files = 0
    if not path.is_dir():
        return per, n_files
    for f in sorted(path.glob("*.jsonl")):
        n_files += 1
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("record") != "task_eval":
                continue
            d = per[r["model"]]
            lat = r.get("latency_ms") or {}
            d["index"].append(float(lat.get("index", 0.0)))
            d["query"].append(float(lat.get("query", 0.0)))
            d["total"].append(float(lat.get("total", 0.0)))
            d["bytes"].append(float(r.get("bytes_per_doc") or 0))
            d["blocks"].append(int(r.get("blocks") or 0))
            d["protocol"].add(r.get("protocol"))
    return per, n_files


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[i]


def summarize(per, models):
    rows = []
    for m in models:
        if m not in per:
            continue
        d = per[m]
        tot = d["total"]
        p50 = st.median(tot)
        rows.append({"model": m, "protocol": ",".join(sorted(x for x in d["protocol"] if x)),
                     "kb_per_doc": st.mean(d["bytes"]) / 1024.0,
                     "p50_index_ms": st.median(d["index"]),
                     "p50_query_ms": st.median(d["query"]),
                     "p50_total_ms": p50, "p95_total_ms": pct(tot, 0.95),
                     "qps": 1000.0 / p50 if p50 else float("nan"),
                     "mean_blocks": st.mean(d["blocks"]), "n": len(tot)})
    return rows


def build_scorer(name: str, root: Path):
    """按 evaluate.py 的默认权重路径构造 scorer；返回 (scorer, kind)。"""
    import scorers
    from ravt_paths import find_weight_dir

    if name == "RAVT":
        return scorers.RavtScorer(
            "RAVT", clip_path=str(root / "clip_local_weights_224"),
            checkpoint=str(root / "checkpoints" / "full_p14_rh42_s0.pth"),
            row_height=42), "image"
    if name == "BGE-M3":
        return scorers.TextEncoderScorer(
            "BGE-M3", model_id=str(root / "models_local" / "bge-m3"),
            pooling="cls", batch_size=64), "text"
    if name == "ColPali":
        base = root / "colpali_offline_weights" / "base_model"
        ad = root / "colpali_offline_weights" / "adapter_model"
        if not (base.exists() and ad.exists()):
            raise FileNotFoundError(f"ColPali 权重缺失：{base} / {ad}")
        return scorers.ColPaliScorer("ColPali", model_id="vidore/colpali-v1.2",
                                     base_model_dir=str(base), adapter_dir=str(ad),
                                     batch_size=4, dtype="bfloat16"), "image"
    if name == "ColQwen2":
        base = find_weight_dir("colqwen2-base", root)
        ad = find_weight_dir("colqwen2", root)
        if not (base.exists() and ad.exists()):
            raise FileNotFoundError(f"ColQwen2 权重缺失：{base} / {ad}")
        return scorers.ColQwen2Scorer("ColQwen2", model_id="vidore/colqwen2-v1.0",
                                      local_dir=str(base), adapter_dir=str(ad),
                                      batch_size=1, dtype="bfloat16"), "image"
    raise KeyError(name)


def measure(root: Path, bench: Path, models, n_blocks: int, noise: float):
    """同批块图上实测文档编码 / 查询 / 显存。"""
    import torch
    from datasets_eval import list_tasks

    tasks = list_tasks(bench, noise_levels=[noise], max_tasks=8)
    if not tasks:
        print("  没有 task，跳过自测")
        return [], None
    task = max(tasks, key=lambda t: min(t.blocks, n_blocks))
    imgs = task.load_images()[:n_blocks]
    texts = task.load_texts()[:n_blocks]
    probe = _Sub(imgs, texts)
    print(f"  自测样本：task {task.task_id}（共 {task.blocks} 块），取前 {len(imgs)} 块，"
          f"noise={noise}")

    out = []
    for name in models:
        try:
            sc, kind = build_scorer(name, root)
        except Exception as e:                                  # noqa: BLE001
            print(f"  [跳过] {name}: {type(e).__name__}: {str(e)[:120]}")
            continue
        try:
            sc.prepare()
        except Exception as e:                                  # noqa: BLE001
            print(f"  [跳过] {name} prepare: {type(e).__name__}: {str(e)[:140]}")
            continue
        try:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            weights_gb = (torch.cuda.memory_allocated() / 1e9) if dev == "cuda" else float("nan")
            if dev == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            docs = sc.encode_documents(probe)
            t_off = time.perf_counter() - t0
            t0 = time.perf_counter()
            scores, _ = sc.score_query(task.query, docs, task=None)
            t_on = time.perf_counter() - t0
            peak_gb = (torch.cuda.max_memory_allocated() / 1e9) if dev == "cuda" else float("nan")
            n_docs = len(scores) if scores is not None else len(imgs)
            row = {"model": name, "kind": kind, "n_docs": n_docs,
                   "offline_ms_per_doc": 1000.0 * t_off / max(n_docs, 1),
                   "online_ms_per_query": 1000.0 * t_on,
                   "kb_per_doc": sc.index_bytes(docs) / 1024.0 / max(n_docs, 1),
                   "weights_gb": weights_gb, "infer_peak_gb": peak_gb}
            out.append(row)
            print(f"  {name:<10} 文档编码 {row['offline_ms_per_doc']:7.2f} ms/块  查询 "
                  f"{row['online_ms_per_query']:7.2f} ms  索引 {row['kb_per_doc']:7.1f} KB/块  "
                  f"权重 {weights_gb:5.2f} GB  峰值 {peak_gb:5.2f} GB")
        except Exception as e:                                  # noqa: BLE001
            print(f"  [跳过] {name} 前向: {type(e).__name__}: {str(e)[:140]}")
        finally:
            try:
                sc.close()
            except Exception:                                   # noqa: BLE001
                pass
            del sc
            try:
                torch.cuda.empty_cache()
            except Exception:                                   # noqa: BLE001
                pass
    return out, {"task_id": task.task_id, "n_docs": len(imgs), "noise": noise}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--bench", default="data/test_bench_tmm_rh42")
    ap.add_argument("--end2end-dir", default="results/jsonl/eval_tmm_rh42")
    ap.add_argument("--precomputed-dir", default="results/jsonl/latency_rh42")
    ap.add_argument("--out", default="results")
    ap.add_argument("--measure", action="store_true", help="同口径实测 + 峰值显存（需要 GPU）")
    ap.add_argument("--measure-blocks", type=int, default=64)
    ap.add_argument("--measure-noise", type=float, default=0.0)
    ap.add_argument("--measure-models", default="RAVT,BGE-M3,ColPali,ColQwen2")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    for p in (root, root / "code" / "ravt"):
        sys.path.insert(0, str(p))
    out = root / a.out
    out.mkdir(parents=True, exist_ok=True)

    # ---------- A. 端到端口径（主评测 JSONL） ----------
    per_e2e, n1 = read_jsonl_dir(root / a.end2end_dir)
    rows_a = summarize(per_e2e, MAIN_MODELS)
    print(f"A. 端到端（{n1} 个 JSONL，{a.end2end_dir}）")
    for r in rows_a:
        print(f"  {r['model']:<32} {r['kb_per_doc']:7.1f} KB/doc  查询 {r['p50_query_ms']:8.0f} ms  "
              f"端到端 P50 {r['p50_total_ms']:8.0f} ms  P95 {r['p95_total_ms']:8.0f} ms  "
              f"{r['qps']:.2f} QPS  n={r['n']}")
    if not rows_a:
        print(f"  ⚠️ {a.end2end_dir} 里没读到 task_eval 记录：确认路径与文件名")

    # ---------- B. 预计算口径（分离建库与查询） ----------
    per_pre, n2 = read_jsonl_dir(root / a.precomputed_dir)
    rows_b = summarize(per_pre, MAIN_MODELS)
    print(f"\nB. 预计算口径（{n2} 个 JSONL，{a.precomputed_dir}）")
    for r in rows_b:
        print(f"  {r['model']:<32} 建库 {r['p50_index_ms']:9.1f} ms/任务  查询 {r['p50_query_ms']:8.1f} ms  "
              f"平均块数 {r['mean_blocks']:6.1f}  n={r['n']}")
    if not rows_b:
        print(f"  ⚠️ {a.precomputed_dir} 里没读到 task_eval 记录")

    # ---------- C. 同口径实测 ----------
    mem_rows, probe = [], None
    if a.measure:
        print("\nC. 同口径实测（文档编码 / 查询 / 显存）")
        want = [m.strip() for m in a.measure_models.split(",") if m.strip()]
        mem_rows, probe = measure(root, root / a.bench, want, a.measure_blocks, a.measure_noise)

    def table(rows, cols):
        head = "| " + " | ".join(c[0] for c in cols) + " |"
        sep = "|" + "---|" * len(cols)
        body = [f"| " + " | ".join(c[1](r) for c in cols) + " |" for r in rows]
        return [head, sep] + body

    md = ["# 效率：统一口径", ""]
    md += ["## A. 端到端口径（含每次重编码 haystack；论文表 II 的那一栏）", ""]
    md += table(rows_a, [("模型", lambda r: r["model"]),
                         ("协议", lambda r: r["protocol"]),
                         ("KB/文档", lambda r: f"{r['kb_per_doc']:.1f}"),
                         ("P50 查询 ms", lambda r: f"{r['p50_query_ms']:.0f}"),
                         ("P50 端到端 ms", lambda r: f"{r['p50_total_ms']:.0f}"),
                         ("P95 端到端 ms", lambda r: f"{r['p95_total_ms']:.0f}"),
                         ("QPS", lambda r: f"{r['qps']:.2f}"),
                         ("n", lambda r: str(r["n"]))])
    md += ["", "论文表 II 的端到端延迟就取自这一栏：`end2end` 协议下每个 task 都会把"
           "整个 haystack 重新编码一次，所以“查询”一列其实包含建库成本。"
           "KB/文档 = 0 表示该基线不存稠密向量（BM25 系）。", ""]
    md += ["## B. 预计算口径（建库与查询分离）", ""]
    md += table(rows_b, [("模型", lambda r: r["model"]),
                         ("建库 ms/任务", lambda r: f"{r['p50_index_ms']:.1f}"),
                         ("查询 ms", lambda r: f"{r['p50_query_ms']:.1f}"),
                         ("平均块数", lambda r: f"{r['mean_blocks']:.1f}"),
                         ("n", lambda r: str(r["n"]))])
    md += ["", "`建库 = 0` 有两种情况：该模型**不支持预计算索引**（ColPali / ColQwen2 的多向量"
           "索引另有格式），于是文档编码被计进“查询”一列；或该次运行命中了向量缓存。"
           "RAVT 这一行才是真正的“离线编码一次 + 在线只查询”口径。", ""]
    if mem_rows:
        md += ["", "## C. 同口径实测（同一台机器、同一批块图）", "",
               f"样本：task {probe['task_id']} 的前 {probe['n_docs']} 块，noise={probe['noise']}。"
               "所有模型都在 224x224 的**块图**上测，因此这是块级口径；"
               "ColPali / ColQwen2 原生的输入是整页，整页成本更高。", ""]
        md += table(mem_rows, [("模型", lambda r: r["model"]),
                               ("文档编码 ms/块", lambda r: f"{r['offline_ms_per_doc']:.2f}"),
                               ("查询 ms", lambda r: f"{r['online_ms_per_query']:.2f}"),
                               ("KB/块", lambda r: f"{r['kb_per_doc']:.1f}"),
                               ("权重 GB", lambda r: f"{r['weights_gb']:.2f}"),
                               ("推理峰值 GB", lambda r: f"{r['infer_peak_gb']:.2f}")])
    else:
        md += ["", "## C. 同口径实测", "", "（未运行：加 `--measure` 需要 GPU）"]
    body = "\n".join(md) + "\n"
    (out / "efficiency.md").write_text(body, encoding="utf-8")
    (out / "efficiency.json").write_text(
        json.dumps({"end2end": rows_a, "precomputed": rows_b, "measured": mem_rows,
                    "probe": probe, "end2end_dir": a.end2end_dir,
                    "precomputed_dir": a.precomputed_dir}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"\n写出 {out}/efficiency.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
