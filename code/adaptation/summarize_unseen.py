"""汇总并配对分析 unseen-family 结果。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def load_task_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    paths = [path] if path.is_file() else sorted(path.glob("*.jsonl"))
    for file_path in paths:
        for line in file_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("record") == "task_eval":
                rows.append(record)
    return rows


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def cluster_bootstrap_ci(
    task_diff: dict[int, float],
    cluster_of: dict[int, int | str],
    reps: int,
    seed: int,
) -> tuple[float, float]:
    clusters: dict[int | str, list[float]] = {}
    for task_id, diff in task_diff.items():
        clusters.setdefault(cluster_of.get(task_id, task_id), []).append(diff)
    keys = sorted(clusters, key=str)
    rng = random.Random(seed)
    estimates = []
    for _ in range(reps):
        chosen = [keys[rng.randrange(len(keys))] for _ in range(len(keys))]
        vals = [value for key in chosen for value in clusters[key]]
        estimates.append(100.0 * statistics.mean(vals))
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--out", default="results/adaptation")
    ap.add_argument("--bootstrap", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261021)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    base = root / "results" / "jsonl" / "eval_unseen240"
    ravt_runs = [
        load_task_rows(path) for path in sorted((base / "ravt").glob("*.jsonl"))
    ]
    ravt_runs = [rows for rows in ravt_runs if rows]
    ocr_rows = load_task_rows(base / "ocr")
    if not ravt_runs or not ocr_rows:
        raise SystemExit(f"缺少结果文件：{base}")
    if len(ravt_runs) != 5:
        raise SystemExit(f"RAVT 种子数应为 5，实际找到 {len(ravt_runs)}")

    levels = sorted({round(float(row["noise"]), 2) for row in ocr_rows})
    summary: list[dict] = []
    for level in levels:
        seed_maps = [
            {
                int(row["task_id"]): int(row["correct"])
                for row in rows
                if round(float(row["noise"]), 2) == level
            }
            for rows in ravt_runs
        ]
        ocr_level = [
            row for row in ocr_rows if round(float(row["noise"]), 2) == level
        ]
        ocr_map = {int(row["task_id"]): int(row["correct"]) for row in ocr_level}
        cluster_map = {}
        for row in ocr_level:
            source_task_id = row.get("source_task_id")
            cluster_map[int(row["task_id"])] = int(
                row["task_id"] if source_task_id is None else source_task_id
            )
        ids = sorted(ocr_map)
        if not ids:
            continue
        if any(set(seed_map) != set(ids) for seed_map in seed_maps):
            raise SystemExit(f"noise={level:g} 的 RAVT/OCR task 集合不一致")

        seed_acc = [
            100.0 * sum(seed_map[task_id] for task_id in ids) / len(ids)
            for seed_map in seed_maps
        ]
        task_means = [
            statistics.mean(seed_map[task_id] for seed_map in seed_maps)
            for task_id in ids
        ]
        task_diffs = {
            task_id: task_means[pos] - ocr_map[task_id]
            for pos, task_id in enumerate(ids)
        }
        diff = 100.0 * statistics.mean(task_diffs.values())
        ci_lo, ci_hi = cluster_bootstrap_ci(
            task_diffs, cluster_map, args.bootstrap, args.seed
        )
        majority = [
            int(sum(seed_map[task_id] for seed_map in seed_maps) >= 3)
            for task_id in ids
        ]
        b = sum(m == 1 and ocr_map[task_id] == 0 for m, task_id in zip(majority, ids))
        c = sum(m == 0 and ocr_map[task_id] == 1 for m, task_id in zip(majority, ids))
        summary.append(
            {
                "noise": level,
                "n_tasks": len(ids),
                "n_clusters": len(set(cluster_map.values())),
                "ravt_mean": statistics.mean(seed_acc),
                "ravt_sd": statistics.stdev(seed_acc),
                "ocr": 100.0 * statistics.mean(ocr_map[task_id] for task_id in ids),
                "diff": diff,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "mcnemar_b": b,
                "mcnemar_c": c,
                "p": exact_mcnemar(b, c),
                "seed_acc": seed_acc,
            }
        )

    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "noise",
        "n_tasks",
        "n_clusters",
        "ravt_mean",
        "ravt_sd",
        "ocr",
        "diff",
        "ci_lo",
        "ci_hi",
        "mcnemar_b",
        "mcnemar_c",
        "p",
    ]
    with (out_dir / "unseen_family240.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow({key: row[key] for key in fieldnames})

    lines = [
        "# 未见腐蚀族扩容实验（transposition / keyboard_typo / deletion）",
        "",
        "RAVT 为五个种子的均值 ± 标准差。差值 = RAVT − char3+OCR；"
        "置信区间按底层 task 聚类 bootstrap，p 为五种子多数票对 char3+OCR 的精确 McNemar 检验。",
        "",
        "| 档位 | task/簇 | RAVT | char3+OCR | 差值 | 95% CI | p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {100*row['noise']:.0f}% | {row['n_tasks']}/{row['n_clusters']} | "
            f"{row['ravt_mean']:.1f} ± {row['ravt_sd']:.1f} | {row['ocr']:.1f} | "
            f"{row['diff']:+.1f} | [{row['ci_lo']:+.1f}, {row['ci_hi']:+.1f}] | "
            f"{row['p']:.3g} |"
        )
    lines += ["", "## 各种子 Recall@1", ""]
    for row in summary:
        values = ", ".join(f"{value:.1f}" for value in row["seed_acc"])
        lines.append(f"- {100*row['noise']:.0f}%: {values}")

    text = "\n".join(lines) + "\n"
    (out_dir / "unseen_family240.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
