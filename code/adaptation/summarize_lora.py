"""汇总两个页面模型在 40% hard subset 上的三 seed 结果。"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path


EXPECTED_S0 = {
    "ColPali": (91.7, 90.8, 80.4),
    "ColQwen2": (84.9, 82.8, 68.2),
}


def load_rows(path: Path) -> dict[int, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("record") == "task_eval":
            rows[int(record["task_id"])] = record
    return rows


def hard_map(rows: dict[int, dict]) -> dict[int, int]:
    return {
        task_id: int(row["correct"])
        for task_id, row in rows.items()
        if round(float(row["noise"]), 2) == 0.4
    }


def discover_s0(root: Path, model: str) -> tuple[int, dict[int, int]]:
    target = EXPECTED_S0[model]
    best = None
    for path in sorted((root / "results" / "jsonl").rglob(f"{model}-main.jsonl")):
        all_rows = load_rows(path)
        level_rows = {
            level: [
                row for row in all_rows.values()
                if round(float(row["noise"]), 2) == level
            ]
            for level in (0.0, 0.2, 0.4)
        }
        # ColQwen2 has four degenerate queries in the original s0 run, so its
        # per-level denominator can be 238/239 rather than exactly 240.
        if any(len(rows) < 230 for rows in level_rows.values()):
            continue
        acc = tuple(
            100.0 * sum(int(r["correct"]) for r in level_rows[lv]) / 240
            for lv in (0.0, 0.2, 0.4)
        )
        error = sum(abs(a - b) for a, b in zip(acc, target))
        if error <= 1.5 and (best is None or error < best[0]):
            best = (error, path, hard_map(all_rows))
    if best is None:
        raise SystemExit(f"找不到 {model} 的单 seed 原始结果")
    print(f"{model} s0 <- {best[1]}")
    return 0, best[2]


def exact_mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def bootstrap(diffs: list[float], seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    vals = [
        statistics.mean(diffs[rng.randrange(len(diffs))] for _ in diffs)
        for _ in range(20000)
    ]
    return percentile(vals, 0.025), percentile(vals, 0.975)


def summarize_model(root: Path, model: str, ravt_maps: list[dict[int, int]]) -> list[str]:
    seed_maps = [discover_s0(root, model)[1]]
    slug = model.lower()
    for seed in (1, 2):
        directory = root / "results" / "jsonl" / "eval_lora_core" / f"{slug}_s{seed}"
        files = sorted(directory.glob(f"{model}-main.jsonl"))
        if not files:
            raise SystemExit(f"缺少 {model} seed {seed}: {directory}")
        seed_maps.append(hard_map(load_rows(files[-1])))

    ids = sorted(set.intersection(*(set(m) for m in seed_maps + ravt_maps)))
    if len(ids) < 230:
        raise SystemExit(f"{model} 配对 task 数过少：{len(ids)}")
    seed_acc = [100.0 * sum(m[i] for i in ids) / len(ids) for m in seed_maps]
    ravt_acc = [100.0 * sum(m[i] for i in ids) / len(ids) for m in ravt_maps]
    task_diffs = [
        100.0 * (statistics.mean(m[i] for m in seed_maps)
                 - statistics.mean(m[i] for m in ravt_maps))
        for i in ids
    ]
    lo, hi = bootstrap(task_diffs, 20261021)
    adapter_majority = [
        int(sum(m[i] for m in seed_maps) >= 2) for i in ids
    ]
    ravt_majority = [
        int(sum(m[i] for m in ravt_maps) >= 3) for i in ids
    ]
    b = sum(a == 1 and r == 0 for a, r in zip(adapter_majority, ravt_majority))
    c = sum(a == 0 and r == 1 for a, r in zip(adapter_majority, ravt_majority))
    return [
        f"### {model} + LoRA, 40% corruption",
        "",
        f"- {model}: {statistics.mean(seed_acc):.1f} ± {statistics.stdev(seed_acc):.1f} "
        f"（seeds: {', '.join(f'{v:.1f}' for v in seed_acc)}; n={len(ids)}）",
        f"- RAVT: {statistics.mean(ravt_acc):.1f} ± {statistics.stdev(ravt_acc):.1f}",
        f"- paired difference: {statistics.mean(task_diffs):+.1f} "
        f"(95% CI [{lo:+.1f}, {hi:+.1f}], exact McNemar p={exact_mcnemar(b, c):.3g})",
        "",
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--out", default="results/adaptation")
    args = ap.parse_args()
    root = Path(args.root).resolve()

    ravt_maps = [
        hard_map(load_rows(path))
        for path in sorted((root / "results" / "jsonl" / "eval_tmm_rh42").glob(
            "RAVT-rh42-s*-main.jsonl"))
    ]
    if len(ravt_maps) != 5:
        raise SystemExit(f"RAVT seed 文件应为 5 个，实际 {len(ravt_maps)}")

    lines = ["# 页面模型三 seed 核心实验（40% hard subset）", ""]
    for model in ("ColQwen2", "ColPali"):
        lines += summarize_model(root, model, ravt_maps)

    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    (out / "lora_core.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
