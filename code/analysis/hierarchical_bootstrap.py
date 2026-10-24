"""Hierarchical bootstrap for the primary comparisons in the IPM paper.

The resampling unit is the design cell (haystack size x needle depth), not an
individual task.  RAVT and adapter seed identities are resampled separately.
For released, single-run baselines only the five RAVT seeds vary; their
intervals condition on the released checkpoint.

Example:

    python code/analysis/hierarchical_bootstrap.py \
      --main-dir results/jsonl/main \
      --colpali-lora-dirs results/jsonl/colpali_s0 \
                         results/jsonl/colpali_s1 \
                         results/jsonl/colpali_s2 \
      --colqwen2-lora-dirs results/jsonl/colqwen2_s0 \
                           results/jsonl/colqwen2_s1 \
                           results/jsonl/colqwen2_s2
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

def load_tasks(path: Path) -> dict[int, dict]:
    tasks: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("record") == "task_eval":
            tasks[int(record["task_id"])] = record
    return tasks


def bootstrap_comparison(
    baseline: dict[int, dict],
    ravt_tasks: list[dict[int, dict]],
    adapter_tasks: list[dict[int, dict]] | None = None,
    n_boot: int = 10_000,
    rng_seed: int = 20261023,
) -> dict[str, float | int]:
    """Bootstrap adapter-minus-RAVT, or RAVT-minus-baseline when no adapter."""
    all_tasks = [baseline, *ravt_tasks, *(adapter_tasks or [])]
    task_ids = set.intersection(*(set(tasks) for tasks in all_tasks))
    task_ids = {
        task_id
        for task_id in task_ids
        if round(float(baseline[task_id]["noise"]), 4) == 0.4
    }

    cells: dict[tuple[int, float], list[int]] = defaultdict(list)
    observed: list[float] = []
    for task_id in sorted(task_ids):
        record = baseline[task_id]
        cells[(int(record["blocks"]), float(record["depth"]))].append(task_id)
        if adapter_tasks:
            adapter_mean = st.mean(tasks[task_id]["correct"] for tasks in adapter_tasks)
            ravt_mean = st.mean(tasks[task_id]["correct"] for tasks in ravt_tasks)
            observed.append(100.0 * (adapter_mean - ravt_mean))
        else:
            ravt_mean = st.mean(tasks[task_id]["correct"] for tasks in ravt_tasks)
            observed.append(100.0 * (ravt_mean - float(record["correct"])))

    keys = sorted(cells)
    ordered_tasks = [task_id for cell in keys for task_id in cells[cell]]
    cell_positions = []
    cursor = 0
    for cell in keys:
        size = len(cells[cell])
        cell_positions.append((cursor, cursor + size))
        cursor += size

    ravt_values = [
        [float(tasks[task_id]["correct"]) for task_id in ordered_tasks]
        for tasks in ravt_tasks
    ]
    n_ravt = len(ravt_tasks)
    ravt_combos = list(itertools.product(range(n_ravt), repeat=n_ravt))
    ravt_means = [
        [
            sum(ravt_values[seed][position] for seed in combo) / n_ravt
            for position in range(len(ordered_tasks))
        ]
        for combo in ravt_combos
    ]

    adapter_combos = None
    adapter_means = None
    if adapter_tasks:
        adapter_values = [
            [float(tasks[task_id]["correct"]) for task_id in ordered_tasks]
            for tasks in adapter_tasks
        ]
        adapter_combos = list(
            itertools.product(range(len(adapter_tasks)), repeat=len(adapter_tasks))
        )
        adapter_means = [
            [
                sum(adapter_values[seed][position] for seed in combo)
                / len(adapter_tasks)
                for position in range(len(ordered_tasks))
            ]
            for combo in adapter_combos
        ]

    baseline_values = [
        float(baseline[task_id]["correct"]) for task_id in ordered_tasks
    ]

    def cell_offsets(
        ravt_combo_index: int,
        adapter_combo_index: int | None,
    ) -> tuple[list[float], list[int]]:
        offsets: list[float] = []
        sizes: list[int] = []
        for start, end in cell_positions:
            size = end - start
            ravt_sum = sum(ravt_means[ravt_combo_index][start:end])
            if adapter_means is not None:
                adapter_sum = sum(adapter_means[adapter_combo_index][start:end])
                offsets.append(100.0 * (adapter_sum - ravt_sum) / size)
            else:
                baseline_sum = sum(baseline_values[start:end])
                offsets.append(100.0 * (ravt_sum - baseline_sum) / size)
            sizes.append(size)
        return offsets, sizes

    rng = random.Random(rng_seed)
    draws: list[float] = []
    for _ in range(n_boot):
        ravt_combo_index = rng.randrange(len(ravt_combos))
        adapter_combo_index = (
            rng.randrange(len(adapter_combos)) if adapter_combos is not None else None
        )
        offsets, sizes = cell_offsets(ravt_combo_index, adapter_combo_index)
        picked_cells = [keys[rng.randrange(len(keys))] for _ in keys]
        numerator = 0.0
        denominator = 0
        for key in picked_cells:
            cell_index = keys.index(key)
            size = sizes[cell_index]
            numerator += offsets[cell_index] * size
            denominator += size
        draws.append(numerator / denominator)

    draws.sort()
    lo_index = int(0.025 * n_boot)
    hi_index = int(0.975 * n_boot) - 1
    return {
        "delta": st.mean(observed),
        "lo": draws[lo_index],
        "hi": draws[hi_index],
        "tasks": len(task_ids),
        "cells": len(cells),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Hierarchical bootstrap over design cells and training seeds."
    )
    ap.add_argument("--main-dir", required=True,
                    help="Directory containing main-task JSONL files.")
    ap.add_argument("--ravt-pattern", default="RAVT-rh42-s{seed}-main.jsonl")
    ap.add_argument("--ravt-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--colpali-lora-dirs", nargs="+", default=[])
    ap.add_argument("--colqwen2-lora-dirs", nargs="+", default=[])
    ap.add_argument("--out", default="results/analysis/hierarchical_bootstrap.md")
    ap.add_argument("--bootstrap", type=int, default=10_000)
    ap.add_argument("--rng-seed", type=int, default=20261023)
    args = ap.parse_args()

    main_dir = Path(args.main_dir)
    ravt = [
        load_tasks(main_dir / args.ravt_pattern.format(seed=seed))
        for seed in args.ravt_seeds
    ]
    baselines = {
        "ColPali": load_tasks(main_dir / "ColPali-main.jsonl"),
        "ColQwen2": load_tasks(main_dir / "ColQwen2-main.jsonl"),
        "DSE-Phi3": load_tasks(main_dir / "DSE-Phi3-main.jsonl"),
        "BM25-char3 + OCR": load_tasks(
            main_dir / "BM25-char3-OCR-tesseract--main.jsonl"
        ),
    }
    adapters = {}
    if args.colpali_lora_dirs:
        adapters["ColPali + LoRA"] = [
            load_tasks(Path(directory) / "ColPali-main.jsonl")
            for directory in args.colpali_lora_dirs
        ]
    if args.colqwen2_lora_dirs:
        adapters["ColQwen2 + LoRA"] = [
            load_tasks(Path(directory) / "ColQwen2-main.jsonl")
            for directory in args.colqwen2_lora_dirs
        ]

    rows = []
    for baseline_name, baseline in baselines.items():
        result = bootstrap_comparison(baseline, ravt, n_boot=args.bootstrap,
                                      rng_seed=args.rng_seed)
        rows.append(
            {
                "comparison": f"RAVT - {baseline_name}",
                "direction": "RAVT",
                **result,
            }
        )
    for adapter_name, adapter in adapters.items():
        result = bootstrap_comparison(adapter[0], ravt, adapter,
                                      n_boot=args.bootstrap,
                                      rng_seed=args.rng_seed)
        rows.append(
            {
                "comparison": f"{adapter_name} - RAVT",
                "direction": "adapter",
                **result,
            }
        )

    lines = [
        "# Hierarchical bootstrap",
        "",
        (
            f"{args.bootstrap:,} resamples; design cells = haystack size x needle depth; "
            f"seed = {args.rng_seed}. RAVT uses {len(ravt)} seeds. "
            "Adapter comparisons use the supplied adapter seeds. "
            "Released single-run baselines condition on their released "
            "checkpoint; their intervals do not estimate baseline training variance."
        ),
        "",
        "| comparison | tasks | cells | delta (points) | 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['comparison']} | {row['tasks']} | {row['cells']} | "
            f"{row['delta']:+.2f} | [{row['lo']:+.2f}, {row['hi']:+.2f}] |"
        )
    lines.append("")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
