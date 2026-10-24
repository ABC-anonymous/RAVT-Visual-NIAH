"""Cluster bootstrap and routing-rule operating points from evaluation JSONL.

The bootstrap resamples design cells (block count x needle depth) rather than
individual tasks because tasks sharing a cell are not independent. The script
also reports routing operating points and leave-one-axis-out correlations.

    python code/analysis/stats_cluster.py \
      --main-dir results/jsonl/main \
      --axes-dir results/jsonl/probes
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

BASELINES = {
    "ColPali@main": "ColPali-main.jsonl",
    "ColQwen2@main": "ColQwen2-main.jsonl",
    "DSE-Phi3@main": "DSE-Phi3-main.jsonl",
    "BGE-M3@main": "BGE-M3-main.jsonl",
    "BM25-word@main": "BM25-word-main.jsonl",
    "BM25-char3@main": "BM25-char3-main.jsonl",
    "BM25-char3+OCR@main": "BM25-char3-OCR-tesseract--main.jsonl",
}


def load(path: Path) -> dict[int, dict]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("record") == "task_eval":
            out[int(r["task_id"])] = r
    return out


def ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(xs: list[float], ys: list[float]) -> float:
    rx, ry = ranks(xs), ranks(ys)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def bootstrap_delta(
    per_task: dict[int, tuple[float, float, tuple]],
    levels: int,
    rng_seed: int,
):
    """per_task: task_id -> (ravt, base, cluster). Resample clusters with replacement."""
    clusters = defaultdict(list)
    for tid, (rv, bs, cl) in per_task.items():
        clusters[cl].append((rv, bs))
    keys = sorted(clusters)
    rng = random.Random(rng_seed)
    deltas = []
    for _ in range(levels):
        pick = [keys[rng.randrange(len(keys))] for _ in keys]
        num = den = 0.0
        for k in pick:
            for rv, bs in clusters[k]:
                num += rv - bs
                den += 1
        if den:
            deltas.append(100 * num / den)
    deltas.sort()
    obs = 100 * st.mean(rv - bs for rv, bs, _ in per_task.values())
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[int(0.975 * len(deltas)) - 1]
    return obs, lo, hi, len(clusters), len(per_task)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Cluster bootstrap and routing operating-point analysis."
    )
    ap.add_argument("--main-dir", default="results/jsonl/main")
    ap.add_argument("--axes-dir", default="results/jsonl/probes")
    ap.add_argument("--out", default="results/analysis/cluster_stats.md")
    ap.add_argument("--ravt-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--bootstrap", type=int, default=4000)
    ap.add_argument("--rng-seed", type=int, default=20261017)
    args = ap.parse_args()

    d = Path(args.main_dir)
    axes_dir = Path(args.axes_dir)
    ravt = [load(d / f"RAVT-rh42-s{s}-main.jsonl") for s in args.ravt_seeds]
    base = {name: load(d / fn) for name, fn in BASELINES.items()}

    lines = ["# Cluster bootstrap (cells = block count x needle depth)", ""]
    lines += ["| noise | baseline | RAVT - baseline (points) | 95% CI | clusters | tasks |",
              "|---|---|---|---|---|---|"]
    for noise in (0.0, 0.2, 0.4):
        for name, recs in base.items():
            per_task = {}
            for tid, rec in recs.items():
                if round(rec["noise"], 4) != noise:
                    continue
                vals = [r[tid]["correct"] for r in ravt if tid in r]
                if not vals:
                    continue
                rv = st.mean(vals)
                cl = (rec["blocks"], rec.get("depth"))
                per_task[tid] = (rv, float(rec["correct"]), cl)
            if not per_task:
                continue
            obs, lo, hi, ncl, n = bootstrap_delta(
                per_task, args.bootstrap, args.rng_seed
            )
            star = "*" if (lo > 0 or hi < 0) else ""
            lines.append(f"| {noise:.1f} | {name} | {obs:+.1f}{star} | "
                         f"[{lo:+.1f}, {hi:+.1f}] | {ncl} | {n} |")
            print(f"noise={noise} {name:<18} {obs:+.1f} [{lo:+.1f}, {hi:+.1f}] "
                  f"clusters={ncl} n={n}")

    # ---- routing-rule operating points ----
    axes = []
    for axis, suffix, models, ravt_files in (
        ("content", "main",
         {"text": "BM25-char3-OCR-tesseract--main.jsonl"},
         [f"RAVT-rh42-s{s}-main.jsonl" for s in args.ravt_seeds]),
        ("photometric", "image",
         {"text": "BM25-char3-OCR-tesseract--image.jsonl"},
         [f"RAVT-DEG--rh42-s{s}-image.jsonl" for s in args.ravt_seeds]),
        ("scan", "real",
         {"text": "BM25-char3-OCR-tesseract--real.jsonl"},
         [f"RAVT-DEG--rh42-s{s}-real.jsonl" for s in args.ravt_seeds]),
    ):
        folder = d if suffix == "main" else axes_dir
        text = load(folder / models["text"])
        rv = [load(folder / f) for f in ravt_files]
        key = "noise" if suffix == "main" else "image_level"
        by_level = defaultdict(list)
        for tid, rec in text.items():
            lvl = rec.get(key)
            if lvl is None:
                continue
            vals = [r[tid]["correct"] for r in rv if tid in r]
            if not vals:
                continue
            by_level[round(float(lvl), 4)].append(
                (rec["correct"], st.mean(vals), rec.get("ocr_cer")))
        for lvl in sorted(by_level):
            rows = by_level[lvl]
            cer = [c for _, _, c in rows if c is not None]
            axes.append({"axis": axis, "level": lvl,
                         "cer": st.mean(cer) if cer else float("nan"),
                         "text": 100 * st.mean(x[0] for x in rows),
                         "ravt": 100 * st.mean(x[1] for x in rows)})

    lines += ["", "## Routing-rule operating points", "",
              "| axis | level | CER | text (char3+OCR) | RAVT | RAVT - text |",
              "|---|---|---|---|---|---|"]
    for a in axes:
        lines.append(f"| {a['axis']} | {a['level']:.2f} | {a['cer']:.3f} | "
                     f"{a['text']:.1f} | {a['ravt']:.1f} | {a['ravt'] - a['text']:+.1f} |")
        print(f"{a['axis']:<12} lvl={a['level']:.2f} CER={a['cer']:.3f} "
              f"text={a['text']:5.1f} ravt={a['ravt']:5.1f} Δ={a['ravt'] - a['text']:+.1f}")

    surv = [(1 - a["cer"]) ** 3 for a in axes]
    text_acc = [a["text"] for a in axes]
    margin = [a["ravt"] - a["text"] for a in axes]
    lines += ["", "## Correlations", ""]
    lines.append(f"- n-gram survival vs text accuracy, all {len(axes)} points: "
                 f"rho = {spearman(surv, text_acc):.2f}")
    lines.append(f"- CER vs margin (RAVT - text), all {len(axes)} points: "
                 f"rho = {spearman([a['cer'] for a in axes], margin):.2f}")
    for ax in ("content", "photometric", "scan"):
        keep = [i for i, a in enumerate(axes) if a["axis"] != ax]
        lines.append(f"- leave-one-axis-out (drop {ax}): "
                     f"survival-text rho = {spearman([surv[i] for i in keep], [text_acc[i] for i in keep]):.2f}, "
                     f"CER-margin rho = {spearman([axes[i]['cer'] for i in keep], [margin[i] for i in keep]):.2f}")
    lines.append("")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
