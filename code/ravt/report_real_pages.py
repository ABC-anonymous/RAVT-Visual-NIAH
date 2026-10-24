"""公开基准（ViDoRe / DocVQA-BEIR）结果汇总。

输入是 evaluate_real_pages.py 写的 real_doc_results.json：
RAVT 的 5 个种子各自一项，汇总成均值±std；其余基线一项一行。
指标含 ViDoRe 惯用的 nDCG@5。

用法:
    python code/ravt/report_real_pages.py --json results/vidore_docvqa/real_doc_results.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

METRICS = ("ndcg@5", "ndcg@10", "recall@1", "recall@5", "recall@10")
RAVT_PAT = re.compile(r"^RAVT-(full_p14_rh42_s\d|deg_rh42_p14_s\d)$")


def mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return (0.0, 0.0)
    mu = sum(vals) / len(vals)
    if len(vals) < 2:
        return (mu, 0.0)
    var = sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)
    return (mu, math.sqrt(var))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", default="results/analysis/vidore_table.md")
    args = ap.parse_args()

    data = json.loads(Path(args.json).read_text(encoding="utf-8"))
    seeds: dict[str, list[dict]] = {}
    out_rows: list[tuple[str, dict[str, str]]] = []

    for name, v in data.items():
        m = RAVT_PAT.match(name)
        if m:
            key = "RAVT (content-trained)" if m.group(1).startswith("full") \
                else "RAVT + imaging degradation"
            seeds.setdefault(key, []).append(v["agg"])
        else:
            out_rows.append((name, {k: f"{v['agg'].get(k, 0.0):.1f}" for k in METRICS}))

    for key, aggs in seeds.items():
        cells = {}
        for k in METRICS:
            mu, sd = mean_std([a.get(k, 0.0) for a in aggs])
            cells[k] = f"{mu:.1f} ± {sd:.1f}"
        out_rows.append((f"{key} (n={len(aggs)} seeds)", cells))

    out_rows.sort(key=lambda r: -float(re.split(r"[ ±]", r[1]["ndcg@5"])[0]))
    md = ["| System | nDCG@5 | nDCG@10 | R@1 | R@5 | R@10 |",
          "| --- | --- | --- | --- | --- | --- |"]
    print(f"{'system':<42}" + "".join(f"{k:>12}" for k in METRICS))
    for name, cells in out_rows:
        print(f"{name:<42}" + "".join(f"{cells[k]:>12}" for k in METRICS))
        md.append(f"| {name} | " + " | ".join(cells[k] for k in METRICS) + " |")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nmarkdown 已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
