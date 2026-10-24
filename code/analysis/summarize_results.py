"""Aggregate evaluation JSONL into reproducible summary tables.

用法:
    python code/analysis/summarize_results.py \
        --run . \
        --old /path/to/old-run
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

NOISES = [0.0, 0.2, 0.4]


# --------------------------------------------------------------------------- #
def load_main(path: Path):
    """-> model -> noise -> {task_id: correct}, 以及 DER 用的 (hit_distractor, has_distractor)"""
    acc: dict[str, dict[float, dict[int, int]]] = defaultdict(lambda: defaultdict(dict))
    dis: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for f in sorted(path.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("record") != "task_eval":
                continue
            m, nl = r["model"], round(float(r["noise"]), 2)
            acc[m][nl][r["task_id"]] = int(r.get("correct", 0))
            if r.get("has_distractor"):
                dis[m].append((int(r.get("hit_distractor") or 0), 1))
    return acc, dis


def load_axes(path: Path):
    """-> model -> axis -> level -> {task_id: correct}"""
    out: dict[str, dict[str, dict[float, dict[int, int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict)))
    for f in sorted(path.glob("*.jsonl")):
        axis = "real" if f.stem.endswith("-real") else "image"
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("record") != "task_eval":
                continue
            lvl = r.get("image_level")
            if lvl is None:
                continue
            out[r["model"]][axis][round(float(lvl), 2)][r["task_id"]] = \
                int(r.get("correct", 0))
    return out


def pct(d: dict[int, int]) -> float:
    return 100.0 * sum(d.values()) / max(len(d), 1)


def mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return (0.0, 0.0)
    mu = sum(vals) / len(vals)
    if len(vals) < 2:
        return (mu, 0.0)
    var = sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)
    return (mu, math.sqrt(var))


def mcnemar(a: dict[int, int], b: dict[int, int]) -> tuple[int, int, float]:
    ids = set(a) & set(b)
    bo = sum(1 for i in ids if a[i] == 1 and b[i] == 0)
    co = sum(1 for i in ids if a[i] == 0 and b[i] == 1)
    n = bo + co
    if n == 0:
        return bo, co, 1.0
    k = min(bo, co)
    return bo, co, min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def boot_ci(diffs: list[float], n_boot: int = 4000, seed: int = 7) -> tuple[float, float]:
    if not diffs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(diffs)
    stats = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        stats.append(100.0 * s / n)
    stats.sort()
    return stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot) - 1]


def star(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="本次回传的目录（含 results/）")
    ap.add_argument("--old", default=None, help="44px 旧结果目录（做跨 bench 对照）")
    ap.add_argument("--axes-dir", default="eval_axes_rh42",
                    help="成像轴的 jsonl 目录名（旧结果是 eval_axes_full）")
    ap.add_argument("--out", default="results/analysis/final_results.md")
    args = ap.parse_args()

    run = Path(args.run)
    main_dir = run / "results" / "jsonl" / "eval_tmm_rh42"
    axes_dir = run / "results" / "jsonl" / args.axes_dir
    lat_dir = run / "results" / "jsonl" / "latency_rh42"
    acc, dis = load_main(main_dir)

    md: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        md.append(s)

    # ---------------- 分组：谁是一组种子 ----------------
    groups: dict[str, list[str]] = {"RAVT-rh42 (5 seeds)": [], "RAVT-DEG-rh42": []}
    singles: list[str] = []
    for m in sorted(acc):
        if re.match(r"^RAVT-rh42-s\d@", m):
            groups["RAVT-rh42 (5 seeds)"].append(m)
        elif m.startswith("RAVT-DEG- rh42"):
            groups["RAVT-DEG-rh42"].append(m)
        else:
            singles.append(m)
    base_runs = sorted(groups["RAVT-rh42 (5 seeds)"])
    # 组名里的种子数按实际数量补上（退化感知这次终于补齐了 5 个）
    for g in list(groups):
        n = len(groups[g])
        if n > 1 and "seeds" not in g:
            groups[f"{g} ({n} seeds)"] = groups.pop(g)

    emit(f"# 42px 定稿结果（{main_dir}）")
    emit()
    emit(f"主模型种子：{len(base_runs)} 个 —— " + ", ".join(base_runs))
    emit()
    if not base_runs:
        emit("⚠️ 这个目录里没有 RAVT-rh42-s* 结果，跳过表 2/表 3（只出成像轴/效率/真实文档）")
        emit()

    # ---------------- 表 1：主表 ----------------
    emit("## 表 1  42px 主测试集（Recall@1，每格 N=240；RAVT 行为 5 种子均值±标准差）")
    emit()
    emit("| 系统 | 0% | 20% | 40% | DER |")
    emit("| --- | --- | --- | --- | --- |")
    rows = []

    def der_of(models: list[str]) -> float:
        flat = [x for m in models for x in dis.get(m, [])]
        return 100.0 * sum(x[0] for x in flat) / max(len(flat), 1)

    for gname in groups:
        models = groups[gname]
        if not models:
            continue
        cells = []
        for nl in NOISES:
            mu, sd = mean_std([pct(acc[m].get(nl, {})) for m in models])
            cells.append(f"{mu:.1f} ± {sd:.1f}" if len(models) > 1 else f"{mu:.1f}")
        rows.append((gname, cells, der_of(models), True))

    for m in singles:
        cells = [f"{pct(acc[m].get(nl, {})):.1f}" for nl in NOISES]
        rows.append((m, cells, der_of([m]), False))
    rows.sort(key=lambda r: -float(r[1][2].split()[0]))   # 按 40% 档降序
    for name, cells, d, _g in rows:
        emit(f"| {name} | " + " | ".join(cells) + f" | {d:.1f} |")
    emit()

    # ---------------- 表 2：显著性（种子平均 vs 各基线） ----------------
    emit("## 表 2  与 42px RAVT（5 种子平均）的配对比较")
    emit()
    emit("| 对照系统 | Δ@0% | Δ@20% | Δ@40% | 40% 95% CI | McNemar p@40% | 同向种子 |")
    emit("| --- | --- | --- | --- | --- | --- | --- |")
    sig_rows = []
    for nl in NOISES:
        pass
    for m in ([] if not base_runs else singles + [g for g in groups if g != "RAVT-rh42 (5 seeds)"]):
        models = groups.get(m, [m])
        deltas = {}
        for nl in NOISES:
            ids = [i for i in acc[models[0]].get(nl, {})
                   if all(i in acc[s].get(nl, {}) for s in base_runs)]
            if not ids:
                deltas[nl] = (0.0, 0.0, 0.0, 1.0, 0)
                continue
            d_avg = [sum(acc[s][nl][i] for s in base_runs) / len(base_runs)
                     - acc[models[0]][nl][i] for i in ids]
            lo, hi = boot_ci(d_avg) if nl == 0.4 else (0.0, 0.0)
            med = sorted(base_runs, key=lambda s: pct(acc[s].get(nl, {})))[len(base_runs) // 2]
            _, _, p = mcnemar(acc[med][nl], acc[models[0]][nl])
            agree = sum(1 for s in base_runs
                        if sum(acc[s][nl][i] - acc[models[0]][nl][i] for i in ids) > 0)
            deltas[nl] = (100.0 * sum(d_avg) / len(ids), lo, hi, p, agree)
        sig_rows.append((m, deltas))
    sig_rows.sort(key=lambda r: -r[1][0.4][0])
    for m, d in sig_rows:
        cells = [f"{d[nl][0]:+.1f}" for nl in NOISES]
        lo, hi, p, agree = d[0.4][1], d[0.4][2], d[0.4][3], d[0.4][4]
        sig = "***" if (lo > 0 or hi < 0) and p < 0.001 else \
              "**" if (lo > 0 or hi < 0) and p < 0.01 else \
              "*" if (lo > 0 or hi < 0) and p < 0.05 else "n.s."
        emit(f"| {m} | " + " | ".join(cells) +
             f" | [{lo:+.1f}, {hi:+.1f}] {sig} | {p:.1e} {star(p)} | {agree}/{len(base_runs)} |")
    emit()

    # ---------------- 表 3：消融 ----------------
    emit("## 表 3  42px 消融（与同种子 RAVT-rh42-s0 配对）")
    emit()
    emit("| 变体 | 0% | 20% | 40% | Δ@40% | McNemar p@40% |")
    emit("| --- | --- | --- | --- | --- | --- |")
    ref = "RAVT-rh42-s0@main"
    abl = [m for m in acc if m.startswith("w/o")]
    abl.sort(key=lambda m: -pct(acc[m].get(0.4, {})))
    emit(f"| RAVT-rh42-s0（参照） | " +
         " | ".join(f"{pct(acc[ref].get(nl, {})):.1f}" for nl in NOISES) + " | - | - |")
    for m in abl:
        cur = acc[m]
        delta = pct(cur.get(0.4, {})) - pct(acc[ref].get(0.4, {}))
        _, _, p = mcnemar(acc[ref].get(0.4, {}), cur.get(0.4, {}))
        emit(f"| {m} | " + " | ".join(f"{pct(cur.get(nl, {})):.1f}" for nl in NOISES) +
             f" | {delta:+.1f} | {p:.1e} {star(p)} |")
    emit()

    # ---------------- 表 4：成像轴 ----------------
    if axes_dir.exists():
        ax = load_axes(axes_dir)
        for axis, title in (("image", "表 4a  成像（photometric）退化轴，42px"),
                            ("real", "表 4b  真实扫描（realistic-scan）轴，42px")):
            models = [m for m in ax if axis in ax[m]]
            if not models:
                continue
            levels = sorted({l for m in models for l in ax[m][axis]})
            emit(f"## {title}")
            emit()
            emit("| 系统 | " + " | ".join(f"{l:.1f}" for l in levels) + " | 退化幅度(0→最大) |")
            emit("| --- |" + " --- |" * (len(levels) + 1))
            order = sorted(models, key=lambda m: -pct(ax[m][axis][levels[-1]]))
            for m in order:
                vals = [pct(ax[m][axis].get(l, {})) for l in levels]
                drop = vals[0] - vals[-1]
                emit(f"| {m} | " + " | ".join(f"{v:.1f}" for v in vals) +
                     f" | {drop:+.1f} |")
            emit()

            # 与 5 种子 RAVT / RAVT-DEG 的配对比较（最严档）
            hard = levels[-1]
            for gname, pat in (("RAVT-rh42", r"^RAVT-rh42-s\d@"),
                               ("RAVT-DEG-rh42", r"^RAVT-DEG- rh42-s\d@")):
                seeds = sorted(m for m in ax if re.match(pat, m) and axis in ax[m])
                if not seeds:
                    continue
                emit(f"与 {gname}（{len(seeds)} 种子平均）在最严档 {hard:.2f} 的配对比较：")
                emit()
                emit("| 对照 | Δ | 95% CI | McNemar p | 同向种子 |")
                emit("| --- | --- | --- | --- | --- |")
                rows_cmp = []
                for m in sorted(ax):
                    if m in seeds or axis not in ax[m]:
                        continue
                    ids = [i for i in ax[seeds[0]][axis].get(hard, {})
                           if all(i in ax[s][axis].get(hard, {}) for s in seeds)
                           and i in ax[m][axis].get(hard, {})]
                    if not ids:
                        continue
                    d = [sum(ax[s][axis][hard][i] for s in seeds) / len(seeds)
                         - ax[m][axis][hard][i] for i in ids]
                    lo, hi = boot_ci(d)
                    med = sorted(seeds, key=lambda s: pct(ax[s][axis].get(hard, {})))[len(seeds) // 2]
                    _, _, p = mcnemar(ax[med][axis][hard], ax[m][axis][hard])
                    agree = sum(1 for s in seeds
                                if sum(ax[s][axis][hard][i] - ax[m][axis][hard][i] for i in ids) > 0)
                    rows_cmp.append((m, 100.0 * sum(d) / len(ids), lo, hi, p, agree))
                rows_cmp.sort(key=lambda r: -r[1])
                for m, delta, lo, hi, p, agree in rows_cmp:
                    sig = "***" if (lo > 0 or hi < 0) and p < 0.001 else \
                          "**" if (lo > 0 or hi < 0) and p < 0.01 else \
                          "*" if (lo > 0 or hi < 0) and p < 0.05 else "n.s."
                    emit(f"| {m} | {delta:+.1f} | [{lo:+.1f}, {hi:+.1f}] {sig} | "
                         f"{p:.1e} {star(p)} | {agree}/{len(seeds)} |")
                emit()

    # ---------------- 表 5：效率 ----------------
    if lat_dir.exists():
        emit("## 表 5  42px 效率（precomputed 协议，在线查询侧）")
        emit()
        emit("| 系统 | 查询 P50 ms | KB/doc |")
        emit("| --- | --- | --- |")
        stats = []
        for f in sorted(lat_dir.glob("*.jsonl")):
            lat, bpd = [], []
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("record") != "task_eval" or r.get("protocol") != "precomputed":
                    continue
                lat.append(r["latency_ms"]["query"])
                bpd.append(r.get("bytes_per_doc", 0))
            if lat:
                lat.sort()
                stats.append((Path(f).stem, lat[len(lat) // 2], sum(bpd) / len(bpd) / 1024))
        stats.sort(key=lambda x: x[1])
        for name, p50, kb in stats:
            emit(f"| {name} | {p50:.1f} | {kb:.1f} |")
        emit()

    # ---------------- 表 6：真实文档 ----------------
    rd = run / "results" / "real_doc_rh42" / "real_doc_results.json"
    if rd.exists():
        d = json.loads(rd.read_text(encoding="utf-8"))
        emit("## 表 6  真实扫描文档（42px 重渲染，DocVQA 40 页 / 31 查询）")
        emit()
        emit("| 系统 | R@1 | R@5 | R@10 | nDCG@10 |")
        emit("| --- | --- | --- | --- | --- |")
        for name, v in d.items():
            a = v["agg"]
            emit(f"| {name} | {a['recall@1']:.1f} | {a['recall@5']:.1f} | "
                 f"{a['recall@10']:.1f} | {a['ndcg@10']:.1f} |")
        emit()

    # ---------------- 表 7：44px -> 42px 同种子对照 ----------------
    if args.old:
        old_acc, _ = load_main(Path(args.old) / "results" / "jsonl" / "eval_tmm")
        pairs = []
        for s in range(5):
            a = f"RAVT-s{s}@main"
            b = f"RAVT-rh42-s{s}@main"
            if a in old_acc and b in acc:
                pairs.append((a, b))
        if pairs:
            emit("## 表 7  行对齐：44px → 42px（同种子 s0–s4，逐 task 配对）")
            emit()
            emit("| 配置 | 0% | 20% | 40% |")
            emit("| --- | --- | --- | --- |")
            for tag, idx in (("44px", 0), ("42px", 1)):
                cells = []
                for nl in NOISES:
                    mu, sd = mean_std([pct((old_acc if idx == 0 else acc)[b if idx == 1 else a]
                                           .get(nl, {})) for a, b in pairs])
                    cells.append(f"{mu:.1f} ± {sd:.1f}")
                emit(f"| {tag}（5 seeds） | " + " | ".join(cells) + " |")
            cells = []
            for nl in NOISES:
                d = [pct(acc[b].get(nl, {})) - pct(old_acc[a].get(nl, {})) for a, b in pairs]
                mu, sd = mean_std(d)
                t = mu / (sd / math.sqrt(len(d))) if sd > 0 else float("inf")
                p = math.erfc(abs(t) / math.sqrt(2)) if sd > 0 else 0.0
                cells.append(f"{mu:+.1f} ± {sd:.1f} (t={t:.1f}, p={p:.1e}) {star(p)}")
            emit("| 差值 | " + " | ".join(cells) + " |")
            emit()
            # 文本基线跨 bench 应当完全不变（读 oracle 文本）
            emit("对照（文本基线，读 oracle 文本、不看图，应 Δ=0）：")
            emit()
            emit("| 系统 | Δ@40% |")
            emit("| --- | --- |")
            for m in sorted(set(old_acc) & set(acc)):
                if m.startswith(("BM25", "BGE")) or "TXT-" in m:
                    d = pct(acc[m].get(0.4, {})) - pct(old_acc[m].get(0.4, {}))
                    emit(f"| {m} | {d:+.1f} |")
            for m in ("ColPali@main", "ColQwen2@main"):
                if m in old_acc and m in acc:
                    d = pct(acc[m].get(0.4, {})) - pct(old_acc[m].get(0.4, {}))
                    emit(f"| {m} | {d:+.1f} |")
            emit()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nmarkdown 已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
