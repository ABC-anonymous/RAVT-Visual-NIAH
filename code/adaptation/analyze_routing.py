"""路由规则的 leave-one-axis-out 评估：阈值在两条轴上选，第三条轴上评。

    python code/adaptation/analyze_routing.py --root .
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load(p):
    out = {}
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("record") == "task_eval":
            out[int(r["task_id"])] = r
    return out


def subset(d, key, lv):
    return [r for r in d.values() if round(float(r.get(key) or 0), 2) == round(lv, 2)]


def acc(rows):
    return 100 * sum(int(r["correct"]) for r in rows) / len(rows) if rows else float("nan")


def cer(rows):
    v = [r["ocr_cer"] for r in rows if r.get("ocr_cer") is not None]
    return sum(v) / len(v) if v else float("nan")  # 分数，不是百分数


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--jsonl", default="results/jsonl",
                    help="服务器上是 results/jsonl；本机分析旧结果时传 "
                         "results/jsonl")
    ap.add_argument("--out", default="results/adaptation/routing")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    j = root / a.jsonl
    base, ax = j / "eval_tmm_rh42", j / "eval_axes_rh42"

    char3 = load(base / "BM25-char3-OCR-tesseract--main.jsonl")
    ravt = [load(base / f"RAVT-rh42-s{s}-main.jsonl") for s in range(5)]
    c3_img = load(ax / "BM25-char3-OCR-tesseract--image.jsonl")
    c3_real = load(ax / "BM25-char3-OCR-tesseract--real.jsonl")
    r_img = [load(ax / f"RAVT-DEG--rh42-s{s}-image.jsonl") for s in range(5)]
    r_real = [load(ax / f"RAVT-DEG--rh42-s{s}-real.jsonl") for s in range(5)]

    points = []
    for lv in (0.0, 0.2, 0.4):
        t = subset(char3, "noise", lv)
        v = [acc(subset(r, "noise", lv)) for r in ravt]
        points.append({"axis": "content", "level": lv, "cer": cer(t),
                       "text": acc(t), "visual": statistics.mean(v)})
    for lv in (0.0, 0.1, 0.2, 0.3, 0.4):
        t = subset(c3_img, "image_level", lv)
        if not t:
            continue
        v = [acc(subset(r, "image_level", lv)) for r in r_img]
        points.append({"axis": "photometric", "level": lv, "cer": cer(t),
                       "text": acc(t), "visual": statistics.mean(v)})
    for lv in sorted({round(float(r.get("image_level") or 0), 2) for r in c3_real.values()}):
        t = subset(c3_real, "image_level", lv)
        v = [acc(subset(r, "image_level", lv)) for r in r_real]
        points.append({"axis": "scan", "level": lv, "cer": cer(t),
                       "text": acc(t), "visual": statistics.mean(v)})

    def regret(p, thr):
        pick = p["text"] if p["cer"] < thr else p["visual"]
        return max(p["text"], p["visual"]) - pick

    axes = sorted({p["axis"] for p in points})
    grid = [x / 100 for x in range(2, 41)]
    rows = []
    for held in axes:
        train = [p for p in points if p["axis"] != held]
        test = [p for p in points if p["axis"] == held]
        best = min(grid, key=lambda t: statistics.mean(regret(p, t) for p in train))
        ok = sum(1 for p in test if regret(p, best) < 1e-9)
        rows.append({"held_out": held, "threshold": best,
                     "accuracy": f"{ok}/{len(test)}",
                     "regret": statistics.mean(regret(p, best) for p in test)})
        print(f"held-out {held}: threshold {best:.2f} accuracy {ok}/{len(test)} "
              f"regret {rows[-1]['regret']:.2f}")

    in_sample_ok = sum(1 for p in points if regret(p, 0.10) < 1e-9)
    print(f"in-sample CER=0.10: {in_sample_ok}/{len(points)} "
          f"regret {statistics.mean(regret(p, 0.10) for p in points):.2f}")

    out = root / a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "route_loao.json").write_text(json.dumps(
        {"points": points, "loao": rows,
         "in_sample": {"accuracy": f"{in_sample_ok}/{len(points)}",
                       "regret": statistics.mean(regret(p, 0.10) for p in points)}},
        ensure_ascii=False, indent=1), encoding="utf-8")
    lines = ["# 路由规则 leave-one-axis-out", "",
             "| 留出轴 | 选中阈值 | 决策正确 | 平均后悔 |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['held_out']} | {r['threshold']:.2f} | {r['accuracy']} | "
                     f"{r['regret']:.2f} |")
    lines += ["", f"样本内（CER=0.10）：{in_sample_ok}/{len(points)}，"
                  f"平均后悔 {statistics.mean(regret(p, 0.10) for p in points):.2f}。"]
    (out / "route_loao.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"结果 -> {out/'route_loao.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
