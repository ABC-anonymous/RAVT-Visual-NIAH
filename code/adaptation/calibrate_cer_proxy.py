"""探针 A：无参考 OCR / 图像质量代理。

部署时拿不到干净真值，算不了 CER；但路由规则需要 CER。这里检验五类无参考量：

  OOV       OCR 词中不在参考词表里的比例（空输出记 1.0，不能被跳过）
  CONF      tesseract 平均词置信度（image_to_data）
  AGREE     1 - CER(tesseract, rapidocr)，双引擎一致度
  SHARP     归一化拉普拉斯方差（模糊/失焦会降）
  CONTRAST  灰度标准差 / 255（光照、透印会变）

合成集上有真值 CER，用来按通道看相关性并做校准；真实扫描件只报代理，
且三档在**同一批 tile** 上比较。

    python code/adaptation/calibrate_cer_proxy.py --root .
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TOKEN = re.compile(r"[a-z]{2,}")


def norm(s: str) -> str:
    return " ".join(s.split())


def tokenize(s: str) -> list[str]:
    return TOKEN.findall(s.lower())


def oov_of(text: str, vocab: set[str]) -> float:
    """空输出是彻底失败，记 1.0；不能因为没词就跳过（否则最差的块被排除）。"""
    tk = tokenize(text)
    if not tk:
        return 1.0
    return sum(1 for w in tk if w not in vocab) / len(tk)


def ranks(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    out = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = r
        i = j + 1
    return out


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 3:
        return float("nan")
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx and dy else float("nan")


def auc(score: list[float], positive: list[bool]) -> float:
    """score 越大越像正类时的 AUC。"""
    pos = [s for s, p in zip(score, positive) if p]
    neg = [s for s, p in zip(score, positive) if not p]
    if not pos or not neg:
        return float("nan")
    r = ranks(score)
    rank_pos = sum(rr for rr, p in zip(r, positive) if p)
    return (rank_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def build_vocab(texts, max_tokens: int) -> tuple[set[str], int]:
    counter: Counter[str] = Counter()
    total = 0
    for txt in texts:
        tk = tokenize(txt)
        counter.update(tk)
        total += len(tk)
        if total >= max_tokens:
            break
    return set(counter), total


def image_stats(path: Path) -> dict:
    """无参考图像质量：模糊、对比度、墨迹比例。"""
    try:
        import numpy as np
        from PIL import Image
    except Exception:                                            # noqa: BLE001
        return {}
    try:
        with Image.open(path) as im:
            g = np.asarray(im.convert("L"), dtype=np.float32)
    except Exception:                                            # noqa: BLE001
        return {}
    if g.ndim != 2 or min(g.shape) < 8:
        return {}
    lap = (4.0 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1]
           - g[1:-1, :-2] - g[1:-1, 2:])
    var = float(g.var())
    return {
        "sharp": float(lap.var() / (var + 1e-6)),
        "contrast": float(g.std() / 255.0),
        "ink": float((g < 128).mean()),
    }


def text_for(engine, job: tuple) -> str:
    """优先读评测时留下的 tesseract 缓存（同一后端、同一放大倍数）。"""
    task, _i, path = job
    if engine.cache_dir is not None:
        cf = Path(engine.cache_dir) / task.path.name / f"{path.stem}.txt"
        if cf.exists():
            return cf.read_text(encoding="utf-8", errors="replace").strip()
    return engine._ocr_image(path)


def parallel_texts(engine, jobs: list[tuple], workers: int) -> list[str]:
    if engine.backend == "tesseract" and workers > 1 and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(lambda j: text_for(engine, j), jobs))
    return [text_for(engine, j) for j in jobs]


def confidence_of(engine, path: Path) -> float | None:
    try:
        import pytesseract
        from PIL import Image
    except Exception:                                            # noqa: BLE001
        return None
    with Image.open(path) as im:
        im = im.convert("RGB")
        if engine.scale != 1.0:
            im = im.resize((int(im.width * engine.scale), int(im.height * engine.scale)),
                           Image.LANCZOS)
        try:
            d = pytesseract.image_to_data(
                im.convert("L"), lang=engine.lang,
                config=f"--psm {engine.psm} -c preserve_interword_spaces=1",
                output_type=pytesseract.Output.DICT)
        except Exception:                                        # noqa: BLE001
            return None
    vals = [float(c) for c, txt in zip(d.get("conf", []), d.get("text", []))
            if txt.strip() and float(c) >= 0]
    return sum(vals) / len(vals) if vals else None


def spread(jobs: list[tuple], n: int) -> list[tuple]:
    """跨 task 均匀抽样，避免全部落在前几个 task 上。"""
    if n <= 0 or not jobs:
        return []
    if n >= len(jobs):
        return list(jobs)
    step = len(jobs) / n
    return [jobs[min(len(jobs) - 1, int(i * step))] for i in range(n)]


def sample_tasks(pool: list, n: int, seed: int) -> list:
    """按块数分层抽样，保持 20–1000 块的覆盖。"""
    if len(pool) <= n:
        return sorted(pool, key=lambda t: t.task_id)
    rng = random.Random(seed)
    groups: dict[int, list] = {}
    for t in pool:
        groups.setdefault(t.blocks, []).append(t)
    keys = sorted(groups)
    picked: list = []
    i = 0
    while len(picked) < n and keys:
        k = keys[i % len(keys)]
        if groups[k]:
            picked.append(groups[k].pop(rng.randrange(len(groups[k]))))
        i += 1
        if i > 100000:
            break
    return sorted(picked, key=lambda t: t.task_id)


def run_synthetic_level(engine, rap, tasks: list, level: float, axis: str, args,
                        rng: random.Random, vocab: set[str]) -> dict:
    from scorers import _edit_distance

    if not tasks:
        return {"axis": axis, "level": level, "error": "no tasks"}

    jobs: list[tuple] = []
    for t in tasks:
        n = len(t.image_paths)
        idx = list(range(n)) if n <= args.blocks else sorted(rng.sample(range(n), args.blocks))
        for i in idx:
            jobs.append((t, i, t.image_paths[i]))
    print(f"  [{axis}={level:g}] {len(tasks)} tasks, {len(jobs)} blocks")

    t0 = time.time()
    texts = parallel_texts(engine, jobs, args.workers)
    print(f"      tesseract {len(jobs)} blocks 用时 {time.time() - t0:.1f}s")

    rows = []
    for (t, i, _p), txt in zip(jobs, texts):
        gt = ""
        if i < len(t.text_paths):
            try:
                gt = norm(t.text_paths[i].read_text(encoding="utf-8", errors="replace"))
            except Exception:                                    # noqa: BLE001
                gt = ""
        tk = tokenize(txt)
        rows.append({
            "axis": axis,
            "level": level,
            "task_id": t.task_id,
            "block": i,
            "cer": (_edit_distance(norm(txt), gt) / len(gt)) if gt else None,
            "oov": oov_of(txt, vocab),
            "n_tokens": len(tk),
            "conf": None,
            "agree": None,
            "sharp": None,
            "contrast": None,
        })

    idx_of = {id(j): k for k, j in enumerate(jobs)}
    for j in jobs:
        st = image_stats(j[2])
        if st:
            rows[idx_of[id(j)]].update({k: st[k] for k in ("sharp", "contrast")})

    for j in spread(jobs, args.conf):
        c = confidence_of(engine, j[2])
        if c is not None:
            rows[idx_of[id(j)]]["conf"] = c

    if rap is not None:
        for j in spread(jobs, args.agree):
            k = idx_of[id(j)]
            rtxt = rap._ocr_image(j[2])
            a, b = norm(texts[k]), norm(rtxt)
            # 两个引擎同时退化成空/极短输出时会"互相一致"，这种对不算
            if max(len(a), len(b)) >= args.min_chars:
                rows[k]["agree"] = 1.0 - _edit_distance(a, b) / max(1, max(len(a), len(b)))

    def mean(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "axis": axis,
        "level": level,
        "n_tasks": len(tasks),
        "n_blocks": len(rows),
        "cer": mean("cer"),
        "oov": mean("oov"),
        "conf": mean("conf"),
        "agree": mean("agree"),
        "sharp": mean("sharp"),
        "contrast": mean("contrast"),
        "empty_rate": sum(1 for r in rows if r["n_tokens"] == 0) / len(rows),
        "rows": rows,
    }


def real_tile_dirs(root: Path, vidore: str) -> dict[str, Path]:
    return {name: root / vidore / sub / "tiles"
            for sub, name in (("", "clean"), ("deg02", "0.2"), ("deg04", "0.4"))}


def run_real(dirs: dict[str, Path], rap, args) -> list[dict]:
    """三档只在同一批 tile 上比较。词表取自全部 clean tile（不是被比较的那批）。"""
    from scorers import _edit_distance

    stems: dict[str, set[str]] = {}
    for name, d in dirs.items():
        stems[name] = {p.stem for p in d.glob("*.txt")} if d.is_dir() else set()
    common = sorted(set.intersection(*stems.values())) if all(stems.values()) else []
    if not common:
        print("  ⚠️  三档没有共同 tile，真实件跳过（各档集合不同，不能直接比）")
        return []

    rng = random.Random(args.seed)
    sample = common if len(common) <= args.real_txt else rng.sample(common, args.real_txt)
    print(f"  共有 tile {len(common)}，抽样 {len(sample)}（三档同一批）")
    conf_ids = sample[: args.real_conf]
    agree_ids = sample[: args.real_agree]

    # 词表取自 clean 档、但排除被比较的这批 tile，避免"用自己算自己"得到 OOV=0
    held = set(sample)
    pool = [p for p in sorted(dirs["clean"].glob("*.txt")) if p.stem not in held]
    vocab, n_tok = build_vocab(
        [p.read_text(encoding="utf-8", errors="replace") for p in pool], args.vocab_tokens)
    print(f"  真实件参考词表 {len(vocab)} 词 / {n_tok} token（取自 {len(pool)} 个未参与比较的 clean tile）")

    out = []
    for name, d in dirs.items():
        oovs, confs, agrees, sharps, conts, empties = [], [], [], [], [], 0
        for s in sample:
            txt = (d / f"{s}.txt").read_text(encoding="utf-8", errors="replace")
            o = oov_of(txt, vocab)
            oovs.append(o)
            if o >= 1.0:
                empties += 1
            st = image_stats(d / f"{s}.png")
            if st:
                sharps.append(st["sharp"])
                conts.append(st["contrast"])
        for s in conf_ids:
            c = confidence_of_simple(d / f"{s}.png", 2.0)
            if c is not None:
                confs.append(c)
        if rap is not None:
            for s in agree_ids:
                png, txt = d / f"{s}.png", d / f"{s}.txt"
                if not png.exists():
                    continue
                a = norm(txt.read_text(encoding="utf-8", errors="replace"))
                b = norm(rap._ocr_image(png))
                if max(len(a), len(b)) >= args.min_chars:
                    agrees.append(1.0 - _edit_distance(a, b) / max(1, max(len(a), len(b))))
            agree_cov = len(agrees) / max(1, len(agree_ids))
        else:
            agree_cov = None
        out.append({
            "grade": name,
            "n_tiles": len(common),
            "n_sampled": len(sample),
            "oov": sum(oovs) / len(oovs) if oovs else None,
            "conf": sum(confs) / len(confs) if confs else None,
            "agree": sum(agrees) / len(agrees) if agrees else None,
            "agree_cov": agree_cov,
            "sharp": sum(sharps) / len(sharps) if sharps else None,
            "contrast": sum(conts) / len(conts) if conts else None,
            "empty_rate": empties / len(sample),
        })
        def f(v, nd=3):
            return "—" if v is None else f"{v:.{nd}f}"
        print(f"  [real={name}] same-tile oov={f(out[-1]['oov'])} "
              f"conf={f(out[-1]['conf'], 1)} agree={f(out[-1]['agree'])} "
              f"empty={f(out[-1]['empty_rate'], 2)}")
    return out


def confidence_of_simple(png: Path, scale: float) -> float | None:
    try:
        import pytesseract
        from PIL import Image
    except Exception:                                            # noqa: BLE001
        return None
    if not png.exists():
        return None
    with Image.open(png) as im:
        im = im.convert("RGB")
        im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
        try:
            d = pytesseract.image_to_data(im.convert("L"), lang="eng",
                                          config="--psm 6 -c preserve_interword_spaces=1",
                                          output_type=pytesseract.Output.DICT)
        except Exception:                                        # noqa: BLE001
            return None
    vals = [float(c) for c, txt in zip(d.get("conf", []), d.get("text", []))
            if txt.strip() and float(c) >= 0]
    return sum(vals) / len(vals) if vals else None


def calibrate(rows: list[dict]) -> dict:
    """按 (axis, task) 分组划分，拟合 CER ~ 五个 no-reference proxies。"""
    keys = ("conf", "agree", "sharp", "contrast")
    use = [r for r in rows if r["cer"] is not None and all(r[k] is not None for k in keys)]
    if len(use) < 40:
        return {"n": len(use), "note": "可用块太少，跳过校准"}
    try:
        import numpy as np
    except Exception:                                            # noqa: BLE001
        return {"n": len(use), "note": "没有 numpy，跳过校准"}
    groups = sorted({(r.get("axis"), r["task_id"]) for r in use})
    test_groups = set(groups[::2])
    tr = [r for r in use if (r.get("axis"), r["task_id"]) not in test_groups]
    te = [r for r in use if (r.get("axis"), r["task_id"]) in test_groups]
    if len(tr) < 20 or len(te) < 20:
        tr, te = use[::2], use[1::2]

    def design(rs):
        return np.array([[1.0, r["oov"], r["conf"] / 100.0, r["agree"],
                          math.log1p(r["sharp"]) / 10.0, r["contrast"] * 10.0] for r in rs])

    X, y = design(tr), np.array([r["cer"] for r in tr])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = design(te) @ coef
    true = np.array([r["cer"] for r in te])
    mae = float(np.mean(np.abs(pred - true)))
    ss_res = float(np.sum((pred - true) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))

    # 单特征基线：只用 OOV 的线性拟合，看看多特征是否值得
    X1 = np.array([[1.0, r["oov"]] for r in tr])
    c1, *_ = np.linalg.lstsq(X1, y, rcond=None)
    p1 = np.array([[1.0, r["oov"]] for r in te]) @ c1
    return {
        "n": len(use), "n_groups": len(groups),
        "n_train": len(tr), "n_test": len(te),
        "split": "grouped_axis_task",
        "coef": [float(c) for c in coef],
        "mae_test": mae,
        "r2_test": 1.0 - ss_res / ss_tot if ss_tot else float("nan"),
        "spearman_pred_true": spearman(list(pred), list(true)),
        "mae_test_oov_only": float(np.mean(np.abs(p1 - true))),
        "spearman_oov_only": spearman(list(p1), list(true)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--out", default="results/adaptation/cer_proxy")
    ap.add_argument("--content-bench", default="data/test_bench_tmm_rh42")
    ap.add_argument("--image-bench", default="data/test_image_rh42")
    ap.add_argument("--vidore", default="results/vidore_docvqa")
    ap.add_argument("--content-levels", type=float, nargs="+", default=[0.0, 0.2, 0.4])
    ap.add_argument("--image-levels", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.4])
    ap.add_argument("--tasks", type=int, default=40)
    ap.add_argument("--blocks", type=int, default=40)
    ap.add_argument("--agree", type=int, default=60)
    ap.add_argument("--conf", type=int, default=60)
    ap.add_argument("--min-chars", type=int, default=8,
                    help="一致度只在较长一侧 ≥ 该字符数时计算")
    ap.add_argument("--real-txt", type=int, default=1500)
    ap.add_argument("--real-conf", type=int, default=60)
    ap.add_argument("--real-agree", type=int, default=80)
    ap.add_argument("--vocab-tasks", type=int, default=120)
    ap.add_argument("--vocab-tokens", type=int, default=400_000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--skip-real", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    for p in (root, root / "code" / "ravt"):
        sys.path.insert(0, str(p))
    from datasets_eval import list_tasks
    from scorers import OcrEngine  # noqa: F401

    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    content = root / args.content_bench
    image = root / args.image_bench
    print(f"root={root}\n合成集词表来源: {content}（{args.vocab_tasks} tasks）")
    content_tasks = list_tasks(content, strict=True) if content.exists() else []
    image_tasks = list_tasks(image, strict=True) if image.exists() else []
    print(f"任务数: content {len(content_tasks)} / image {len(image_tasks)}")

    clean = [t for t in content_tasks if round(t.noise, 4) == 0.0][: args.vocab_tasks]
    texts: list[str] = []
    for k, t in enumerate(clean, 1):
        for p in t.text_paths[:60]:
            try:
                texts.append(p.read_text(encoding="utf-8", errors="replace"))
            except Exception:                                    # noqa: BLE001
                pass
        if k % 30 == 0:
            print(f"  词表读取 {k}/{len(clean)} tasks，{len(texts)} 段")
    vocab, n_vocab_tokens = build_vocab(texts, args.vocab_tokens)
    print(f"词表 {len(vocab)} 词 / 语料 {n_vocab_tokens} token")

    tess = OcrEngine(backend="tesseract", cache_dir=str(root / "results" / "ocr_cache"),
                     scale=2.0, psm=6, lang="eng")
    rap = None
    if args.agree > 0 or args.real_agree > 0:
        try:
            rap = OcrEngine(backend="rapidocr", cache_dir=None)
        except Exception as e:                                   # noqa: BLE001
            print(f"⚠️  RapidOCR 不可用，跳一致度：{e}")

    results: list[dict] = []
    for axis, bench, levels in (("noise", content, args.content_levels),
                                ("image", image, args.image_levels)):
        if not bench.exists():
            print(f"⚠️  缺 {bench}，跳过 {axis} 轴")
            continue
        print(f"== {axis} 轴 ==")
        all_tasks = content_tasks if axis == "noise" else image_tasks
        for lv in levels:
            pool = [t for t in all_tasks
                    if round((t.noise if axis == "noise" else t.image_level), 4) == round(lv, 4)]
            picked = sample_tasks(pool, args.tasks, args.seed)
            try:
                r = run_synthetic_level(tess, rap, picked, lv, axis, args, rng, vocab)
            except Exception as e:                               # noqa: BLE001
                print(f"  [{axis}={lv:g}] 失败：{e}")
                r = {"axis": axis, "level": lv, "error": repr(e)}
            results.append(r)

    real = []
    if not args.skip_real:
        print("== 真实扫描件（三档同一批 tile）==")
        try:
            real = run_real(real_tile_dirs(root, args.vidore), rap, args)
        except Exception as e:                                   # noqa: BLE001
            print(f"  真实件失败：{e}")

    ok = [r for r in results if "error" not in r]
    summary: dict = {"config": vars(args), "levels": [], "real": real,
                     "vocab_size": len(vocab)}
    for r in ok:
        entry = {k: r[k] for k in ("axis", "level", "n_tasks", "n_blocks", "cer", "oov",
                                   "conf", "agree", "sharp", "contrast", "empty_rate")}
        cers = [x["cer"] for x in r["rows"] if x["cer"] is not None]
        oovs = [x["oov"] for x in r["rows"] if x["cer"] is not None]
        entry["auc_oov"] = auc(oovs, [c > 0.10 for c in cers])
        summary["levels"].append(entry)

    stats: dict = {}
    for axis in ("noise", "image"):
        pts = [r for r in ok if r["axis"] == axis and r["cer"] is not None]
        if len(pts) >= 3:
            for key in ("oov", "conf", "agree", "sharp", "contrast"):
                sub = [p for p in pts if p[key] is not None]
                if len(sub) >= 3:
                    stats[f"level_spearman_{key}_{axis}"] = spearman(
                        [p[key] for p in sub], [p["cer"] for p in sub])

    block_rows = [x for r in ok for x in r["rows"] if x["cer"] is not None]
    if block_rows:
        stats["block_spearman_oov"] = spearman([x["oov"] for x in block_rows],
                                               [x["cer"] for x in block_rows])
        stats["block_auc_oov_cer10"] = auc([x["oov"] for x in block_rows],
                                           [x["cer"] > 0.10 for x in block_rows])
    cal = calibrate(block_rows)
    summary["stats"] = stats
    summary["calibration"] = cal
    summary["calibration_rows"] = block_rows

    lines = ["# 探针 A：无参考 OCR / 图像质量代理", "",
             f"合成集参考词表：{len(vocab)} 词（{args.vocab_tasks} 个干净 task，"
             f"{n_vocab_tokens} token）", "",
             "| 轴 | 档位 | blocks | CER | OOV | 置信度 | 一致度 | SHARP | CONTRAST | 空输出 |",
             "|---|---|---|---|---|---|---|---|---|---|"]

    def f(v, nd=3):
        return "—" if v is None else f"{v:.{nd}f}"

    for e in summary["levels"]:
        lines.append(
            f"| {e['axis']} | {e['level']:g} | {e['n_blocks']} | {f(e['cer'])} | "
            f"{f(e['oov'])} | {f(e['conf'], 1)} | {f(e['agree'])} | {f(e['sharp'], 1)} | "
            f"{f(e['contrast'])} | {f(e['empty_rate'], 2)} |")

    lines += ["", "## 每档的相关性（档位均值，n = 档位数）", ""]
    for k, v in stats.items():
        lines.append(f"- {k} = {v:.3f}")
    lines += ["", "## 校准：CER ~ OOV + CONF + AGREE + SHARP + CONTRAST"
              "（按 axis × task 分组留出）", ""]
    for k, v in cal.items():
        lines.append(f"- {k} = {v:.4f}" if isinstance(v, float) else f"- {k} = {v}")

    if real:
        lines += ["", "## 真实扫描件（三档同一批 tile；词表取自未参与比较的 clean tile）", "",
                  "| 档位 | 共有 tiles | 抽样 | OOV | 置信度 | 一致度 | 一致度覆盖 | SHARP | CONTRAST | 空输出 |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in real:
            lines.append(f"| {r['grade']} | {r['n_tiles']} | {r['n_sampled']} | "
                         f"{f(r['oov'])} | {f(r['conf'], 1)} | {f(r['agree'])} | "
                         f"{f(r.get('agree_cov'), 2)} | {f(r['sharp'], 1)} | "
                         f"{f(r['contrast'])} | {f(r['empty_rate'], 2)} |")

    lines += ["", "## 读法", "",
              "- 每条轴单独看：内容污染轴 OOV 应该最强；成像退化轴看 SHARP/CONTRAST/一致度；",
              "- 校准一栏给出多特征与只用 OOV 的留出 MAE 对比，多特征明显更好才写进路由；",
              "- 真实件表里三档是同一批 tile，空输出率必须单列，否则最差的块会被跳过。"]

    (out_dir / "cer_proxy.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "cer_proxy.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n结果 -> {out_dir/'cer_proxy.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
