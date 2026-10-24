"""从本地 DocVQA BEIR 子集导出「真实文档验证集」：页面图 + 查询 + qrels。

为什么用它：DocVQA 的页面是**真实扫描件**（不是我们合成退化的渲染图），
查询是真实用户问题，qrels 给出 query→page 的权威相关性。于是可以做一次
完全不依赖合成退化的外部验证：给定问题，找到它所在的页面。

难点（本脚本之后由 evaluate_real_pages.py 解决）：我们的模型是**行切分**的，
真实页面没有"行"的标注，必须先做文本行检测，再把相邻若干行拼成 mosaic。

用法:
    python code/ravt/export_real_pages.py \
        --pages 500 --out data/vidore_docvqa
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "data" / "docvqa_beir"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=40, help="导出多少张页面图")
    ap.add_argument("--out", default="real_docs")
    ap.add_argument("--max-side", type=int, default=1600,
                    help="页面图最长边缩放到多少像素（原图 1653x2339，缩小以省空间）")
    args = ap.parse_args()

    from datasets import Dataset

    corpus = Dataset.from_file(str(next((SRC / "corpus").glob("*.arrow"))))
    queries = Dataset.from_file(str(next((SRC / "queries").glob("*.arrow"))))
    qrels = Dataset.from_file(str(next((SRC / "qrels").glob("*.arrow"))))

    q2p: dict[int, list[int]] = {}
    for row in qrels:
        if row["score"] > 0:
            q2p.setdefault(int(row["query-id"]), []).append(int(row["corpus-id"]))
    qtext = {int(r["query-id"]): r["query"] for r in queries}

    # 挑页面：按查询顺序贪心——每个查询尽量带进来一个"新"页面，
    # 这样同样 40 张页面能覆盖尽量多的查询（检索集更有区分度）。
    chosen: list[int] = []
    chosen_set: set[int] = set()
    for q in sorted(q2p):
        if len(chosen) >= args.pages:
            break
        ps = [p for p in q2p[q] if p not in chosen_set]
        if not ps:
            continue
        for p in ps:
            if len(chosen) >= args.pages:
                break
            chosen.append(p)
            chosen_set.add(p)
    chosen_set = set(chosen)
    used_queries = [q for q, ps in q2p.items() if set(ps) & chosen_set]

    out = REPO / args.out
    (out / "pages").mkdir(parents=True, exist_ok=True)
    meta_rows = []
    for p in chosen:
        img = corpus[p]["image"].convert("L")
        if max(img.size) > args.max_side:
            scale = args.max_side / max(img.size)
            img = img.resize((int(img.width * scale), int(img.height * scale)))
        name = f"page_{p:04d}.png"
        img.save(out / "pages" / name)
        meta_rows.append({"page_id": p, "file": name, "w": img.width, "h": img.height})

    with (out / "queries.jsonl").open("w", encoding="utf-8") as fh:
        for q in used_queries:
            fh.write(json.dumps({"query_id": q, "query": qtext[q],
                                 "relevant_pages": sorted(set(q2p[q]) & chosen_set)},
                                ensure_ascii=False) + "\n")
    (out / "pages.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in meta_rows) + "\n",
        encoding="utf-8")

    size = sum(f.stat().st_size for f in (out / "pages").glob("*.png")) / 2 ** 20
    print(f"导出 {len(meta_rows)} 张页面图（{size:.1f} MB，最长边 {args.max_side}）-> {out}/pages")
    print(f"查询 {len(used_queries)} 条 -> {out}/queries.jsonl")
    print(f"页面索引 -> {out}/pages.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
