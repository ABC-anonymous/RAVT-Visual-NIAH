"""评测集审计：检查生成结果是否修复了已知缺陷，并可比对两份评测集是否等价。

用法:
    # 审计一份评测集（数据驱动版，比静态审计更有说服力）
    python code/ravexp/audit_benchmark.py --audit data/niah_robust_test

    # 比对两份评测集是否完全一致（用于验证并行/串行生成等价）
    python code/ravexp/audit_benchmark.py --compare data/_smoke/bench_serial data/_smoke/bench_par
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent  # 扁平包：数据/权重都放在脚本同级目录


def _abs(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else REPO_ROOT / path


def _file_digest(root: Path) -> dict[str, str]:
    out = {}
    for task in sorted(d for d in root.iterdir() if d.is_dir()):
        h = hashlib.sha256()
        for f in sorted(task.rglob("*")):
            if f.is_file():
                h.update(str(f.relative_to(task)).encode())
                h.update(f.read_bytes())
        out[task.name] = h.hexdigest()
    return out


def compare(a: Path, b: Path) -> int:
    da, db = _file_digest(a), _file_digest(b)
    only_a = sorted(set(da) - set(db))
    only_b = sorted(set(db) - set(da))
    diff = sorted(k for k in set(da) & set(db) if da[k] != db[k])
    print(f"A: {a}  task 数 {len(da)}")
    print(f"B: {b}  task 数 {len(db)}")
    print(f"仅 A 有: {len(only_a)}   仅 B 有: {len(only_b)}   内容不同: {len(diff)}")
    for k in (only_a + only_b + diff)[:10]:
        print(f"  {k}")
    if not (only_a or only_b or diff):
        print("✅ 两份评测集完全一致（含全部图像字节）")
        return 0
    print("❌ 存在差异")
    return 1


def audit(root: Path) -> int:
    tasks = sorted(d for d in root.iterdir() if d.is_dir())
    if not tasks:
        print(f"{root} 下没有 task 目录")
        return 1
    rows, dist, n_blocks, n_imgs, cells = [], 0, [], 0, Counter()
    metas = []
    for t in tasks:
        mp = t / "metadata.json"
        if not mp.exists():
            continue
        m = json.loads(mp.read_text(encoding="utf-8"))
        metas.append(m)
        rows.append(int(m.get("target_row_idx", -1)))
        dist += int(bool(m.get("has_distractor", False)))
        n_blocks.append(int(m.get("block_count", 0)))
        img_dir = t / "visual_haystack"
        n_imgs += len(list(img_dir.glob("*.png"))) if img_dir.exists() else 0
        cells[(m.get("noise_level"), int(m.get("block_count", 0)))] += 1

    print(f"评测集: {root}")
    print(f"  task 数: {len(metas)}   图像块总数: {n_imgs}")
    print(f"  块数取值: {sorted(set(n_blocks))}")
    row_hist = Counter(rows)
    print(f"  needle 行号分布: {dict(sorted(row_hist.items()))}")
    if len(row_hist) == 1:
        print("  ⚠️ needle 永远落在同一行 —— 这是旧版 benchmark 的已知缺陷，"
              "RAVT 的行对齐机制在评测中未被考察")
    else:
        print("  ✅ needle 覆盖了多行，行对齐机制会被真正考察")
    ratio = dist / max(len(metas), 1) * 100
    print(f"  含干扰项的 task 比例: {ratio:.1f}%（确定性分配时应约 50%）")

    counts = Counter(cells.values())
    print(f"  每个 (noise, blocks) 单元格的 task 数分布: {dict(sorted(counts.items()))}")
    if min(cells.values()) != max(cells.values()):
        print("  ⚠️ 单元格式样本量不一致，聚合表格时需要报告 N")
    corruptions = {m.get("corruptions") for m in metas}
    print(f"  噪声家族标记: {corruptions}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default=None)
    ap.add_argument("--compare", nargs=2, default=None)
    args = ap.parse_args()
    if args.compare:
        raise SystemExit(compare(_abs(args.compare[0]), _abs(args.compare[1])))
    if args.audit:
        raise SystemExit(audit(_abs(args.audit)))
    ap.print_help()


if __name__ == "__main__":
    main()
