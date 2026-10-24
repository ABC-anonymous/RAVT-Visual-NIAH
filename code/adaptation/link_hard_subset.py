"""从主 benchmark 建立 40% 腐蚀的 240-task 评测视图。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--source", default="data/test_bench_tmm_rh42")
    ap.add_argument("--out", default="data/test_bench_tmm_rh42_n40")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    source = root / args.source
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)

    linked = 0
    for task_dir in sorted(source.glob("Task_*")):
        meta_path = task_dir / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if round(float(meta.get("noise_level", -1.0)), 2) != 0.4:
            continue
        target = out / task_dir.name
        if not target.exists():
            os.symlink(task_dir.resolve(), target, target_is_directory=True)
        linked += 1

    n = sum(1 for path in out.glob("Task_*") if path.is_dir())
    if n != 240:
        raise SystemExit(f"40% 子集应有 240 个 task，实际 {n}")
    print(f"40% hard subset: {n} tasks -> {out}（本次确认 {linked} 个源 task）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
