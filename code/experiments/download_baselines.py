"""阶段 C 的准备步骤：在**本机**（有外网）把新基线权重下好，打包上传到无外网的服务器。

下三样（体积已核对，2026-10）：
    vidore/colqwen2.5-base      7.01 GB  —— ColQwen2.5 的基座（Qwen2.5-VL-3B 继续预训练版）
    vidore/colqwen2.5-v0.2      ~250 MB  —— ColQwen2.5 的 LoRA 适配器
    vidore/colpali-v1.3         ~250 MB  —— ColPali 家族更晚的适配器（与服务器已有基座共用）

以及 colpali-engine 0.3.18 的 wheel（含 ColQwen2_5 类）与 peft 的 wheel（离线补依赖用）。

    python code/experiments/download_baselines.py --dest weights_upload

下载完的目录结构已经按服务器路径排好，解压到 $base 即可：
    weights_upload/models_local/colqwen2.5-base/...
    weights_upload/models_local/colqwen2.5/...
    weights_upload/colpali_offline_weights/adapter_v13/...
    weights_upload/wheels/*.whl
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

#: (仓库, 目标子路径, allow_patterns)
REPOS = [
    ("vidore/colqwen2.5-base", "models_local/colqwen2.5-base",
     ["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model", "*.py"]),
    ("vidore/colqwen2.5-v0.2", "models_local/colqwen2.5",
     ["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model"]),
    ("vidore/colpali-v1.3", "colpali_offline_weights/adapter_v13",
     ["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model"]),
]

WHEELS = ["colpali-engine==0.3.18", "peft"]


def dir_size(p: Path) -> str:
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return f"{total / 1024 ** 3:.2f} GB" if total > 1024 ** 3 else f"{total / 1024 ** 2:.0f} MB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="weights_upload")
    ap.add_argument("--skip-wheels", action="store_true")
    a = ap.parse_args()
    dest = Path(a.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("需要 huggingface_hub：pip install -U huggingface_hub")
        return 1

    for repo, sub, patterns in REPOS:
        target = dest / sub
        print(f"\n=== {repo} -> {target}")
        snapshot_download(repo_id=repo, local_dir=str(target), allow_patterns=patterns)
        print(f"    大小 {dir_size(target)}")

    if not a.skip_wheels:
        wdir = dest / "wheels"
        wdir.mkdir(parents=True, exist_ok=True)
        for spec in WHEELS:
            print(f"\n=== wheel: {spec}")
            subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps",
                            "--only-binary=:all:", "-d", str(wdir), spec], check=False)
    print("\n完成。打包：")
    print(f"  tar -cf baseline_weights.tar -C {dest.parent} {dest.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
