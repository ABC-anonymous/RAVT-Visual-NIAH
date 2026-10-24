"""基准目录解析：让脚本无论解压到哪里，都能找到 dataset/ 与 clip_local_weights_224/。

规则（按优先级）：
1. 命令行 --base-dir 指定的目录
2. 脚本所在目录，如果它下面有 dataset/ 或 clip_local_weights_224/
3. 向上最多 3 层，找到第一个含 dataset/ 或 clip_local_weights_224/ 的目录
4. 都没有 → 用脚本所在目录（并在启动时提示）
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Windows 控制台默认 GBK，脚本里的 ⚠️/✅ 会直接抛 UnicodeEncodeError。
# 服务器是 UTF-8 不受影响；这里统一兜底，省得在 Windows 上连 --plan 都跑不了。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SCRIPT_DIR = Path(__file__).resolve().parent

#: "强标志"：只在项目根出现，先只用它们判断基准目录。
#: 不要把 "data" / "results" 这类通用名字放进来：上一级仓库里往往有一个无关的
#: data/ 目录，会把基准目录劫持走（本仓库测试时就踩过这个坑）。
MARKERS_STRONG = ("dataset", "clip_local_weights_224", "colpali_offline_weights",
                  "checkpoints")
#: "弱标志"：只在强标志全找不到时才启用。
#: 注意 models_local/ 同时会出现在权重包内部（ravt_tmm_colqwen2_weights/models_local/），
#: 如果把它当强标志，权重包里的脚本会把权重包自己当成项目根 —— 这是个真踩过的坑。
MARKERS_WEAK = ("models_local",)
MARKERS = MARKERS_STRONG                     # 兼容旧引用

#: 打包后权重可能落在二级目录（例如 ravt_tmm_colqwen2_weights/models_local/），
#: 权重查找按这个顺序探测。
WEIGHT_CONTAINERS = ("models_local", "ravt_tmm_colqwen2_weights/models_local")


def _has_marker(p: Path, markers=MARKERS_STRONG) -> bool:
    """基准目录的标志：有项目级的目录，或者直接放着一个 parquet/arrow 语料。"""
    if any((p / name).exists() for name in markers):
        return True
    return bool(next(p.glob("*.parquet"), None) or next(p.glob("*.arrow"), None))


def find_base_dir(explicit: str | None = None, max_up: int = 3) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.exists():
            raise SystemExit(f"--base-dir 不存在: {p}")
        return p
    # 第一轮：只看强标志（脚本自己 -> 逐级向上）
    if _has_marker(SCRIPT_DIR):
        return SCRIPT_DIR
    p = SCRIPT_DIR
    for _ in range(max_up):
        p = p.parent
        if _has_marker(p):
            return p
    # 第二轮：强标志都没有，再退到弱标志（例如只有一个 models_local/ 的精简项目）
    markers = MARKERS_STRONG + MARKERS_WEAK
    if _has_marker(SCRIPT_DIR, markers):
        return SCRIPT_DIR
    p = SCRIPT_DIR
    for _ in range(max_up):
        p = p.parent
        if _has_marker(p, markers):
            return p
    return SCRIPT_DIR


def find_weight_dir(name: str, base: str | Path | None = None) -> Path:
    """在几个可能的位置里找权重目录（权重包可以解压成二级目录）。

    顺序：<base>/models_local/<name> > <base>/ravt_tmm_colqwen2_weights/models_local/<name>
         > <脚本目录>/models_local/<name> > <base>/<name>
    都不存在时返回第一个候选（调用方据此报错）。
    """
    root = Path(base).resolve() if base else find_base_dir()
    cands = [root / c / name for c in WEIGHT_CONTAINERS]
    cands.append(SCRIPT_DIR / "models_local" / name)
    cands.append(root / name)
    for c in cands:
        if c.exists() and any(c.iterdir()):
            return c
    return cands[0]


def find_corpus(base: Path, corpus_arg: str | None = None) -> str:
    """按优先级找语料：显式参数 > dataset/ 目录 > 基准目录下的 parquet > arrow。

    语料可能是分片目录，也可能是单个 parquet 文件（单文件没有分片可切，
    调用方会用行级切分）。
    """
    if corpus_arg and corpus_arg not in ("dataset", ""):
        return str(Path(corpus_arg).expanduser())
    if (base / "dataset").exists():
        return "dataset"
    for pattern in ("*.parquet", "*.arrow"):
        hits = sorted(base.glob(pattern))
        if hits:
            return str(hits[0]) if len(hits) == 1 else str(base)
    return corpus_arg or "dataset"


def apply_base_dir(base: Path) -> Path:
    """把基准目录同步给所有已加载模块里的 REPO_ROOT。"""
    base = Path(base).resolve()
    for mod in list(sys.modules.values()):
        if mod is not None and hasattr(mod, "REPO_ROOT"):
            try:
                mod.REPO_ROOT = base
            except Exception:
                pass
    return base


def setup(explicit: str | None = None, quiet: bool = False, chdir: bool = True) -> Path:
    """确定基准目录，并把工作目录切过去，保证所有相对路径口径一致。"""
    base = apply_base_dir(find_base_dir(explicit))
    if chdir:
        os.chdir(base)
    if not quiet:
        print(f"基准目录: {base}")
        if base != SCRIPT_DIR:
            print(f"  （脚本在 {SCRIPT_DIR}，向上找到了 dataset/ 或权重目录）")
        missing = [m for m in ("dataset", "clip_local_weights_224") if not (base / m).exists()]
        if missing:
            print(f"  ⚠️ 该目录下没有 {missing}，相关步骤要显式传路径参数"
                  "（例如 --corpus /path/to/dataset）")
    return base
