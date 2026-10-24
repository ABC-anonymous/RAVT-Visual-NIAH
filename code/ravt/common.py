"""公共工具：路径解析、JSONL 读写、随机种子、环境快照、计时。"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import platform
import random
import socket
import sys
import time
from pathlib import Path
from typing import Any, Iterable

#: 仓库根目录（code/05_harness/common.py -> 上溯三级）
REPO_ROOT = Path(__file__).resolve().parent  # 扁平包：数据/权重都放在脚本同级目录


def repo_path(*parts: str) -> Path:
    """把相对仓库根目录的路径拼成绝对路径。"""
    return REPO_ROOT.joinpath(*parts)


def ensure_dir(path: Path) -> Path:
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# --------------------------------------------------------------------------- #
# JSONL
# --------------------------------------------------------------------------- #
class JsonlWriter:
    """追加写 JSONL，每写一行立即 flush（长任务中途被杀也不丢数据）。"""

    def __init__(self, path: Path):
        self.path = Path(path)
        ensure_dir(self.path.parent)
        self._fh = open(self.path, "a", encoding="utf-8")

    def write(self, record: dict) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _json_default(obj):
    """让 numpy / torch 标量可序列化。"""
    for attr in ("item", "tolist"):
        if hasattr(obj, attr):
            try:
                return getattr(obj, attr)()
            except Exception:
                pass
    return str(obj)


def read_jsonl(path: Path) -> list[dict]:
    records = []
    if not Path(path).exists():
        return records
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # 忽略被截断的末行
    return records


def iter_jsonl_files(directory: Path, pattern: str = "*.jsonl") -> Iterable[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(directory.rglob(pattern))


def load_done_keys(path: Path, keys: tuple[str, ...]) -> set:
    """读取已有 JSONL，返回已完成记录的 key 集合（用于断点续跑）。"""
    done = set()
    for rec in read_jsonl(path):
        if rec.get("record") == "failure":
            continue  # 失败的任务允许重试
        try:
            done.add(tuple(rec[k] for k in keys))
        except KeyError:
            continue
    return done


def atomic_write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=_json_default)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# 复现性
# --------------------------------------------------------------------------- #
def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def worker_init_fn(seed: int):
    """DataLoader worker 的种子初始化，保证多进程下的可复现性。"""
    # 必须返回**可 pickle** 的对象：Windows 的 DataLoader 用 spawn 启动子进程，
    # 闭包式的局部函数会报
    #   AttributeError: Can't get local object 'worker_init_fn.<locals>._init'
    # （Linux 用 fork 侥幸能跑，所以在服务器上一直没暴露）
    return functools.partial(_seed_worker, seed)


def _seed_worker(seed: int, worker_id: int) -> None:
    seed_everything(seed + worker_id)


def config_hash(cfg: dict, length: int = 8) -> str:
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:length]


def make_run_id(prefix: str, cfg: dict) -> str:
    """人类可读 + 可去重的 run_id，例如 ravt_p14_s0_1a2b3c4d。"""
    return f"{prefix}_{config_hash(cfg)}"


# --------------------------------------------------------------------------- #
# 环境快照
# --------------------------------------------------------------------------- #
def env_snapshot() -> dict:
    snap = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "timestamp": now_iso(),
    }
    try:
        import torch
        snap["torch"] = torch.__version__
        snap["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            snap["cuda_version"] = torch.version.cuda
            snap["gpu_count"] = torch.cuda.device_count()
            snap["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            snap["gpu_memory_gb"] = [
                round(torch.cuda.get_device_properties(i).total_memory / 1024 ** 3, 1)
                for i in range(torch.cuda.device_count())
            ]
    except Exception as exc:
        snap["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import transformers
        snap["transformers"] = transformers.__version__
    except Exception:
        pass
    return snap


# --------------------------------------------------------------------------- #
# 计时
# --------------------------------------------------------------------------- #
class PhaseTimer:
    """分段计时：E4 要求把 渲染 / 编码 / 打分 分开报告。"""

    def __init__(self):
        self.phases: dict[str, float] = {}
        self._t0: dict[str, float] = {}
        self._wall0 = time.perf_counter()

    def start(self, phase: str) -> None:
        self._t0[phase] = time.perf_counter()

    def stop(self, phase: str) -> float:
        if phase not in self._t0:
            return 0.0
        dt = (time.perf_counter() - self._t0.pop(phase)) * 1000.0
        self.phases[phase] = self.phases.get(phase, 0.0) + dt
        return dt

    def total_ms(self) -> float:
        return (time.perf_counter() - self._wall0) * 1000.0

    def result(self) -> dict:
        out = {k: round(v, 3) for k, v in self.phases.items()}
        out["total"] = round(self.total_ms(), 3)
        return out
