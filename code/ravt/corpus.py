"""语料读取层（重设计后的**唯一**文本入口）。

支持的输入形态
--------------
1. HuggingFace `datasets.save_to_disk()` 输出的目录（本次的主力格式）
        dataset/
            dataset_info.json
            state.json
            data-00000-of-00010.arrow      <- 分片
            data-00001-of-00010.arrow
            ...
2. 单个 `.arrow` 文件，或 glob（`dataset/*.arrow`）
3. 兼容旧的 jsonl（每行 `{"text": ...}`）

设计要点
--------
* **流式**：按分片顺序逐批读取，不会把整份语料读进内存。
* **确定性**：分片按文件名排序，采样池对任何机器都一致，
  这是"评测集可复现"的前提（旧 benchmark 就是没固定种子才无法重建）。
* **自动识别文本列**：`text` / `content` / `document` / ... 或第一个足够长的字符串列。
* **分片缺失报警**：文件名里的 `of-00010` 与实际找到的数量不符时明确警告，
  避免"只用了 1/10 的语料"这种静默错误。
* **缓存**：arrow -> jsonl 的一次性转换，后续反复调用（生成训练块 + 生成评测集）
  不必重复解析 arrow。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parent  # 扁平包：数据/权重都放在脚本同级目录

#: 优先尝试的文本列名
_TEXT_COLUMN_CANDIDATES = [
    "text", "content", "document", "passage", "body", "raw", "sentence",
    "plain_text", "contents", "docs", "doc", "chunk",
]

#: 匹配各种分片命名：data-00000-of-00010.arrow、train-00000-of-00002.parquet、
#: validation-00000-of-00001.parquet 等
_SHARD_RE = re.compile(r"[\w.-]+?-(\d+)-of-(\d+)\.(?:arrow|parquet)$")


# --------------------------------------------------------------------------- #
@dataclass
class CorpusSpec:
    """一份语料的描述。"""

    path: str = "dataset"                 # 目录 / 文件 / glob
    text_column: str | None = None        # None = 自动识别
    min_len: int = 15
    max_docs: int | None = None           # 最多读取多少条（None = 全部）
    shard_limit: int | None = None        # 只读前 N 个分片（调试用）
    #: 只使用这些分片编号（如 [0] 训练 / [1] 测试）。
    #: 用分片做 train/test 切分的好处是两边天然不相交，且不需要复制数据。
    shard_ids: list[int] | None = None
    #: 单文件（如 parquet）没有分片可切时，改用**行级切分**：
    #:   split="train" → 不是测试行的那部分
    #:   split="test"  → 每隔 test_every 行取 1 行
    #:   split=None    → 全部（不切分）
    #: 交错取样（而不是切前/后段）是为了不依赖文件内部的行顺序，
    #: 避免"文件恰好按主题排序"导致的分布偏移。
    split: str | None = None
    test_every: int = 10
    cache_dir: str = "data/processed/corpus_cache"
    use_cache: bool = True
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d.pop("extra", None)
        return d


def _abs(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


# --------------------------------------------------------------------------- #
# 文件发现
# --------------------------------------------------------------------------- #
def resolve_files(path: str, shard_ids: list[int] | None = None) -> tuple[list[Path], str]:
    """返回 (排序后的文件列表, 类型)。类型 ∈ {arrow, parquet, jsonl}。"""
    p = _abs(path)
    files: list[Path]
    kind: str
    SUFFIX_KIND = {".arrow": "arrow", ".parquet": "parquet", ".jsonl": "jsonl"}
    if p.is_dir():
        found = None
        for suffix, k in SUFFIX_KIND.items():          # arrow > parquet > jsonl
            hits = sorted(p.rglob(f"*{suffix}"))
            if hits:
                found, kind = hits, k
                break
        if found is None:
            raise FileNotFoundError(f"{p} 下没有 .arrow / .parquet / .jsonl")
        files = found
    elif p.is_file():
        files = [p]
        kind = SUFFIX_KIND.get(p.suffix.lower(), "jsonl")
    else:
        import glob as _glob
        hits = sorted(Path(h) for h in _glob.glob(str(p)))
        if not hits:
            raise FileNotFoundError(f"路径不存在且 glob 无匹配: {path}")
        files = hits
        kind = SUFFIX_KIND.get(hits[0].suffix.lower(), "jsonl")

    if shard_ids:
        selected = []
        for f in files:
            m = _SHARD_RE.search(f.name)
            sid = int(m.group(1)) if m else 0
            if sid in shard_ids:
                selected.append(f)
        if not selected:
            raise FileNotFoundError(
                f"在 {path} 里找不到分片 {shard_ids}。已知分片: "
                f"{[int(_SHARD_RE.search(f.name).group(1)) for f in files if _SHARD_RE.search(f.name)]}")
        files = selected
    return files, kind


def split_shards(path: str, train: list[int] = (0,), test: list[int] = (1,),
                 **kw) -> tuple[CorpusSpec, CorpusSpec]:
    """按分片切分语料（默认：分片 0 训练 / 分片 1 测试）。

    分片切分比"随机划分"更安全：两边天然不相交，不需要复制数据，
    而且论文里可以直接写"训练/测试来自互不重叠的数据分片"。
    """
    train_spec = CorpusSpec(path=path, shard_ids=list(train), **kw)
    test_spec = CorpusSpec(path=path, shard_ids=list(test), **kw)
    return train_spec, test_spec


def split_rows(path: str, test_every: int = 10, **kw) -> tuple[CorpusSpec, CorpusSpec]:
    """单文件（parquet/arrow/jsonl）的行级切分：每 test_every 行取 1 行作测试。

    交错取样而不是切前/后段，是为了不依赖文件内部的行顺序——避免文件恰好按主题
    排序时把某一类内容整段划进测试集。
    """
    train_spec = CorpusSpec(path=path, split="train", test_every=test_every, **kw)
    test_spec = CorpusSpec(path=path, split="test", test_every=test_every, **kw)
    return train_spec, test_spec


def decide_split_mode(path: str) -> str:
    """auto 模式：有 >=2 个不同分片编号就用分片切分，否则用行级切分。"""
    try:
        files, _ = resolve_files(path)
    except Exception:
        return "rows"
    ids = {int(m.group(1)) for f in files if (m := _SHARD_RE.search(f.name))}
    return "shards" if len(ids) >= 2 else "rows"


def check_shards(files: list[Path]) -> dict:
    """检查分片完整性：文件名里的 `of-N` 与实际找到的数量是否一致。"""
    nums, totals = [], set()
    for f in files:
        m = _SHARD_RE.search(f.name)
        if m:
            nums.append(int(m.group(1)))
            totals.add(int(m.group(2)))
    info = {"n_found": len(files), "shard_ids": sorted(nums),
            "declared_total": sorted(totals)}
    if totals:
        expected = max(totals)
        info["complete"] = (len(set(nums)) == expected)
        if not info["complete"]:
            found = set(nums)
            info["missing_shards"] = [i for i in range(expected) if i not in found][:20]
    else:
        info["complete"] = None
    return info


# --------------------------------------------------------------------------- #
# 读取
# --------------------------------------------------------------------------- #
def _batch_iter(path: Path, batch_size: int = 4096) -> Iterator[object]:
    """逐批产出 RecordBatch。兼容 parquet / arrow file / arrow stream / datasets 四种写法。"""
    import pyarrow as pa

    # 0) parquet（HF datasets 导出、pandas.to_parquet 都走这条）
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(str(path))
        for batch in pf.iter_batches(batch_size=batch_size):
            yield batch
        return

    # 1) 标准 Arrow file
    try:
        src = pa.memory_map(str(path), "rb")
        reader = pa.ipc.open_file(src)
        for i in range(reader.num_record_batches):
            yield reader.get_batch(i)
        return
    except Exception:
        pass

    # 2) Arrow stream（HuggingFace datasets 用的就是这种，带自定义 footer）
    try:
        src = pa.memory_map(str(path), "rb")
        reader = pa.ipc.open_stream(src)
        while True:
            try:
                yield reader.read_next_batch()
            except StopIteration:
                return
    except Exception:
        pass

    # 3) 兜底：交给 datasets 解析（会物化，但最兼容）
    from datasets import Dataset
    ds = Dataset.from_file(str(path))
    for batch in ds.iter(batch_size=batch_size):
        yield pa.record_batch(batch)


def read_schema(path: Path) -> tuple[list[str], dict]:
    """读取第一个 batch 的名称与类型，用于自动识别文本列。"""
    for batch in _batch_iter(path):
        types = {n: str(t) for n, t in zip(batch.schema.names, batch.schema.types)}
        return batch.schema.names, types
    return [], {}


def detect_text_column(path: Path, explicit: str | None = None) -> str:
    names, types = read_schema(path)
    if not names:
        raise RuntimeError(f"{path} 里没有数据")
    if explicit:
        if explicit not in names:
            raise KeyError(f"指定的文本列 {explicit!r} 不存在，实际列: {names}")
        return explicit

    for cand in _TEXT_COLUMN_CANDIDATES:
        if cand in names:
            return cand
    # 退而求其次：找字符串列，看第一条样本的长度
    # （parquet 里的图像/向量列是 binary/struct/list，必须排除掉）
    str_cols = [n for n in names
                if types.get(n, "").lower() in ("string", "large_string", "utf8",
                                                "large_utf8", "string_view")
                or types.get(n, "").lower().startswith("dictionary")]
    for batch in _batch_iter(path):
        for col in str_cols:
            try:
                v = batch.column(col)[0].as_py()
            except Exception:
                continue
            if isinstance(v, str) and len(v) >= 20:
                return col
        break
    raise KeyError(
        f"无法自动识别文本列。{path} 的列为 {names}（类型 {types}）。"
        f"请用 --corpus-text-column 显式指定。"
    )


def _extract(value) -> str:
    """把单元格值统一成字符串。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):                      # 嵌套结构里找 text
        for k in ("text", "content", "raw", "value"):
            if k in value and isinstance(value[k], str):
                return value[k]
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return " ".join(_extract(v) for v in value)
    return str(value)                                # 数字等


def iter_texts(spec: CorpusSpec) -> Iterator[str]:
    """按分片顺序流式产出满足 min_len 的文本（可选行级 train/test 切分）。"""
    files, kind = resolve_files(spec.path, spec.shard_ids)
    if spec.shard_limit:
        files = files[:spec.shard_limit]

    def _keep(idx: int) -> bool:
        """行级切分：split=None 全要；test 取每 test_every 行 1 行；train 取其余。"""
        if not spec.split:
            return True
        is_test = (idx % max(2, spec.test_every) == 0)
        return is_test if spec.split == "test" else (not is_test)

    emitted = 0
    seen = 0            # 已过滤（min_len 通过）的全局行号，用于行级切分
    if kind == "jsonl":
        for f in files:
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = _extract(rec.get("text") if isinstance(rec, dict) else rec).strip()
                    if len(text) < spec.min_len:
                        continue
                    if not _keep(seen):
                        seen += 1
                        continue
                    seen += 1
                    yield text
                    emitted += 1
                    if spec.max_docs and emitted >= spec.max_docs:
                        return
        return

    first = files[0]
    column = detect_text_column(first, spec.text_column)
    for f in files:
        # 每个分片单独确认列名（分片之间列名理论上一致，但便宜起见只在需要时报错）
        try:
            col = detect_text_column(f, spec.text_column) if f != first else column
        except Exception:
            col = column
        for batch in _batch_iter(f):
            if col not in batch.schema.names:
                continue
            values = batch.column(col)
            for i in range(batch.num_rows):
                text = _extract(values[i].as_py()).strip()
                if len(text) < spec.min_len:
                    continue
                if not _keep(seen):
                    seen += 1
                    continue
                seen += 1
                yield text
                emitted += 1
                if spec.max_docs and emitted >= spec.max_docs:
                    return


# --------------------------------------------------------------------------- #
# 缓存 + 对外接口
# --------------------------------------------------------------------------- #
def _cache_key(spec: CorpusSpec, files: list[Path]) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(spec.to_dict(), sort_keys=True).encode())
    for f in files:
        st = f.stat()
        h.update(f"{f.name}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


def cache_path(spec: CorpusSpec) -> Path | None:
    if not spec.use_cache or not spec.cache_dir:
        return None
    files, _ = resolve_files(spec.path, spec.shard_ids)
    return _abs(spec.cache_dir) / f"corpus_{_cache_key(spec, files)}.jsonl"


def build_cache(spec: CorpusSpec) -> Path:
    """把语料转成 jsonl 缓存（一次性），后续读取走缓存更快。"""
    out = cache_path(spec)
    if out is None:
        raise RuntimeError("未启用缓存")
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8") as fh:
        for text in iter_texts(spec):
            fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            n += 1
    print(f"  语料缓存已写入 {out}（{n} 条）")
    return out


def load_texts(spec: CorpusSpec, limit: int | None = None) -> list[str]:
    """把语料读成内存列表（可选提前截断），需要反复随机采样时用这个。"""
    if limit and spec.max_docs and limit > spec.max_docs:
        limit = spec.max_docs
    cp = cache_path(spec)
    if cp is not None and cp.exists():
        texts = []
        with open(cp, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                texts.append(json.loads(line)["text"])
                if limit and len(texts) >= limit:
                    break
        if texts:
            return texts
    texts = []
    for t in iter_texts(spec):
        texts.append(t)
        if limit and len(texts) >= limit:
            break
    return texts


def describe(spec: CorpusSpec, peek: int = 3) -> dict:
    """给用户/日志看的语料体检报告。"""
    files, kind = resolve_files(spec.path, spec.shard_ids)
    info = {"path": spec.path, "kind": kind, "n_files": len(files),
            "files": [f.name for f in files[:12]],
            "shard_ids": spec.shard_ids,
            "split": spec.split, "test_every": spec.test_every,
            "shards": check_shards(files)}
    if kind == "arrow":
        names, types = read_schema(files[0])
        info["columns"] = names
        info["column_types"] = types
        try:
            info["text_column"] = detect_text_column(files[0], spec.text_column)
        except Exception as exc:
            info["text_column"] = None
            info["text_column_error"] = str(exc)
    if kind == "arrow" and not info.get("text_column"):
        info["samples"] = []          # 没有文本列就不用采样了
        return info
    samples = []
    for i, t in enumerate(iter_texts(spec)):
        samples.append(t[:160])
        if i + 1 >= peek:
            break
    info["samples"] = samples
    return info


def print_report(spec: CorpusSpec) -> dict:
    info = describe(spec)
    print(f"语料: {info['path']}  (类型 {info['kind']}, {info['n_files']} 个文件)")
    if info.get("split"):
        ratio = 1 / max(2, info["test_every"])
        print(f"  行级切分: {info['split']}（每 {info['test_every']} 行取 1 行作测试，"
              f"测试占比 {ratio:.1%}）")
    sh = info["shards"]
    if info.get("shard_ids"):
        # 刻意只取部分分片（train/test 切分），不算"分片缺失"
        print(f"  分片选择: {info['shard_ids']}"
              f"（该数据集共 {sh.get('declared_total') or '?'} 个分片，这是有意的切分）")
    elif sh.get("declared_total"):
        flag = "✅" if sh.get("complete") else "⚠️"
        print(f"  {flag} 分片: 找到 {sh['n_found']}，声明总数 {sh['declared_total']}")
        if sh.get("missing_shards"):
            print(f"     缺失分片(前 20): {sh['missing_shards']}")
            print("     缺失部分不会进入采样池 —— 请确认这是预期行为。")
    if info.get("columns"):
        print(f"  列: {info['columns']}")
        print(f"  文本列: {info.get('text_column')}"
              + (f"  ({info.get('text_column_error')})" if info.get("text_column_error") else ""))
    for i, s in enumerate(info["samples"], 1):
        print(f"  样本{i}: {s}")
    return info
