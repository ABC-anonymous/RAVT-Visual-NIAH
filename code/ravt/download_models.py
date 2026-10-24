"""检查/下载评测所需的基线权重。

如果服务器能联网，直接跑这个即可，不必把权重打进包里：
    python download_models.py                       # 下 BGE-M3 / ByT5-small / DSE-Phi3
    python download_models.py --models bge-m3

当前评测矩阵需要的权重：
    bge-m3        BAAI/bge-m3                     稠密文本基线
    byt5-small    google/byt5-small               字符级（tokenizer-free）文本基线
    dse-phi3      Tevatron/dse-phi3-docmatix-v1   DSE（文档截图稠密检索，审稿人点名的 prior work）

已移除：Jina-CLIP-v2（原论文里该基线是坏的，离线加载也不可靠）。

如果服务器不能联网：
1) 先确认缓存里是否已有（跑过旧实验的话大概率有）：
       python download_models.py --check
2) 没有的话，在有网的机器上下好 models_local/，整个目录拷到服务器同目录即可。

ColPali（PaliGemma-3B 约 11GB）不在默认列表里，它通常已经在
colpali_offline_weights/ 下。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

#: 评测矩阵实际需要的权重（与 evaluate.py 一致）
#: 注意：Jina-CLIP-v2 已从评测矩阵移除，不再检查它。
MODELS = {
    "bge-m3": "BAAI/bge-m3",
    "byt5-small": "google/byt5-small",
    "dse-phi3": "Tevatron/dse-phi3-docmatix-v1",
    # TMM 版新增的领域内强势基线：ColQwen2 = 基座（全量）+ LoRA 适配器
    "colqwen2-base": "vidore/colqwen2-base",
    "colqwen2": "vidore/colqwen2-v1.0",
}


WEIGHT_SUFFIX = (".safetensors", ".bin")


def _has_weights(root: Path) -> tuple[bool, int]:
    """目录里是否有权重文件；返回 (有, 文件数)。"""
    if not root.exists():
        return False, 0
    files = [f for f in root.rglob("*") if f.is_file()]
    has = any(f.suffix in WEIGHT_SUFFIX and f.stat().st_size > 1024 for f in files)
    return has, len(files)


def candidates(name: str, repo: str) -> list[tuple[str, Path]]:
    """评测代码实际会按这个顺序找权重（与 scorers.local_weights_or_repo 一致）。"""
    slug = "models--" + repo.replace("/", "--")
    try:
        from ravt_paths import find_base_dir, SCRIPT_DIR
        base, here = find_base_dir(), SCRIPT_DIR
    except Exception:
        base = here = Path(__file__).resolve().parent
    out = [
        ("项目 models_local/", base / "models_local" / name),
        ("权重包二级目录", base / "ravt_tmm_colqwen2_weights" / "models_local" / name),
        ("脚本同级 models_local/", here / "models_local" / name),
        ("包内 hf_cache/", here / "hf_cache" / "hub" / slug),
    ]
    env = os.environ.get("HF_HOME")
    if env:
        p = Path(env)
        out.append(("HF_HOME 缓存", (p / "hub" if p.name != "hub" else p) / slug))
    out.append(("默认 HF 缓存", Path.home() / ".cache" / "huggingface" / "hub" / slug))
    return out


def check() -> int:
    print(f"工作目录: {Path.cwd()}")
    print("按评测代码实际查找顺序检查权重：\n")
    missing = []
    for name, repo in MODELS.items():
        found = None
        print(f"— {name}  ({repo})")
        for label, path in candidates(name, repo):
            has, n = _has_weights(path)
            mark = "✅" if has else "  "
            suffix = f"（{n} 个文件）" if path.exists() else "（不存在）"
            print(f"    {mark} {label:<18} {path} {suffix}")
            if has and found is None:
                found = (label, path)
        if found:
            print(f"    → 就绪，将使用：{found[0]} {found[1]}")
        else:
            print("    → ❌ 缺失")
            missing.append(name)
        print()

    if missing:
        print(f"缺失: {missing}")
        print("· 服务器能联网：python download_models.py")
        print("· 不能联网：把包里的 models_local/ 与 hf_cache/ 两个目录整个拷到本目录")
        return 1
    print("✅ 权重全部就绪，可以离线跑评测")
    return 0


def verify() -> int:
    """真正把每个模型加载一次（比只看文件更可靠）。"""
    import scorers
    from config import EvalSpec
    try:
        from ravt_paths import find_weight_dir
        paths = {n: str(find_weight_dir(n)) for n in ("bge-m3", "byt5-small", "dse-phi3")}
    except Exception:
        here = Path(__file__).resolve().parent
        paths = {n: str(here / "models_local" / n)
                 for n in ("bge-m3", "byt5-small", "dse-phi3")}
    specs = [
        EvalSpec(name="BGE-M3", kind="text_encoder",
                 model_id=paths["bge-m3"],
                 pooling="cls", batch_size=8),
        EvalSpec(name="ByT5-char", kind="text_encoder",
                 model_id=paths["byt5-small"],
                 pooling="mean", batch_size=8, extra={"encoder_class": "t5enc"}),
        EvalSpec(name="DSE-Phi3", kind="image_encoder",
                 model_id=paths["dse-phi3"],
                 batch_size=1, dtype="bfloat16",
                 extra={"family": "dse", "attn": "sdpa"}),
    ]
    ok = 0
    for spec in specs:
        try:
            s = scorers.build_scorer(spec, device=None)   # 自动选 cuda（DSE 在 CPU 上装不下）
            s.prepare()
            print(f"  ✅ {spec.name} 加载成功")
            ok += 1
        except Exception as exc:
            print(f"  ❌ {spec.name} 加载失败: {type(exc).__name__}: {str(exc)[:160]}")
    print(f"\n{ok}/{len(specs)} 个基线可加载")
    print("（加载失败的会被评测自动跳过，不影响其他模型）")
    return 0 if ok == len(specs) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models_local")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="真正加载一次每个基线（更可靠，需 1~2 分钟）")
    ap.add_argument("--base-dir", default=None,
                    help="项目根目录；默认从脚本目录向上自动查找")
    args = ap.parse_args()

    try:
        from ravt_paths import setup
        setup(args.base_dir, quiet=True)
    except Exception:
        pass

    if args.check:
        raise SystemExit(check())
    if args.verify:
        raise SystemExit(verify())

    from huggingface_hub import snapshot_download
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    out_root = Path(args.out)
    for name in args.models:
        target = out_root / name
        print(f"[下载] {MODELS[name]} -> {target}")
        snapshot_download(repo_id=MODELS[name], local_dir=str(target),
                          ignore_patterns=["*.h5", "*.msgpack", "*.onnx", "*.tflite", "*.ot"],
                          max_workers=4, resume_download=True)
        size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
        print(f"[完成] {name}  {size / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
