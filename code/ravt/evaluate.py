"""完整评测：一条命令跑完 所有模型 × 所有测试集，结果写成 JSONL。

评测矩阵
--------
模型（按论文表格顺序）
    RAVT-*        你的 5 个随机种子 + 8 项消融（从 checkpoints/ 自动扫描）
    BM25-word     词级 BM25（论文里的 BM25）
    BM25-char3    字符 3-gram BM25      —— 廉价的抗噪替代方案（审稿人要求）
    BM25-denoise  同形字还原后 BM25     —— 同上
    BGE-M3        稠密文本基线
    CLIP-zero-shot 原生 CLIP 零样本（不训练/不投影/不切行）—— 消融地板
    DSE-Phi3      Document Screenshot Embedding（审稿人点名的 prior work，单向量稠密检索）
    ColPali       多向量 late-interaction 视觉基线（需本地权重）

测试集
    main          主结果：v2_fixed（needle 覆盖 0~4 行、干扰项确定分配）
    hard          跨域：训练没见过的噪声家族（键盘错字/删除/换位 + 图像退化）
    legacy        旧口径（needle 恒在第 0 行）—— 用于和旧论文数字对照

输出
    results/jsonl/eval/<模型>@<测试集>.jsonl   每个 (模型, 任务) 一行
    results/eval_summary.md                    汇总表（带 N 与 95% CI）

断点续跑：已完成的 (run_id, task_id) 自动跳过，中断后原样重跑即可。

用法
----
    # 最常见：主结果 + 跨域，全部模型，端到端延迟
    python evaluate.py --ckpt-dir checkpoints --bench-main data/test_bench

    # 只跑主结果、只跑 RAVT 与 BM25（先看有没有问题）
    python evaluate.py --models RAVT,BM25 --benches main

    # 加上预计算协议的延迟（文档 embedding 只算一次，只统计查询侧）
    python evaluate.py --protocol both
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# 项目根（放 models_local/、data/、checkpoints/ 的那层）。脚本现在解压在子目录里，
# 所以先向上查找一次；切工作目录的完整 setup() 在 main() 里做。
try:
    from ravt_paths import find_base_dir
    PROJECT_ROOT = find_base_dir()
except Exception:                    # ravt_paths 缺失时退回脚本目录（老行为）
    PROJECT_ROOT = HERE


def _writable(path: Path) -> bool:
    """目录是否存在且可写（不存在就尝试创建）。"""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("1", encoding="utf-8")
        probe.unlink()
        return True
    except Exception:
        return False


def setup_hf_cache() -> None:
    """配置 HuggingFace 缓存与**动态模块缓存**。

    自定义模型（DSE 等 trust_remote_code 模型）必须有一个可写的
    transformers_modules 目录，否则会报 PermissionError。

    优先级：包内 hf_cache/（若可写）→ 默认 ~/.cache/huggingface。
    注意：包内目录可能是用 root 解压出来的，对当前用户不可写 —— 所以必须探测，
    不能只看"存在就用"。
    """
    # 优先用项目根的 hf_cache/（权重包可能解压在那里），其次是脚本同级
    bundled = PROJECT_ROOT / "hf_cache"
    if not bundled.exists() and (HERE / "hf_cache").exists():
        bundled = HERE / "hf_cache"
    if bundled.exists():
        if _writable(bundled):
            os.environ.setdefault("HF_HOME", str(bundled))
        else:
            print(f"⚠️ 包内 {bundled} 不可写（权限问题？），改用默认 HF 缓存"
                  f"（可 rm -rf {bundled} 消除此提示）")

    env_cache = os.environ.get("HF_MODULES_CACHE")
    candidates = [Path(env_cache) if env_cache else None,
                  (bundled / "modules") if _writable(bundled) else None,
                  Path.home() / ".cache" / "huggingface" / "modules"]
    for cand in candidates:
        if cand is not None and _writable(cand):
            os.environ["HF_MODULES_CACHE"] = str(cand)
            print(f"动态模块缓存: {cand}")
            return
    print("⚠️ 找不到可写的动态模块缓存目录 —— trust_remote_code 模型可能加载失败")


setup_hf_cache()

try:
    from config import EvalSpec                   # noqa: E402
except ImportError as _exc:                       # pragma: no cover
    raise SystemExit(
        f"❌ 从 config.py 里读不到 EvalSpec（{_exc}）。\n"
        "   最可能的原因：这个目录里先解压了**训练包**，它的 config.py 覆盖了评测包的版本。\n"
        "   解决：重新解压一次评测包，或直接把评测包里的 config.py 拷过来覆盖：\n"
        "       unzip -o ravt_tmm_code/ravt.zip\n"
        "   两个包都是扁平结构，解压顺序无所谓（新版两边发的是同一份 config.py）。\n"
        "   验证：python -c \"import config; print(hasattr(config, 'EvalSpec'))\"  应输出 True"
    ) from None
from corpus import CorpusSpec, split_rows         # noqa: E402
from scorers import ocr_ready                     # noqa: E402
from stage_eval import evaluate_run               # noqa: E402

#: 消融 checkpoint 名 -> 论文里的行名
ABLATION_LABELS = {
    "no_fusion": "w/o Multi-Layer Fusion",
    "no_slicing": "w/o RAVT (global pooling)",
    "no_gated_proj": "w/o Gated Projector",
    "no_buckets": "w/o Length Bucketing",
    "no_noise": "w/o Curriculum Noise",
    "no_local_loss": "w/o Local (row) Loss",
    "no_margin_loss": "w/o Margin Loss",
    "no_local_no_margin": "w/o Local+Margin Loss",
}


# --------------------------------------------------------------------------- #
def discover_checkpoints(ckpt_dir: Path) -> tuple[list[Path], list[Path]]:
    """返回 (主模型 checkpoints, 消融 checkpoints)。"""
    # TMM 版的主模型有两组：full_*（原论文，字符噪声训练）与
    # deg_*/degreal_*（退化感知训练）。两组都进"主模型"一栏。
    full = sorted(list(ckpt_dir.glob("full_*.pth"))
                  + list(ckpt_dir.glob("deg_*.pth"))
                  + list(ckpt_dir.glob("degreal_*.pth")))
    abl = sorted(p for p in ckpt_dir.glob("no_*.pth"))
    return full, abl


def ravt_name(path: Path) -> str:
    """full_p14_s0.pth -> RAVT-s0；no_slicing_p14_s0.pth -> w/o RAVT (global pooling)"""
    stem = path.stem
    if stem.startswith("full_"):
        seed = stem.rsplit("_s", 1)[-1]
        # full_p14_rh42_s0 -> RAVT-rh42-s0（带标签的实验变体，避免与主模型同名）
        mt = re.match(r"full_p\d+_([A-Za-z0-9]+)_s", stem)
        if mt:
            return f"RAVT-{mt.group(1)}-s{seed}"
        return f"RAVT-s{seed}"
    if stem.startswith("degreal_"):
        seed = stem.rsplit("_s", 1)[-1]
        return f"RAVT-DEG-real-s{seed}"
    if stem.startswith("deg"):
        seed = stem.rsplit("_s", 1)[-1]
        tag = stem.split("_p", 1)[0].replace("deg", "").strip("_")
        return f"RAVT-DEG{('- ' + tag) if tag else ''}-s{seed}"
    for key, label in ABLATION_LABELS.items():
        if stem.startswith(key + "_"):
            seed = stem.rsplit("_s", 1)[-1]
            # no_slicing_p14_rh42_s0 -> "w/o RAVT (global pooling)-rh42 (s0)"
            mt = re.match(rf"{key}_p\d+_([A-Za-z0-9]+)_s", stem)
            tag = f"-{mt.group(1)}" if mt else ""
            return f"{label}{tag} (s{seed})"
    return stem


def local_weights_or_repo(models_dir: str, name: str, repo_id: str) -> str:
    """有本地权重就用本地（离线可用），否则回落到 HF 缓存/在线仓库。

    权重可能落在二级目录里（例如 ravt_tmm_colqwen2_weights/models_local/），
    所以按 ravt_paths.find_weight_dir 的顺序探测，而不是只认 models_local/。
    """
    try:
        from ravt_paths import find_weight_dir
        p = find_weight_dir(name)
    except Exception:
        p = Path(models_dir) / name
    if p.exists() and any(p.iterdir()):
        n_files = sum(1 for _ in p.rglob("*") if _.is_file())
        print(f"  {name}: 使用本地权重 {p}（{n_files} 个文件）")
        return str(p)
    print(f"  {name}: 本地目录 {models_dir}/{name} 不存在 -> 改用 {repo_id}"
          "（优先读取 HF 缓存；若离线请确保缓存里已有）")
    return repo_id


def make_specs(args) -> list[EvalSpec]:
    """构造完整评测矩阵。name 里带 @<测试集> 以保证不同测试集的 JSONL 不互相覆盖。"""
    specs: list[EvalSpec] = []
    #: 记录被跳过的基线及原因 —— 必须写进汇总，否则"某条基线没跑"会被完全忽略
    skipped: list[tuple[str, str]] = []
    want = None if args.models.upper() == "ALL" else \
        {m.strip().lower() for m in args.models.split(",") if m.strip()}

    def need(tag: str) -> bool:
        """--models 的选择判断。

        支持子串（大小写不敏感）：用户看着汇总表里的 `BM25-word@main` 写
        `--models BM25-word` 也该生效，而不是得到一个空矩阵。
        """
        if want is None:
            return True
        t = tag.lower()
        return any(t == w or t in w or w in t for w in want)

    ckpt_dir = Path(args.ckpt_dir)
    full, abl = discover_checkpoints(ckpt_dir)
    print(f"checkpoint 扫描: {len(full)} 个主模型 + {len(abl)} 个消融（{ckpt_dir}）")
    if not full and not abl:
        print(f"⚠️ {ckpt_dir} 下没有 full_*.pth / no_*.pth，只评测基线")

    # 最容易犯的错：checkpoint 明明在，但 --models 没选 RAVT，
    # 于是汇总表里没有"我们自己的模型"这一行，却不容易被察觉。
    notes: list[str] = []
    if (full or abl) and not (want is None or "ravt" in want):
        msg = (f"{ckpt_dir} 里有 {len(full)} 个主模型 + {len(abl)} 个消融，"
               f"但 --models={args.models} 没包含 RAVT")
        print(f"⚠️ {msg} —— 汇总表里不会有你们的模型。要跑就加 --models all（或 RAVT）")
        notes.append(msg + "（本次未评测）")

    if need("ravt"):
        for p in full:
            specs.append(EvalSpec(name=ravt_name(p), kind="ravt", checkpoint=str(p), row_height=args.row_height,
                                  patch_size=args.patch_size, aggregation="max_row"))
        for p in abl:
            specs.append(EvalSpec(name=ravt_name(p), kind="ravt", checkpoint=str(p), row_height=args.row_height,
                                  patch_size=args.patch_size, aggregation="max_row"))
        # 行级 late interaction 聚合（E3 的修复尝试）：只对主模型跑，作为对照
        if args.late_interaction:
            for p in full:
                specs.append(EvalSpec(name=f"{ravt_name(p)} +LI", kind="ravt", row_height=args.row_height,
                                      checkpoint=str(p), patch_size=args.patch_size,
                                      aggregation="late_interaction"))

    if need("bm25"):
        for tok, nm in (("word", "BM25-word"), ("char3", "BM25-char3"),
                        ("char5", "BM25-char5"), ("denoise", "BM25-denoise")):
            specs.append(EvalSpec(name=nm, kind="bm25", tokenizer=tok, needs_gpu=False))

    # ---- 真实流水线：先 OCR 再检索（审稿人第 1/4 条要的对照）----
    ocr_tag: str | None = None          # OCR 可用时形如 "[tesseract]"
    common_ocr: dict = {}
    if need("ocr") or need("all"):
        ok, backend = ocr_ready(getattr(args, "ocr_backend", "auto"))
        if not ok:
            print(f"⚠️ 跳过 OCR 基线：\n{backend}")
            skipped.append(("OCR 三件套 (BM25-word+OCR / BM25-char3+OCR / BGE-M3+OCR)", backend))
        else:
            # 把后端名写进模型名：这样 Tesseract 与 PP-OCR 的结果可以并存并直接对比
            tag = f"[{backend}]"
            ocr_tag = tag
            print(f"  OCR 后端: {backend}（模型名将带 {tag} 后缀）")
            common_ocr = {"ocr_backend": backend,
                          "ocr_scale": getattr(args, "ocr_scale", None),
                          "ocr_psm": getattr(args, "ocr_psm", 6)}
            specs += [
                EvalSpec(name=f"BM25-word+OCR{tag}", kind="ocr_text", needs_gpu=False,
                         extra={"base_kind": "bm25", "base_tokenizer": "word", **common_ocr}),
                EvalSpec(name=f"BM25-char3+OCR{tag}", kind="ocr_text", needs_gpu=False,
                         extra={"base_kind": "bm25", "base_tokenizer": "char3", **common_ocr}),
                EvalSpec(name=f"BM25-char5+OCR{tag}", kind="ocr_text", needs_gpu=False,
                         extra={"base_kind": "bm25", "base_tokenizer": "char5", **common_ocr}),
            ]
            mid_ocr = local_weights_or_repo(args.models_dir, "bge-m3", "BAAI/bge-m3")
            specs.append(EvalSpec(name=f"BGE-M3+OCR{tag}", kind="ocr_text", pooling="cls",
                                  batch_size=64, model_id=mid_ocr,
                                  extra={"base_kind": "text_encoder", **common_ocr}))

    # ---- E2 归因实验：用**同一套噪声课程**微调过的文本编码器（BGE-M3）----
    # 训练脚本 train_text_baseline.py 会把权重存成 checkpoints/text_bge_typo_s0/ 这样的目录。
    # 这条基线回答审稿人最尖锐的问题："起作用的到底是渲染成图，还是加噪训练？"
    for d in sorted(p for p in ckpt_dir.glob("text_*") if p.is_dir()):
        specs.append(EvalSpec(name=f"TXT-{d.name}", kind="text_encoder",
                              checkpoint=str(d), model_id=str(d),
                              pooling="cls", batch_size=64))
        if ocr_tag:
            specs.append(EvalSpec(name=f"TXT-{d.name}+OCR{ocr_tag}", kind="ocr_text",
                                  pooling="cls", batch_size=64, model_id=str(d),
                                  extra={"base_kind": "text_encoder",
                                         "base_model_path": str(d), **common_ocr}))

    if need("bge") or need("bge-m3"):
        mid = local_weights_or_repo(args.models_dir, "bge-m3", "BAAI/bge-m3")
        specs.append(EvalSpec(name="BGE-M3", kind="text_encoder",
                              model_id=mid, pooling="cls", batch_size=64))

    # ---- 字符级（tokenizer-free）文本编码器：回应"为什么不用 ByT5/CANINE" ----
    if need("byt5") or need("all"):
        mid = local_weights_or_repo(args.models_dir, "byt5-small", "google/byt5-small")
        specs.append(EvalSpec(name="ByT5-char", kind="text_encoder", model_id=mid,
                              pooling="mean", batch_size=32,
                              extra={"encoder_class": "t5enc"}))

    # 注：Jina-CLIP-v2 已按需求移除（原论文里该基线是坏的，且离线加载不可靠）

    # ---- 原生 CLIP 零样本：不做任何适配，验证"我们的训练/适配带来了多少" ----
    if need("clip") or need("zeroshot") or need("all"):
        if Path(args.clip_path).exists():
            specs.append(EvalSpec(name="CLIP-zero-shot", kind="clip_zeroshot",
                                  batch_size=64,
                                  extra={"clip_path": args.clip_path}))
        else:
            print(f"⚠️ 跳过 CLIP-zero-shot：找不到 backbone {args.clip_path}")
            skipped.append(("CLIP-zero-shot", f"找不到 backbone {args.clip_path}"))

    # ---- DSE：审稿人 3 点名的"最接近的已有工作"，论文必须引用并对比 ----
    if need("dse") or need("all"):
        dse_path = Path(args.models_dir) / "dse-phi3"
        if dse_path.exists() and any(dse_path.rglob("*.bin")):
            specs.append(EvalSpec(name="DSE-Phi3", kind="image_encoder",
                                  model_id=str(dse_path), batch_size=args.dse_batch,
                                  dtype="bfloat16",
                                  extra={"family": "dse", "attn": args.dse_attn}))
        else:
            print(f"⚠️ 跳过 DSE：找不到权重 {dse_path}")
            skipped.append(("DSE-Phi3", f"找不到权重 {dse_path}"))

    if need("colpali"):
        base = Path(args.colpali_base)
        adapter = Path(args.colpali_adapter)
        if base.exists() and adapter.exists():
            specs.append(EvalSpec(name="ColPali", kind="image_encoder",
                                  model_id="vidore/colpali-v1.2", batch_size=4,
                                  dtype="bfloat16",
                                  extra={"base_model_dir": str(base),
                                         "adapter_dir": str(adapter)}))
        else:
            print(f"⚠️ 跳过 ColPali：找不到本地权重\n"
                  f"   期望 {base}\n"
                  f"    与 {adapter}\n"
                  "   （跑过旧实验的服务器上通常已有 colpali_offline_weights/）")
            skipped.append(("ColPali", f"找不到本地权重 {base} / {adapter}"))

    # ---- ColQwen2：ColPali 家族的当前最强形态，TMM 版新增的领域内强势基线 ----
    if need("colqwen") or need("colqwen2") or need("all"):
        try:
            from ravt_paths import find_weight_dir
            base_path = find_weight_dir("colqwen2-base")
            adapter_path = find_weight_dir("colqwen2")
        except Exception:
            base_path = Path(args.colqwen_base)
            adapter_path = Path(args.colqwen_dir)
        has_base = base_path.exists() and any(base_path.glob("*.safetensors"))
        has_adapter = adapter_path.exists() and any(adapter_path.glob("*.safetensors"))
        if has_base and has_adapter:
            specs.append(EvalSpec(name="ColQwen2", kind="image_encoder",
                                  model_id="vidore/colqwen2-v1.0",
                                  batch_size=args.colqwen_batch, dtype="bfloat16",
                                  extra={"family": "colqwen2",
                                         "local_dir": str(base_path),
                                         "adapter_dir": str(adapter_path),
                                         "attn": args.colqwen_attn}))
        else:
            # 注意：vidore/colqwen2-v1.0 只是 LoRA 适配器，必须配 vidore/colqwen2-base 使用
            print(f"⚠️ 跳过 ColQwen2：找不到权重\n"
                  f"   基座 {base_path}（{'有' if has_base else '缺'}）\n"
                  f"   适配器 {adapter_path}（{'有' if has_adapter else '缺'}）")
            skipped.append(("ColQwen2", f"找不到基座 {base_path} / 适配器 {adapter_path}"))
    if not full and (want is None or "ravt" in want):
        skipped.append(("RAVT（5 种子 + 8 消融）", f"{ckpt_dir} 下没有 full_*.pth / no_*.pth"))
    args._skipped = skipped
    args._notes = notes
    only = getattr(args, "only_names", None)
    if only:
        tokens = [t.strip().lower() for t in str(only).split(",") if t.strip()]
        before = len(specs)
        specs = [s for s in specs
                 if any(t in s.name.lower() for t in tokens)]
        print(f"--only-names={only}：{before} -> {len(specs)} 个 run")
        if not specs:
            print("   ⚠️ 筛完为空，检查拼写（可用 --models all 先看运行名）")
    skip = getattr(args, "skip_names", None)
    if skip:
        tokens = [t.strip().lower() for t in str(skip).split(",") if t.strip()]
        before = len(specs)
        dropped = [s.name for s in specs
                   if any(t in s.name.lower() for t in tokens)]
        specs = [s for s in specs
                 if not any(t in s.name.lower() for t in tokens)]
        print(f"--skip-names={skip}：{before} -> {len(specs)} 个 run"
              + (f"（排除 {', '.join(dropped[:6])}{' …' if len(dropped) > 6 else ''}）"
                 if dropped else "（没匹配到任何模型）"))
    return specs


def enforce_missing_policy(args, skipped: list) -> None:
    """按 --on-missing 处理"被跳过的基线"。

    默认 **error**：只要有基线没能进矩阵，立刻报错退出 —— 避免出现
    "以为跑了 OCR、实际静默跳过"这种情况（真的发生过）。
    确实想放过某些基线时，用 --allow-missing OCR,ColPali 逐个豁免。
    """
    if not skipped:
        return
    allow = [x.strip().lower() for x in (args.allow_missing or "").split(",") if x.strip()]
    blocking = [(n, r) for n, r in skipped
                if not any(a in n.lower() for a in allow)]
    if args.on_missing == "warn" or not blocking:
        return
    print("\n" + "!" * 100)
    print(f"❌ 有 {len(blocking)} 组基线无法进入评测矩阵 —— 已按 --on-missing=error 中止：")
    for name, reason in blocking:
        print(f"\n   ❌ {name}")
        for line in str(reason).strip().splitlines():
            print(f"      {line}")
    print("\n   处理方式：")
    print("     · 装好依赖后重跑（推荐）")
    print("     · 确实要跳过它：加 --allow-missing "
          f"{','.join(n.split()[0] for n, _ in blocking)}")
    print("     · 全部跳过不报错：加 --on-missing warn")
    print("!" * 100)
    raise SystemExit(2)


def bench_list(args) -> list[dict]:
    """返回本次要评测的测试集列表。

    两条独立的噪声轴 + 一个叠加：
      main/langacy : seen（只改文本内容，与训练同分布）
      char         : unseen_char（只改文本内容，训练没见过的错字类型）
      image        : unseen_image（**只改图像质量**，文本本身是干净的）
      hard         : unseen_all（两条叠加）

    为什么必须分开：字符替换噪声对 OCR 几乎无损（实测 CER 1.6%），
    它回答的是"谁更容忍错字"；只有成像退化才会把 OCR 打崩（CER 43~50%），
    那才是"文本流水线 vs 视觉流水线"的公平对比。

    image / hard 支持强度梯度：--hard-scales 0.5,1.0,1.5 生成不同强度的测试集。
    """
    out: list[dict] = []
    if "main" in args.benches:
        out.append(dict(tag="main", path=args.bench_main, kind="main",
                        desc="v3_tmm 主测试集（20~1000 块 × 3 噪声档 × 3 深度）"))
    if "legacy" in args.benches:
        out.append(dict(tag="legacy", path=args.bench_legacy, kind="legacy",
                        desc="needle 恒在第 0 行（旧口径）"))
    if "char" in args.benches:
        out.append(dict(tag="char", path=args.bench_char, kind="char",
                        desc="未见过的字符级错误（轴向 1：源文本错误）"))
    if "real" in args.benches:
        out.append(dict(tag="real", path=args.bench_real, kind="real",
                        desc="真实扫描件退化（透视/折痕/透印/摩尔纹/复印/椒盐/光照）"))
    if "grid" in args.benches:
        out.append(dict(tag="grid", path=args.bench_grid, kind="grid",
                        desc="2D 网格：内容错字率 × 成像退化强度（热图）"))
    if "image" in args.benches:
        for scale in [float(x) for x in str(args.hard_scales).split(",") if x.strip()]:
            tag = "image" if scale == 1.0 else f"image_x{scale:g}".replace(".", "")
            path = args.bench_image if scale == 1.0 else f"{args.bench_image}_x{scale:g}".replace(".", "")
            out.append(dict(tag=tag, path=path, kind="image", scale=scale,
                            desc=f"纯成像退化 ×{scale:g}（轴向 2：文本干净，基线必须 OCR）"))
    if "hard" in args.benches:
        for scale in [float(x) for x in str(args.hard_scales).split(",") if x.strip()]:
            tag = "hard" if scale == 1.0 else f"hard_x{scale:g}".replace(".", "")
            path = args.bench_hard if scale == 1.0 else f"{args.bench_hard}_x{scale:g}".replace(".", "")
            out.append(dict(tag=tag, path=path, kind="hard", scale=scale,
                            desc=f"未见噪声家族 + 图像退化 ×{scale:g}"))
    return out


def bench_composition(path: Path) -> dict | None:
    """扫一个测试集目录的 metadata.json，返回它**真实**的网格构成。"""
    blocks, noises, depths, images, n = set(), set(), set(), set(), 0
    per_cell: dict[tuple, int] = {}
    for d in path.glob("Task_*"):
        mp = d / "metadata.json"
        if not mp.exists():
            continue
        try:
            m = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            continue
        n += 1
        blocks.add(m.get("block_count"))
        noises.add(m.get("noise_level"))
        depths.add(m.get("depth"))
        if m.get("image_level") is not None:
            images.add(m.get("image_level"))
        cell = (m.get("block_count"), m.get("noise_level"), m.get("image_level"))
        per_cell[cell] = per_cell.get(cell, 0) + 1
    if not n:
        return None
    clean = lambda s: sorted(x for x in s if x is not None)  # noqa: E731
    return {"n": n, "blocks": clean(blocks), "noises": clean(noises),
            "depths": clean(depths), "images": clean(images),
            "cell_counts": sorted(per_cell.values())}


def format_composition(c: dict) -> str:
    parts = [f"task {c['n']}", f"块数 {c['blocks']}", f"噪声 {c['noises']}",
             f"深度 {c['depths']}"]
    counts = sorted(set(c.get("cell_counts") or []))
    if len(counts) > 1:
        # 每个 (块数×档位) 格子的 task 数不一致 => 这份测试集是半成品
        parts.append(f"⚠️ 网格不齐（每格 {counts[0]}~{counts[-1]} 个，"
                     f"应为 {counts[-1]}）")
    if len(c["images"]) > 1:
        parts.append(f"成像 {c['images']}")
    return "｜".join(parts)


def ensure_bench(args, entry: dict) -> bool:
    """测试集不存在时按需生成（hard / legacy）。main 需要用户先跑 prepare_data。"""
    tag, path, desc = entry["tag"], entry["path"], entry["desc"]
    if Path(path).exists() and any(Path(path).glob("Task_*")):
        n = len(list(Path(path).glob("Task_*")))
        # 不要只信标签/描述：直接扫 metadata.json 把**实际**网格打出来，
        # 免得"路径叫 test_bench、里面却是上一版 297 个 task"这种情况看不出来。
        comp = bench_composition(Path(path))
        print(f"测试集 {tag}: {path}（{n} 个 task）")
        if comp:
            print(f"   实际网格: {format_composition(comp)}")
            counts = sorted(set(comp.get("cell_counts") or []))
            if len(counts) > 1:
                print(f"   ⚠️ 这个测试集是**半成品**（每格 {counts[0]}~{counts[-1]} 个 task）——"
                      "多半是上次构建被中断留下的。评测要重跑：\n"
                      f"      rm -rf {path} && 重跑本命令（会用 --repeats 重新生成完整网格）")
            if max(comp["blocks"]) < 1000:
                print(f"   ℹ️ 最大块数只有 {max(comp['blocks'])}：长上下文曲线（→1000 块）"
                      "没有覆盖。请用 code/ravt/build_benchmark.py 生成 "
                      "v3_tmm 网格（--test-bench data/test_bench_tmm）")
        return True
    if tag == "main":
        print(f"❌ 主测试集不存在: {path}\n"
              "   请先用 build_benchmark.py 生成（--test-tasks 300）")
        return False

    print(f"测试集 {tag} 不存在，按需生成到 {path} ...")
    from stage_data import build_benchmark
    from config import TMM_GRIDS
    kind = entry["kind"]
    try:
        test_spec = split_rows(args.corpus_test or args.corpus, test_every=args.test_every)
        if kind in ("hard", "image", "char", "real", "grid"):
            grid = dict(TMM_GRIDS[kind])
            grid["repeats"] = args.repeats
            if kind == "image" and args.image_levels:
                grid["image_levels"] = list(args.image_levels)
            if args.image_coverage is not None:
                grid["image_coverage"] = args.image_coverage
            corruptions = {"hard": "unseen_all", "image": "unseen_image",
                           "char": "unseen_char", "real": "real_scan",
                           "grid": "seen_image"}[kind]
            build_benchmark("v2_fixed", out_root=path, corpus=test_spec[1],
                            corruptions=corruptions, workers=args.workers,
                            grid_override=grid, overwrite=True,
                            image_scale=entry.get("scale", 1.0),
                            log_path=f"results/{tag}_bench_build.jsonl")
        else:  # legacy
            grid = {"block_counts": [20, 50, 100], "noise_levels": [0.0, 0.2, 0.4],
                    "depths": [0.1, 0.5, 0.9], "repeats": args.repeats}
            build_benchmark("v1_legacy", out_root=path, corpus=test_spec[1],
                            workers=args.workers, grid_override=grid, overwrite=True,
                            log_path=f"results/{tag}_bench_build.jsonl")
    except Exception as exc:
        # 只是这一条轴建不出来（多半是没给 --corpus-test），不该拖死整个评测
        print(f"   ❌ 生成失败，跳过该测试轴: {type(exc).__name__}: {exc}")
        print("      （补充轴需要语料：加 --corpus-test <parquet/arrow 路径>，"
              "或先用 code/ravt/build_benchmark.py --extra-benches 生成）")
        return False
    return True


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h) * 100, min(1.0, c + h) * 100)


def summarize(out_dir: Path, md_path: Path, skipped: list | None = None,
              notes: list | None = None) -> None:
    """读全部 JSONL，输出 模型 × 测试集 × 噪声档 的 Recall@1 + CI + DER。"""
    if skipped:
        print("\n" + "!" * 100)
        print(f"⚠️ 本次有 {len(skipped)} 组基线【未参与评测】—— 汇总表里不会有它们的行：")
        for name, reason in skipped:
            first = str(reason).strip().splitlines()[0]
            print(f"   ❌ {name}")
            print(f"      原因: {first}")
        print("!" * 100)

    if notes:
        print("\n" + "!" * 100)
        for note in notes:
            print(f"ℹ️  {note}")
        print("!" * 100)

    rows = []
    failed_keys: set[tuple[str, int]] = set()   # (run_id, task_id) 曾经失败过
    done_keys: set[tuple[str, int]] = set()     # (run_id, task_id) 已经产出结果
    run_info: dict[str, tuple[str, str]] = {}   # run_id -> (model, bench)
    for f in sorted(out_dir.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("record") == "task_eval":
                rows.append(r)
                run_info.setdefault(r.get("run_id", ""),
                                    (r.get("model", "?"), r.get("bench", "main")))
                done_keys.add((r.get("run_id", ""), r.get("task_id", -1)))
            elif r.get("record") == "failure":
                failed_keys.add((r.get("run_id", ""), r.get("task_id", -1)))

    def failures_of(model: str, bench: str) -> int:
        """仍然**没有产出结果**的 task 数。

        重跑成功后 JSONL 里会同时留着旧的 failure 记录和新的 task_eval 记录，
        所以必须用集合差，避免"明明补跑完了，表里还挂着几十个失败"。
        """
        return sum(1 for (rid, tid) in failed_keys
                   if (rid, tid) not in done_keys and run_info.get(rid) == (model, bench))

    if not rows:
        print("没有评测记录")
        return

    print("\n" + "=" * 100)
    print(f"汇总：{len(rows)} 条记录")
    print("=" * 100)
    lines = ["# 完整评测汇总", "",
             "每个单元格：Recall@1 [95% CI] (N)。DER 为预测落入干扰块的比例。", ""]
    if skipped:
        lines += ["## ⚠️ 本次未参与的基线", ""]
        for name, reason in skipped:
            lines.append(f"- **{name}** —— {str(reason).strip().splitlines()[0]}")
        lines.append("")
    if notes:
        lines += ["## ⚠️ 提醒", ""]
        lines += [f"- {n}" for n in notes]
        lines.append("")

    # 按 (模型, 测试集) 分组
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        bench = r.get("bench", "main")
        groups.setdefault((r["model"], bench), []).append(r)

    by_bench: dict[str, list[tuple[str, list[dict]]]] = {}
    for (model, bench), recs in groups.items():
        by_bench.setdefault(bench, []).append((model, recs))

    for bench, items in sorted(by_bench.items()):
        print(f"\n### 测试集 {bench}")
        lines.append(f"\n## 测试集 {bench}\n")
        noises = sorted({r["noise"] for _, recs in items for r in recs})
        # 只有测试集**真的有**成像轴时才多打一张表（老测试集没有这个字段）
        img_levels = sorted({float(r["image_level"]) for _, recs in items for r in recs
                             if r.get("image_level") is not None})
        # 而且要和噪声轴**真的不同**才算独立一轴：
        # 老的/主测试集里 metadata 的 image_level 往往就是 noise_level，
        # 否则会打出一张与噪声档完全重复的表。
        img_is_own_axis = any(
            abs(float(r["image_level"]) - float(r["noise"])) > 1e-9
            for _, recs in items for r in recs if r.get("image_level") is not None)
        # 打印测试集构成：一眼能看出"这是旧的 297-task 集还是新的 720-task 集"。
        # 注意按 task_id **去重**：items 里是"每个模型一份记录"，
        # 直接数记录会得到 模型数×task 数（曾经显示成 6000 就是这么来的）。
        _blocks = sorted({int(r["blocks"]) for _, recs in items for r in recs})
        per_noise: dict[float, set[int]] = {}
        all_ids: set[int] = set()
        for _, recs in items:
            for r in recs:
                per_noise.setdefault(float(r["noise"]), set()).add(int(r["task_id"]))
                all_ids.add(int(r["task_id"]))
        info = (f"块数 {_blocks}｜task 总数 {len(all_ids)}｜每档 " +
                "/".join(f"{format(nz, '.2f')}:{len(per_noise.get(nz, ()))}" for nz in noises))
        print(f"\n（{info}）")
        lines.append(f"*{info}*\n")

        # ---- 表 A/B：按噪声档（内容轴）与成像档（成像轴）分层 ----
        axes: list[tuple[str, str, list[float]]] = [("noise", "噪声档", noises)]
        if len(img_levels) > 1 and img_is_own_axis:
            axes.append(("image_level", "成像退化档", img_levels))
        for axis_key, axis_name, levels in axes:
            cells_width = 26 if len(levels) <= 4 else 15

            def _fmt(lo, hi, k, n, width=cells_width):
                if width >= 20:
                    return f"{k / n * 100:5.1f}% [{lo:4.1f}, {hi:4.1f}] n={n}".rjust(width)
                return f"{k / n * 100:5.1f}% N={n}".rjust(width)

            header = (f"{'model':<28}" + "".join(f"{axis_name + '=' + format(v, '.2f'):>{cells_width}}"
                                                 for v in levels)
                      + f"{'DER':>12}{'平局':>6}{'退化':>6}{'失败':>6}")
            title = f"按{axis_name}分层"
            print(f"\n-- {title}")
            print(header)
            print("-" * len(header))
            lines += [f"**{title}**", "", "```", header]
            for model, recs in sorted(items, key=lambda x: -sum(r["correct"] for r in x[1]) / len(x[1])):
                cells = []
                for lv in levels:
                    sub = [r for r in recs
                           if r.get(axis_key) is not None
                           and abs(float(r[axis_key]) - lv) < 1e-6]
                    if not sub:
                        cells.append(f"{'-':>{cells_width}}")
                        continue
                    k = sum(r["correct"] for r in sub)
                    lo, hi = wilson(k, len(sub))
                    cells.append(_fmt(lo, hi, k, len(sub)))
                dis = [r for r in recs if r.get("has_distractor")]
                der = (sum(r["hit_distractor"] for r in dis) / len(dis) * 100) if dis else float("nan")
                n_deg = sum(1 for r in recs if r.get("degenerate_input"))
                n_tie = sum(1 for r in recs if r.get("score_all_tied"))
                der_s = f"{der:11.1f}%" if der == der else f"{'-':>12}"
                deg_s = f"{n_deg:>6}" if n_deg else f"{'-':>6}"
                n_fail = failures_of(model, bench)
                fail_s = f"{n_fail:>6}" if n_fail else f"{'-':>6}"
                tie_s = f"{n_tie:>6}" if n_tie else f"{'-':>6}"
                line = f"{model:<28}" + "".join(cells) + der_s + tie_s + deg_s + fail_s
                print(line)
                lines.append(line)
            lines += ["```", ""]

        # ---- 表 C：按块数分层（长上下文曲线，TMM 版新增）----
        blocks = sorted({int(r["blocks"]) for _, recs in items for r in recs})
        if len(blocks) > 1:
            lo_n, hi_n = min(noises), max(noises)
            header = (f"{'model':<28}"
                      + "".join(f"{'B' + str(b):>15}" for b in blocks)
                      + f"{'DER':>8}")
            title = f"按块数分层（噪声 {lo_n:.2f} 档）"
            print(f"\n-- {title}")
            print(header)
            lines += [f"**{title}**", "", "```", header]
            for model, recs in sorted(items, key=lambda x: -sum(r["correct"] for r in x[1]) / len(x[1])):
                cells = []
                for b in blocks:
                    sub = [r for r in recs if int(r["blocks"]) == b
                           and abs(r["noise"] - lo_n) < 1e-6]
                    cells.append(f"{'-':>15}" if not sub
                                 else f"{sum(r['correct'] for r in sub) / len(sub) * 100:5.1f}% N={len(sub)}".rjust(15))
                dis = [r for r in recs if r.get("has_distractor") and int(r["blocks"]) in blocks
                       and abs(r["noise"] - lo_n) < 1e-6]
                der = (sum(r["hit_distractor"] for r in dis) / len(dis) * 100) if dis else float("nan")
                line = (f"{model:<28}" + "".join(cells)
                        + (f"{der:7.1f}%" if der == der else f"{'-':>8}"))
                print(line)
                lines.append(line)
            lines += ["```", "",
                      f"最脏档（noise={hi_n:.2f}）的块数曲线见原始 JSONL："
                      "`results/jsonl/eval/*.jsonl` 里每条记录都带 `blocks` / `noise` / `image_level`。", ""]

        lines += ["", "最后四列：DER = 预测落入干扰块的比例；"
                  "平局 = 所有块得分完全相同（稀疏检索器：查询词全部未命中）的任务数，"
                  "已保守计为未命中；退化 = 文档表示完全相同（基线自身失效）的任务数；"
                  "失败 = 报错未产出的任务数（非 0 说明这条基线还没跑完，"
                  "重跑同一命令会自动补）。", ""]

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n汇总已写入 {md_path}")
    print("把上面这段（或 results/eval_summary.md）发我，我来算显著性并整理成论文表格。")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="完整评测：所有模型 × 所有测试集")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--base-dir", default=None,
                    help="项目根目录（放 models_local/、data/、checkpoints/ 的那层）。"
                         "默认从脚本目录向上自动查找；解压成子目录后不用手动指定")
    ap.add_argument("--models", default="all",
                    help="all 或逗号分隔: RAVT,BM25,OCR,BGE,ByT5,CLIP,ColPali,ColQwen2,DSE")
    ap.add_argument("--only-names", default=None,
                    help="在 --models 选出的集合里再按**模型名**子串筛一层（逗号分隔）。"
                         "例：--models all --only-names RAVT-s0,BM25,ColQwen2 "
                         "只跑这几种，用来在补充轴上控成本")
    ap.add_argument("--skip-names", default=None,
                    help="按模型名子串**排除**（逗号分隔）。例：--models all "
                         "--skip-names DSE,ColPali 跳过最慢的两个 4B 级模型")
    ap.add_argument("--benches", nargs="+", default=["main"],
                    choices=["main", "legacy", "char", "image", "real", "hard", "grid"])
    ap.add_argument("--bench-main", default="data/test_bench")
    ap.add_argument("--bench-legacy", default="data/test_legacy")
    ap.add_argument("--bench-hard", default="data/test_hard")
    ap.add_argument("--bench-real", default="data/test_real",
                    help="真实扫描件退化测试集（透视/折痕/透印/摩尔纹/复印/椒盐/光照）")
    ap.add_argument("--bench-grid", default="data/test_grid",
                    help="2D 热图测试集：内容错字率 × 成像退化强度")
    ap.add_argument("--bench-image", default="data/test_image",
                    help="纯成像退化测试集（文本干净，用来公平对比 OCR 流水线与视觉路线）")
    ap.add_argument("--bench-char", default="data/test_char",
                    help="未见过的字符级错误测试集")
    ap.add_argument("--image-levels", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.4],
                    help="纯成像退化测试集的退化档位（默认 5 档，便于画曲线）")
    ap.add_argument("--hard-scales", default="1.0",
                    help="跨域测试集的图像退化强度倍率，逗号分隔。"
                         "例如 0.5,1.0,1.5 生成三档，用于画退化曲线")
    ap.add_argument("--image-coverage", type=float, default=None,
                    help="成像退化的覆盖率（0..1）。默认 None = 与强度同值（老行为）。"
                         "显式给值可做'覆盖率×强度'解耦实验。")
    ap.add_argument("--models-dir", default="models_local",
                    help="本地基线权重目录（bge-m3/、byt5-small/、dse-phi3/）")
    ap.add_argument("--clip-path", default="clip_local_weights_224",
                    help="CLIP backbone 目录（用于 CLIP-zero-shot 消融）")
    ap.add_argument("--dse-batch", type=int, default=1,
                    help="DSE 的批大小（Phi-3-vision 显存占用大，默认 1）")
    ap.add_argument("--dse-attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"],
                    help="DSE 的注意力实现；没装 flash-attn 就用 sdpa")
    ap.add_argument("--colqwen-dir", default="models_local/colqwen2",
                    help="ColQwen2 的 LoRA 适配器目录（vidore/colqwen2-v1.0）")
    ap.add_argument("--colqwen-base", default="models_local/colqwen2-base",
                    help="ColQwen2 的基座权重目录（vidore/colqwen2-base，约 8.4GB）")
    ap.add_argument("--colqwen-batch", type=int, default=1,
                    help="ColQwen2 的批大小（Qwen2-VL-2B，bf16 约 5GB，默认 1）")
    ap.add_argument("--colqwen-attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"],
                    help="ColQwen2 的注意力实现；加载失败会自动回退 eager")
    ap.add_argument("--ocr-backend", default="auto",
                    choices=["auto", "tesseract", "rapidocr", "paddleocr"],
                    help="OCR 后端。auto = tesseract > rapidocr(PP-OCR 的 ONNX 版) > paddleocr")
    ap.add_argument("--ocr-scale", type=float, default=None,
                    help="OCR 前放大倍数。缺省按后端取最优值："
                         "tesseract 2x（CER≈0.016）、rapidocr 4x（CER≈0.046）")
    ap.add_argument("--ocr-psm", type=int, default=6,
                    help="tesseract 的页面分割模式（6=整块文本，我们的 block 是 5 行文本）")
    ap.add_argument("--on-missing", default="error", choices=["error", "warn"],
                    help="有基线进不了评测矩阵时：error=直接报错退出（默认），warn=只警告")
    ap.add_argument("--allow-missing", default=None,
                    help="豁免的基线名（逗号分隔，子串匹配），例如 ColPali,CLIP")
    ap.add_argument("--colpali-base", default="colpali_offline_weights/base_model")
    ap.add_argument("--colpali-adapter", default="colpali_offline_weights/adapter_model")
    ap.add_argument("--corpus", default=None, help="生成 hard/legacy 测试集用的语料")
    ap.add_argument("--corpus-test", default=None, help="生成 hard/legacy 测试集用的测试语料")
    ap.add_argument("--test-every", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=33, help="生成 hard/legacy 时的重复数")
    ap.add_argument("--workers", type=int, default=8, help="生成测试集时的进程数")
    ap.add_argument("--patch-size", type=int, default=14)
    ap.add_argument("--row-height", type=int,
                    default=int(os.environ.get("RAVT_BLOCK_HEIGHT", 44)),
                    help="行高（像素），必须与训练/渲染一致：44=默认；"
                         "42=与 patch(14) 精确对齐的实验设置。也可用环境变量 "
                         "RAVT_BLOCK_HEIGHT 统一控制")
    ap.add_argument("--protocol", default="end2end", choices=["end2end", "precomputed", "both"])
    ap.add_argument("--out-dir", default="results/jsonl/eval")
    ap.add_argument("--cache-dir", default="results/cache")
    ap.add_argument("--device", default=None)
    ap.add_argument("--late-interaction", action="store_true",
                    help="额外评测行级 late interaction 聚合（E3 的修复尝试）")
    ap.add_argument("--max-tasks", type=int, default=None,
                    help="每个测试集只跑前 N 个任务（冒烟用，例如新基线先跑 3 条）")
    ap.add_argument("--blocks", default=None,
                    help="只跑这些块数（逗号分隔），例如 --blocks 20,50,100。"
                         "补充轴上想快速拿信号时用，可以把 1000 块的慢任务排除掉")
    args = ap.parse_args()
    args._blocks = ([int(x) for x in str(args.blocks).split(",") if x.strip()]
                    if args.blocks else None)

    # 定位项目根：解压到子目录（code/ravt/）时，data/ models_local/ checkpoints/
    # results/ 都在上一层。setup() 会自动向上查找并切换工作目录，
    # 之后所有相对路径的口径就统一了。
    from ravt_paths import setup
    setup(args.base_dir)

    t0 = time.time()
    specs = make_specs(args)
    enforce_missing_policy(args, getattr(args, "_skipped", []))
    if not specs:
        raise SystemExit("评测矩阵为空，检查 --models 参数")

    protocols = ["end2end", "precomputed"] if args.protocol == "both" else [args.protocol]
    benches = bench_list(args)
    print(f"\n评测矩阵: {len(specs)} 个模型 × {len(benches)} 个测试集 × "
          f"{len(protocols)} 种协议")
    for b in benches:
        print(f"  [{b['tag']}] {b['path']}  —— {b['desc']}")

    ok_benches = [b for b in benches if ensure_bench(args, b)]

    for b in ok_benches:
        tag, path = b["tag"], b["path"]
        for proto in protocols:
            for spec in specs:
                if spec.needs_gpu:
                    pass  # 设备选择交给 stage_eval
                # run_id 里带测试集标签与协议，避免不同组合互相覆盖
                s = EvalSpec(**{**spec.to_dict(), "name": f"{spec.name}@{tag}",
                                "extra": {**spec.extra, "bench": tag}})
                try:
                    evaluate_run(s, bench_root=path, protocol=proto,
                                 out_dir=args.out_dir, cache_dir=args.cache_dir,
                                 device=args.device,
                                 block_counts=getattr(args, "_blocks", None),
                                 max_tasks=args.max_tasks)
                except Exception as exc:
                    print(f"  ❌ {s.name} ({proto}) 失败: {type(exc).__name__}: {exc}")

    summarize(Path(args.out_dir), Path("results/eval_summary.md"),
              skipped=getattr(args, "_skipped", None),
              notes=getattr(args, "_notes", None))
    print(f"\n总耗时 {(time.time() - t0) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
