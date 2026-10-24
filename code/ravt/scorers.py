"""检索器注册表：把每个 baseline 抽象成"编码文档 + 打分查询"两步。

统一的接口设计让 E4 的延迟协议可以直接实现：

    end2end      : encode_documents + score_query 都算进延迟（论文 Table 5 的口径）
    precomputed  : encode_documents 只做一次并缓存，延迟只统计 score_query
                   （这才是 dual-encoder 的正确用法，应当是毫秒级）
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from model_defs import HOMOGLYPH_INVERSE_FOR_DENOISE, block_scores


# --------------------------------------------------------------------------- #
# 基类
# --------------------------------------------------------------------------- #
class Scorer:
    kind = "base"
    needs_gpu = True
    uses_images = True
    supports_precompute = True
    #: 所有块得分完全相同时是否算"正常结果"。
    #: 稀疏检索器（BM25）在"查询词一个都没命中"时会全 0 分 —— 这是它的**真实结果**
    #: （应记为未命中），不是配置错误；神经检索器出现全等分数则一定有问题，要报错。
    allows_score_ties = False

    def __init__(self, name: str, cfg: dict | None = None):
        self.name = name
        self.cfg = cfg or {}
        self._index_bytes_per_block = None

    def prepare(self) -> None:
        """加载模型/权重。"""

    def encode_documents(self, task) -> Any:
        raise NotImplementedError

    def score_query(self, query: str, docs: Any, task=None) -> tuple[np.ndarray, dict]:
        """返回 (每个块的得分, 附加信息)。"""
        raise NotImplementedError

    def close(self) -> None:
        pass

    # ---- 可选 ----
    def index_bytes(self, docs: Any) -> int:
        """文档端索引占用的字节数（E11 的存储开销量化）。"""
        if isinstance(docs, np.ndarray):
            return int(docs.nbytes)
        if isinstance(docs, (list, tuple)):
            return int(sum(d.nbytes if isinstance(d, np.ndarray) else 0 for d in docs))
        return 0


# --------------------------------------------------------------------------- #
# BM25 家族（CPU，无需模型）
# --------------------------------------------------------------------------- #
_GIBBERISH = set("#*@%&?!~+=")


def _denoise(text: str) -> str:
    # 先还原同形字，再丢弃乱码符号。
    # 注意顺序：'@' 既是 'a' 的同形字、也在乱码字符表里，必须先还原再考虑丢弃。
    out = []
    for ch in text:
        if ch in HOMOGLYPH_INVERSE_FOR_DENOISE:
            out.append(HOMOGLYPH_INVERSE_FOR_DENOISE[ch])
            continue
        if ch in _GIBBERISH:
            continue
        out.append(ch)
    return "".join(out)


def tokenize(text: str, kind: str) -> list[str]:
    if kind == "word":
        return text.lower().split()
    if kind == "denoise":
        return _denoise(text).lower().split()
    m = re.fullmatch(r"char(\d+)", kind)
    if m:
        n = int(m.group(1))
        s = " ".join(text.lower().split())
        if len(s) < n:
            return [s or "<empty>"]
        return [s[i:i + n] for i in range(len(s) - n + 1)]
    raise ValueError(f"未知的 tokenizer 类型: {kind}")


class Bm25Scorer(Scorer):
    allows_score_ties = True

    kind = "bm25"
    needs_gpu = False
    uses_images = False

    def __init__(self, name: str, tokenizer: str = "word"):
        super().__init__(name, {"tokenizer": tokenizer})
        self.tokenizer = tokenizer

    def encode_texts(self, texts: list[str]):
        """接受显式文本（供 OCR 变体复用）。"""
        return [tokenize(t, self.tokenizer) for t in texts]

    def encode_documents(self, task):
        return self.encode_texts(task.load_texts())

    def score_query(self, query: str, docs, task=None):
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi(docs)
        scores = np.asarray(bm25.get_scores(tokenize(query, self.tokenizer)), dtype=np.float32)
        return scores, {}


# --------------------------------------------------------------------------- #
# 文本双塔（BGE-M3 / ByT5 / CANINE / 微调后的版本）
# --------------------------------------------------------------------------- #
class TextEncoderScorer(Scorer):
    kind = "text_encoder"
    uses_images = False

    def __init__(self, name: str, model_id: str, checkpoint: str | None = None,
                 pooling: str = "cls", batch_size: int = 64, max_length: int = 512,
                 device: str | None = None, encoder_class: str = "auto"):
        super().__init__(name, {"model_id": model_id, "checkpoint": checkpoint,
                                "pooling": pooling, "max_length": max_length,
                                "encoder_class": encoder_class})
        self.model_id = checkpoint or model_id
        self.pooling = pooling
        self.batch_size = batch_size
        self.max_length = max_length
        self.encoder_class = encoder_class
        self.device = device
        self._model = None
        self._tokenizer = None

    def prepare(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.encoder_class == "t5enc":
            # ByT5 这类 encoder-decoder：只用 encoder，得到逐字符表示
            from transformers import T5EncoderModel
            self._model = T5EncoderModel.from_pretrained(self.model_id).to(self.device).eval()
        else:
            self._model = AutoModel.from_pretrained(self.model_id).to(self.device).eval()

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """接受显式文本（供 OCR 变体复用）。"""
        return self._encode(texts)

    def _encode(self, texts: list[str]) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        vecs = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                enc = self._tokenizer(batch, padding=True, truncation=True,
                                      max_length=self.max_length, return_tensors="pt").to(self.device)
                out = self._model(**enc)
                if self.pooling == "mean":
                    mask = enc["attention_mask"].unsqueeze(-1).float()
                    feat = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
                else:
                    feat = out.last_hidden_state[:, 0] if hasattr(out, "last_hidden_state") else out.pooler_output
                vecs.append(F.normalize(feat, dim=1).cpu().numpy())
        return np.concatenate(vecs, axis=0).astype(np.float32)

    def encode_documents(self, task):
        return self.encode_texts(task.load_texts())

    def score_query(self, query: str, docs, task=None):
        q = self._encode([query])[0]
        return (docs @ q).astype(np.float32), {}


# --------------------------------------------------------------------------- #
# RAVT（我们的方法）
# --------------------------------------------------------------------------- #
class RavtScorer(Scorer):
    kind = "ravt"
    uses_images = True

    def __init__(self, name: str, clip_path: str, checkpoint: str | None = None,
                 patch_size: int = 14, n_rows: int = 5, row_height: int = 44,
                 aggregation: str = "max_row", batch_size: int = 64,
                 device: str | None = None, use_slicing: bool = True,
                 use_fusion: bool = True, use_gated_proj: bool = True):
        super().__init__(name, {"clip_path": clip_path, "checkpoint": checkpoint,
                                "patch_size": patch_size, "aggregation": aggregation})
        from model_defs import RAVTConfig
        self.model_cfg = RAVTConfig(
            clip_path=clip_path, patch_size=patch_size, n_rows=n_rows,
            row_height=row_height, use_slicing=use_slicing,
            use_fusion=use_fusion, use_gated_proj=use_gated_proj,
        )
        self.checkpoint = checkpoint
        self.aggregation = aggregation
        self.batch_size = batch_size
        self.device = device
        self._model = None
        self._processor = None
        self._tokenizer = None

    def prepare(self):
        import torch
        from transformers import CLIPProcessor, CLIPTokenizer
        from model_defs import RAVTModel

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = RAVTModel(self.model_cfg).to(self.device).eval()
        if self.checkpoint:
            state = torch.load(self.checkpoint, map_location=self.device)
            state = {k.replace("module.", ""): v for k, v in state.items()}
            missing, unexpected = self._model.load_state_dict(state, strict=False)
            self.load_report = {"missing": len(missing), "unexpected": len(unexpected)}
        self._processor = CLIPProcessor.from_pretrained(self.model_cfg.clip_path)
        self._tokenizer = CLIPTokenizer.from_pretrained(self.model_cfg.clip_path)
        warn = self._model.patch_mapping_warning()
        if warn:
            print(f"  ⚠️  [{self.name}] {warn}")

    def encode_documents(self, task):
        import torch

        feats = []
        imgs = task.load_images()
        with torch.no_grad():
            for i in range(0, len(imgs), self.batch_size):
                batch = imgs[i:i + self.batch_size]
                px = self._processor(images=batch, return_tensors="pt")["pixel_values"].to(self.device)
                feats.append(self._model.encode_vision(px).cpu().numpy())
        return np.concatenate(feats, axis=0).astype(np.float32)  # (blocks, n_rows, D)

    def score_query(self, query: str, docs, task=None):
        import torch

        with torch.no_grad():
            enc = self._tokenizer(query, padding="max_length", max_length=77,
                                  truncation=True, return_tensors="pt").to(self.device)
            t = self._model.encode_text(enc["input_ids"], enc["attention_mask"])[0]
            temp = self._model.temperature.detach()
            rows = torch.from_numpy(docs).to(self.device)
            scores = block_scores(rows, t, temp, self.aggregation).cpu().numpy()
            # 记录预测块的行内最优行，便于分析"行定位"是否真的被用到
            if rows.dim() == 3:
                sims = torch.einsum("bnd,d->bn", rows, t)
                best_row = sims.argmax(dim=-1).cpu().numpy().tolist()
            else:
                best_row = [-1] * rows.shape[0]
        return scores.astype(np.float32), {"row_scores_best": best_row}


# --------------------------------------------------------------------------- #
# ColPali（多向量 late interaction）
# --------------------------------------------------------------------------- #
class ColPaliScorer(Scorer):
    kind = "image_encoder"
    uses_images = True
    supports_precompute = False   # 变量长多向量索引，缓存格式另算，先不做

    def __init__(self, name: str, model_id: str, base_model_dir: str | None = None,
                 adapter_dir: str | None = None, batch_size: int = 4,
                 dtype: str = "bfloat16", device: str | None = None):
        super().__init__(name, {"model_id": model_id})
        self.model_id = model_id
        self.base_model_dir = base_model_dir
        self.adapter_dir = adapter_dir
        self.batch_size = batch_size
        self.dtype_name = dtype
        self.device = device
        self._model = None
        self._processor = None

    def prepare(self):
        import torch
        from colpali_engine.models import ColPali, ColPaliProcessor

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.dtype_name == "bfloat16" else torch.float32
        if self.base_model_dir and self.adapter_dir:
            from peft import PeftModel
            base = ColPali.from_pretrained(self.base_model_dir, torch_dtype=dtype,
                                           device_map=self.device, local_files_only=True)
            self._model = PeftModel.from_pretrained(base, self.adapter_dir,
                                                    local_files_only=True).eval()
            self._processor = ColPaliProcessor.from_pretrained(self.base_model_dir,
                                                               local_files_only=True)
        else:
            self._model = ColPali.from_pretrained(self.model_id, torch_dtype=dtype,
                                                  device_map=self.device).eval()
            self._processor = ColPaliProcessor.from_pretrained(self.model_id)

    def encode_documents(self, task):
        import torch

        docs = []
        imgs = task.load_images()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                             enabled=torch.cuda.is_available()):
            for i in range(0, len(imgs), self.batch_size):
                batch = imgs[i:i + self.batch_size]
                enc = self._encode_batch(batch)
                if "pixel_values" in enc:
                    enc["pixel_values"] = enc["pixel_values"].to(self._model.dtype)
                docs.append(self._model(**enc).float().cpu().numpy())
        return docs

    def _encode_batch(self, batch):
        """把一批页面图变成模型输入。

        必须用 colpali-engine 的 `process_images`：手工拼 `text=["<image>"]` 时，
        transformers 5.x 的 PaliGemma processor 会套用默认 `max_length=50`，
        把 1024 个图像 token 截掉，然后报
            Mismatch in `image` token count between text and `input_ids`
        （这正是上一轮 ColPali 720/720 全失败的原因）。
        """
        try:
            return self._processor.process_images(batch).to(self.device)
        except AttributeError:                # 老版 colpali-engine 没有这个接口
            return self._processor(text=["<image>"] * len(batch), images=batch,
                                   return_tensors="pt", truncation=False).to(self.device)

    def score_query(self, query: str, docs, task=None):
        import torch

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                             enabled=torch.cuda.is_available()):
            try:
                enc = self._processor.process_queries([query]).to(self.device)
            except AttributeError:
                enc = self._processor.tokenizer(f"Query: {query}", return_tensors="pt",
                                                padding=True).to(self.device)
            q = self._model(**enc).float()
        q_mask = enc["attention_mask"][0].float()
        out = []
        for d in docs:
            sims = torch.einsum("nd,bmd->bnm", q[0], torch.from_numpy(d).to(self.device))
            max_sims = sims.max(dim=-1).values * q_mask.unsqueeze(0)
            out.append(max_sims.sum(dim=-1).cpu().numpy())
        return np.concatenate(out).astype(np.float32), {}

class ColQwen2Scorer(Scorer):
    """ColQwen2 / ColQwen2.5 —— ColPali 家族的当前最强形态（Qwen2-VL backbone）。

    TMM 版新增的"领域内强势模型"基线。与 ColPali 的区别只在 backbone 与
    processor 的取词方式；查询/文档都是多向量，按 ColBERT 式 MaxSim 打分，
    因此与 ColPali 的协议完全一致（只换模型，不换协议）。

    加载容错（DSE 那次的教训）：
      * transformers 5.x 不再接受 `torch_dtype`，改用 `dtype`，失败则回退；
      * `attn_implementation` 按 sdpa -> eager -> 默认 依次尝试；
      * `use_cache` 在 transformers 5.x 里不再是构造函数参数，失败则去掉重试。
    """

    kind = "image_encoder"
    uses_images = True
    supports_precompute = False

    def __init__(self, name: str, model_id: str, local_dir: str | None = None,
                 adapter_dir: str | None = None, processor_dir: str | None = None,
                 batch_size: int = 1,
                 dtype: str = "bfloat16", device: str | None = None,
                 attn: str = "sdpa", local_files_only: bool = True):
        super().__init__(name, {"model_id": model_id})
        #: ColQwen2 在 HF 上是"基座 + LoRA 适配器"两段式：
        #:   vidore/colqwen2-base（全量）+ vidore/colqwen2-v1.0（LoRA）
        #: local_dir 给基座，adapter_dir 给适配器；只给 adapter 会加载失败。
        self.model_id = local_dir or model_id
        self.adapter_dir = adapter_dir
        self.processor_dir = processor_dir or self.model_id
        self.batch_size = batch_size
        self.dtype_name = dtype
        self.device = device
        self.attn = attn
        self.local_files_only = local_files_only
        self._model = None
        self._processor = None

    def prepare(self):
        import torch
        from colpali_engine.models import ColQwen2, ColQwen2Processor

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.dtype_name == "bfloat16" else torch.float32
        last_err = None
        for kw in ({"attn_implementation": self.attn, "use_cache": False},
                   {"attn_implementation": self.attn},
                   {"attn_implementation": "eager"},
                   {}):
            for dtype_key in ("dtype", "torch_dtype"):
                try:
                    self._model = ColQwen2.from_pretrained(
                        self.model_id, device_map=self.device, local_files_only=self.local_files_only,
                        **{dtype_key: dtype}, **kw).eval()
                    break
                except (TypeError, ValueError) as exc:
                    last_err = exc
            if self._model is not None:
                print(f"  ColQwen2 加载成功（{kw or '默认参数'} / {self.model_id}）")
                break
        if self._model is None:
            raise RuntimeError(f"ColQwen2 加载失败: {type(last_err).__name__}: {last_err}")
        if self.adapter_dir:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(
                self._model, self.adapter_dir, local_files_only=self.local_files_only).eval()
            print(f"  ColQwen2 适配器已挂载：{self.adapter_dir}")
        self._processor = ColQwen2Processor.from_pretrained(
            self.processor_dir, local_files_only=self.local_files_only)

    def encode_documents(self, task):
        import torch

        docs = []
        imgs = task.load_images()
        with torch.no_grad():
            for i in range(0, len(imgs), self.batch_size):
                batch = imgs[i:i + self.batch_size]
                enc = self._processor.process_images(batch).to(self.device)
                if "pixel_values" in enc:
                    enc["pixel_values"] = enc["pixel_values"].to(self._model.dtype)
                docs.append(self._model(**enc).float().cpu().numpy())
        return docs

    def score_query(self, query: str, docs, task=None):
        import torch

        with torch.no_grad():
            enc = self._processor.process_queries([query]).to(self.device)
            q = self._model(**enc).float()
        q_mask = enc["attention_mask"][0].float()
        out = []
        for d in docs:
            sims = torch.einsum("nd,bmd->bnm", q[0], torch.from_numpy(d).to(self.device))
            max_sims = sims.max(dim=-1).values * q_mask.unsqueeze(0)
            out.append(max_sims.sum(dim=-1).cpu().numpy())
        return np.concatenate(out).astype(np.float32), {}


# --------------------------------------------------------------------------- #
# 通用图文双塔（Jina-CLIP-v2 / DSE 类 checkpoint / 微调过的 CLIP）
# --------------------------------------------------------------------------- #
class ImageTextDualEncoderScorer(Scorer):
    """适用于任何提供 encode_image/encode_text 或 get_image_features/get_text_features
    的 HF 模型——Jina-CLIP-v2、DSE 类截图检索模型、微调后的 CLIP 都归这一类。

    原仓库的 Jina 基线之所以全是 0.0000，正是这一类模型接错预处理/归一化的典型症状；
    这里显式做了 L2 归一化，并要求 batch>1 时输出维度自检。
    """

    kind = "image_encoder"
    uses_images = True

    def __init__(self, name: str, model_id: str, checkpoint: str | None = None,
                 batch_size: int = 32, device: str | None = None,
                 auto_process: bool = True, image_size: int | None = None):
        super().__init__(name, {"model_id": model_id})
        self.model_id = checkpoint or model_id
        self.batch_size = batch_size
        self.device = device
        self.auto_process = auto_process
        self.image_size = image_size
        self._model = None
        self._processor = None

    def prepare(self):
        import torch
        from transformers import AutoModel

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Jina-CLIP 的实现代码里 `from transformers.models.clip.modeling_clip import
        # clip_loss`，而新版 transformers 已删除该符号 —— 不打补丁会直接 ImportError。
        # （旧仓库的 compare.py 里也有同样的 shim；这正是当年 Jina 基线全 0 的原因之一）
        try:
            import transformers.models.clip.modeling_clip as _mc
            if not hasattr(_mc, "clip_loss"):
                _mc.clip_loss = lambda *a, **k: None
        except Exception:
            pass

        # 先按"只用本地缓存"加载（离线服务器场景）；缓存里没有才允许联网下载
        try:
            local_dir = Path(self.model_id)
            if local_dir.exists():
                # 自定义模型（如 jina-clip-v2）会在 __init__ 里用 config._name_or_path
                # 去加载 tokenizer / image processor；离线时必须把它指向本地目录，
                # 否则它会去 hf.co 找模型仓库而失败。
                from transformers import AutoConfig
                cfg = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True,
                                                 local_files_only=True)
                cfg._name_or_path = str(local_dir.resolve())
                self._model = AutoModel.from_pretrained(
                    self.model_id, config=cfg, trust_remote_code=True,
                    local_files_only=True).to(self.device).eval()
            else:
                self._model = AutoModel.from_pretrained(
                    self.model_id, trust_remote_code=True,
                    local_files_only=True).to(self.device).eval()
        except Exception as exc:
            print(f"  本地缓存里没有 {self.model_id}（{type(exc).__name__}），尝试在线下载 ...")
            self._model = AutoModel.from_pretrained(
                self.model_id, trust_remote_code=True).to(self.device).eval()
        if self.auto_process:
            try:
                from transformers import AutoProcessor
                self._processor = AutoProcessor.from_pretrained(self.model_id,
                                                                trust_remote_code=True,
                                                                local_files_only=True)
            except Exception:
                self._processor = None

    def _prep_images(self, imgs):
        if self._processor is not None:
            return self._processor(images=imgs, return_tensors="pt")["pixel_values"].to(self.device)
        from torchvision import transforms
        size = self.image_size or 224
        tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                                 transforms.Normalize([0.5] * 3, [0.5] * 3)])
        import torch
        return torch.stack([tf(im) for im in imgs]).to(self.device)

    def _image_features(self, imgs):
        import torch
        import torch.nn.functional as F

        if hasattr(self._model, "encode_image"):
            feats = self._model.encode_image(imgs)
            feats = torch.as_tensor(np.asarray(feats), device=self.device) \
                if not torch.is_tensor(feats) else feats
            return F.normalize(feats.float(), dim=-1)
        px = self._prep_images(imgs).to(self._model.dtype if hasattr(self._model, "dtype") else None)
        out = self._model(pixel_values=px) if not hasattr(self._model, "get_image_features") \
            else self._model.get_image_features(pixel_values=px)
        feat = out if torch.is_tensor(out) else out.pooler_output
        return F.normalize(feat.float(), dim=-1)

    def _text_features(self, texts):
        import torch
        import torch.nn.functional as F

        if hasattr(self._model, "encode_text"):
            feats = self._model.encode_text(texts)
            feats = feats if torch.is_tensor(feats) else torch.as_tensor(np.asarray(feats),
                                                                         device=self.device)
            return F.normalize(feats.float(), dim=-1)
        if self._processor is None:
            raise RuntimeError(f"{self.model_id} 需要 AutoProcessor 才能编码文本")
        enc = self._processor(text=texts, return_tensors="pt", padding=True,
                              truncation=True).to(self.device)
        out = self._model.get_text_features(**enc)
        feat = out if torch.is_tensor(out) else out.pooler_output
        return F.normalize(feat.float(), dim=-1)

    def encode_documents(self, task):
        import torch

        imgs = task.load_images()
        vecs = []
        with torch.no_grad():
            for i in range(0, len(imgs), self.batch_size):
                vecs.append(self._image_features(imgs[i:i + self.batch_size]).cpu().numpy())
        docs = np.concatenate(vecs, axis=0).astype(np.float32)
        if not np.isfinite(docs).all():
            raise RuntimeError(f"{self.name}: 图像特征出现 NaN/Inf，请检查预处理与 dtype")
        if np.allclose(docs.std(axis=0), 0):
            raise RuntimeError(
                f"{self.name}: 所有图像特征几乎完全相同——这正是原仓库 Jina 基线"
                "全 0.0000 的症状（预处理/分辨率/权重不匹配）。禁止把该结果写进表格。"
            )
        return docs

    def score_query(self, query: str, docs, task=None):
        import torch

        with torch.no_grad():
            q = self._text_features([query])[0].cpu().numpy()
        return (docs @ q).astype(np.float32), {}


# --------------------------------------------------------------------------- #
# OCR：文档图像 -> 文本 -> 文本检索器（模拟真实"扫描件 + OCR"流水线）
# --------------------------------------------------------------------------- #
def ocr_backends() -> dict[str, str]:
    """返回 {后端名: 不可用原因}（原因为空字符串表示可用）。

    支持两种后端，避免"服务器装不了 tesseract 就只能放弃 OCR 基线"：
      * tesseract —— 精度好，但需要系统安装二进制
      * rapidocr  —— 纯 pip（onnxruntime + 自带 ONNX 模型），无需系统依赖
    """
    out: dict[str, str] = {}

    try:
        import pytesseract
        # tesseract 装了但不在 PATH 是常见情况（尤其 Windows / 自定义安装路径），
        # 这里主动找一遍再放弃。
        if os.environ.get("TESSERACT_CMD"):
            pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]
        try:
            pytesseract.get_tesseract_version()
            out["tesseract"] = ""
        except Exception:
            found = None
            import glob
            for cand in ("/usr/bin/tesseract", "/usr/local/bin/tesseract",
                         "/opt/homebrew/bin/tesseract",
                         r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                         r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
                if os.path.exists(cand):
                    found = cand
                    break
            if found is None:
                found = next(iter(glob.glob("/usr/share/tesseract*/tesseract")), None)
            if found:
                pytesseract.pytesseract.tesseract_cmd = found
            try:
                pytesseract.get_tesseract_version()
                out["tesseract"] = ""
            except Exception as exc:
                out["tesseract"] = (f"找不到 tesseract 可执行文件（{type(exc).__name__}）"
                                    "；可设环境变量 TESSERACT_CMD 指向它")
    except ImportError:
        out["tesseract"] = "未安装 pytesseract（pip install pytesseract，纯 Python 包约 20KB）"

    try:
        import rapidocr_onnxruntime  # noqa: F401
        out["rapidocr"] = ""
    except Exception as exc:
        out["rapidocr"] = f"未安装 rapidocr-onnxruntime（{type(exc).__name__}）"

    # PaddleOCR（PP-OCR 系列）是当前文档 AI 里最常用的开源 OCR；
    # RapidOCR 是它的 ONNX 移植，识别模型同源，但依赖轻得多（不用 paddlepaddle）。
    try:
        import paddleocr  # noqa: F401
        out["paddleocr"] = ""
    except Exception as exc:
        out["paddleocr"] = f"未安装 paddleocr（{type(exc).__name__}）"
    return out


def pick_ocr_backend(preferred: str = "auto") -> str | None:
    avail = ocr_backends()
    if preferred != "auto":
        return preferred if not avail.get(preferred, "x") else None
    # 顺序：tesseract（经典、快）-> rapidocr（PP-OCR 模型的轻量版）-> paddleocr（完整版）
    for name in ("tesseract", "rapidocr", "paddleocr"):
        if not avail.get(name, "x"):
            return name
    return None


def ocr_ready(preferred: str = "auto") -> tuple[bool, str]:
    """是否有可用的 OCR 后端；返回 (可用, 说明)。"""
    backend = pick_ocr_backend(preferred)
    if backend:
        return True, backend
    avail = ocr_backends()
    return False, ("没有任何可用的 OCR 后端：\n"
                   + "\n".join(f"    - {k}: {v}" for k, v in avail.items())
                   + "\n  任选其一安装：\n"
                   "    sudo apt-get install -y tesseract-ocr tesseract-ocr-eng   （需 apt/联网）\n"
                   "    pip install rapidocr-onnxruntime                          （纯 pip，无需系统依赖）")


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein 距离（文本短，直接 O(nm) DP，不引入额外依赖）。"""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class OcrEngine:
    """对文档块图像做 OCR，结果带磁盘缓存（多个基线共享同一份 OCR 文本）。

    一个 297 任务的测试集约 1.7 万张图，tesseract 每张几十毫秒；
    而 BM25 / BGE-M3 / ByT5 三个基线用的是**同一份** OCR 结果，所以必须缓存。
    """

    #: 各后端在自己的最佳放大倍数下工作（实测，见 code/00_setup/compare_ocr_engines.py）：
    #:   tesseract 2x -> CER 0.016     rapidocr 4x -> CER 0.046
    #:   tesseract 在 11px 小字上本来就比 PP-OCR 准 5~8 倍，而且保留词间空格
    DEFAULT_SCALE = {"tesseract": 2.0, "rapidocr": 4.0, "paddleocr": 4.0}

    def __init__(self, cache_dir: str = "results/ocr_cache", scale: float | None = None,
                 psm: int = 6, lang: str = "eng", backend: str = "auto"):
        self.backend = pick_ocr_backend(backend)
        if self.backend is None:
            _, why = ocr_ready(backend)
            raise RuntimeError(f"OCR 不可用：{why}")
        if scale is None:
            scale = self.DEFAULT_SCALE.get(self.backend, 3.0)
        # 缓存目录里带上后端与放大倍数：不同后端的 OCR 结果不一样，不能混用
        self.cache_dir = (Path(cache_dir) / f"{self.backend}_x{scale:g}"
                          if cache_dir else None)
        self.scale = scale
        self.psm = psm
        self.lang = lang
        #: tesseract 可多线程并行（每个 tile 一次子进程）；其他后端引擎不是线程安全的
        self.workers = int(os.environ.get("RAVT_OCR_WORKERS", "8"))
        self._engine = self._make_engine()
        print(f"  使用 OCR 后端: {self.backend}（放大 {scale:g}x）")

    def _make_engine(self):
        if self.backend == "rapidocr":
            from rapidocr_onnxruntime import RapidOCR
            return RapidOCR()
        if self.backend == "paddleocr":
            from paddleocr import PaddleOCR
            # v3 与 v2 的构造参数不同，逐个试
            try:      # PaddleOCR 3.x
                return PaddleOCR(lang="en",
                                 use_doc_orientation_classify=False,
                                 use_doc_unwarping=False,
                                 use_textline_orientation=False)
            except TypeError:      # PaddleOCR 2.x
                return PaddleOCR(lang="en", use_angle_cls=True, show_log=False)
        return None                      # tesseract 走 pytesseract 函数式调用

    @staticmethod
    def _paddle_texts(engine, arr) -> str:
        """兼容 PaddleOCR v3（.predict -> dict）与 v2（.ocr -> 嵌套列表）。"""
        if hasattr(engine, "predict"):
            out = engine.predict(arr)
            texts = []
            for page in (out or []):
                if isinstance(page, dict):
                    texts.extend(page.get("rec_texts") or [])
                elif isinstance(page, (list, tuple)):
                    for item in page:
                        if isinstance(item, (list, tuple)) and len(item) >= 2 \
                                and isinstance(item[1], (list, tuple)):
                            texts.append(item[1][0])
            return " ".join(texts)
        out = engine.ocr(arr, cls=True)
        texts = []
        for page in (out or []):
            for line in (page or []):
                try:
                    texts.append(line[1][0])
                except Exception:
                    pass
        return " ".join(texts)

    def _ocr_image(self, path: Path) -> str:
        from PIL import Image
        import numpy as np

        with Image.open(path) as im:
            im = im.convert("RGB")
            # 原图文字只有 11px 高，直接识别几乎认不出，先放大（实测放大 3 倍
            # 后 CER 从不可用降到 ~0.09）
            if self.scale != 1.0:
                im = im.resize((int(im.width * self.scale), int(im.height * self.scale)),
                               Image.LANCZOS)
            if self.backend == "tesseract":
                import pytesseract
                # 关键参数：
                #   --psm 6                    整块文本（我们的 block 是 5 行文本）
                #   preserve_interword_spaces  保留词间空格 —— 不加这个，tesseract
                #                              经常把词粘在一起，词级 BM25 会直接崩
                config = f"--psm {self.psm} -c preserve_interword_spaces=1"
                txt = pytesseract.image_to_string(im.convert("L"), lang=self.lang,
                                                  config=config)
                return " ".join(txt.split())
            if self.backend == "paddleocr":
                return " ".join(self._paddle_texts(self._engine, np.asarray(im)).split())
            res, _ = self._engine(np.asarray(im))
            if not res:
                return ""
            return " ".join(item[1] for item in res)

    def texts_for_task(self, task) -> tuple[list[str], float]:
        """返回 (每块的 OCR 文本, 平均字符错误率 CER)。

        CER 以"渲染时用的文本"（text_haystack，含字符噪声）为参照，
        衡量 OCR 到底把页面上的字认对了多少。

        tesseract 走线程池并行：每个 tile 是一次子进程调用，1 万个 tile 串行要 3~4 小时。
        rapidocr/paddleocr 的引擎对象不是线程安全的，保持串行。
        """
        cache_dir = (self.cache_dir / task.path.name) if self.cache_dir else None
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

        gt_texts = task.load_texts()
        paths = list(task.image_paths)
        out: list[str] = [""] * len(paths)
        todo: list[int] = []
        for i, img_path in enumerate(paths):
            cf = (cache_dir / f"{img_path.stem}.txt") if cache_dir else None
            if cf and cf.exists():
                out[i] = cf.read_text(encoding="utf-8").strip()
            else:
                todo.append(i)

        if todo and self.backend == "tesseract" and self.workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            print(f"  OCR 并行: {len(todo)} 个待识别 / {self.workers} 线程"
                  f"（缓存命中 {len(paths) - len(todo)}）")
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                for i, text in zip(todo, ex.map(lambda j: self._ocr_image(paths[j]), todo)):
                    out[i] = text
        else:
            for i in todo:
                out[i] = self._ocr_image(paths[i])

        if cache_dir:
            for i, img_path in enumerate(paths):
                (cache_dir / f"{img_path.stem}.txt").write_text(out[i], encoding="utf-8")

        cers = []
        for i, text in enumerate(out):
            if i < len(gt_texts):
                gt = " ".join(gt_texts[i].split())
                if gt:
                    cers.append(_edit_distance(text, gt) / len(gt))
        return out, (sum(cers) / len(cers) if cers else float("nan"))


class OcrTextScorer(Scorer):
    """OCR 包装器：文档图像 -> OCR 文本 -> 任意文本检索器。

    查询侧不做 OCR（用户输入本来就是文本），只有文档侧经过 OCR。
    """

    kind = "ocr_text"
    uses_images = True

    def __init__(self, name: str, base: Scorer, engine: OcrEngine):
        super().__init__(name, {"base": base.name, "kind_base": base.kind})
        self.base = base
        self.engine = engine
        self.needs_gpu = base.needs_gpu
        # 包装器要继承基座的"平局"语义（OCR+BM25 里 OCR 把整块读成空串时也会全 0 分）
        self.allows_score_ties = getattr(base, "allows_score_ties", False)
        self._last_cer = float("nan")

    def prepare(self):
        self.base.prepare()

    def encode_documents(self, task):
        texts, cer = self.engine.texts_for_task(task)
        self._last_cer = cer
        if not hasattr(self.base, "encode_texts"):
            raise RuntimeError(f"{self.base.name} 不支持从显式文本编码，无法用于 OCR 流水线")
        return self.base.encode_texts(texts)

    def score_query(self, query: str, docs, task=None):
        scores, extra = self.base.score_query(query, docs, task)
        extra = dict(extra or {})
        extra["ocr_cer"] = (round(float(self._last_cer), 4)
                            if self._last_cer == self._last_cer else None)
        return scores, extra

    def close(self):
        self.base.close()


# --------------------------------------------------------------------------- #
# 原生 CLIP 零样本（消融：不做任何适配，直接用 backbone 本身）
# --------------------------------------------------------------------------- #
class ClipZeroShotScorer(Scorer):
    """把 CLIP backbone 直接当检索器用，不训练、不加适配器、不做行切割。

    这是本方法的"地板"：视觉塔的 CLS 向量 ↔ 文本塔的 CLS 向量做余弦。
    用来回答"我们的融合/投影/行对齐/训练到底带来了多少增益"。

    注意：本仓库的 `clip_local_weights_224` 是**分离导出**的视觉塔 + 文本塔
    （没有 CLIP 原生的 visual_projection / text_projection），所以两侧 CLS
    未必在同一空间里 —— 分数偏低本身就是结论："未经适配的 CLIP 不能直接用于
    渲染文本检索"。这一点要在论文里如实写。
    """

    kind = "clip_zeroshot"
    uses_images = True

    def __init__(self, name: str, clip_path: str, batch_size: int = 64,
                 device: str | None = None):
        super().__init__(name, {"clip_path": clip_path})
        self.clip_path = clip_path
        self.batch_size = batch_size
        self.device = device
        self._vision = None
        self._text = None
        self._processor = None
        self._tokenizer = None

    def prepare(self):
        import torch
        from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

        # 优先按**完整 CLIP** 加载：CLIP 的 text_projection(768→512) /
        # vision_projection(1024→512) 才把两塔对齐，零样本检索才是良定义的。
        # 本仓库导出的 clip_local_weights_224 是分离的视觉/文本塔，若缺少这两个
        # 投影层，两塔根本不在同一空间（1024 vs 768），上轮就是这么整跑失败的：
        #     matmul: (n?,k),(k,m?) size 768 is different from 1024
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._clip = None
        try:
            self._clip = CLIPModel.from_pretrained(
                self.clip_path, local_files_only=True).to(self.device).eval()
            print(f"  {self.name}: 以完整 CLIPModel 加载（含 text/vision projection）")
        except Exception as exc:
            raise RuntimeError(
                f"CLIP-zero-shot 需要**完整** CLIP 权重（含 text_projection / "
                f"vision_projection），但 {self.clip_path} 里没有：{type(exc).__name__}: {exc}\n"
                "  该目录是本项目导出的分离塔（视觉塔 + text_encoder/），两塔维度不同，"
                "无法直接做零样本检索。\n"
                "  两个选择：① 该基线直接不跑（用 --models 里去掉 CLIP）；"
                "② 提供原生 CLIP 权重目录（如 openai/clip-vit-large-patch14）后再跑。"
            ) from None
        self._processor = CLIPProcessor.from_pretrained(self.clip_path,
                                                        local_files_only=True)
        self._tokenizer = CLIPTokenizer.from_pretrained(self.clip_path,
                                                        local_files_only=True)
        print(f"  {self.name}: 未做任何训练/适配，直接使用 backbone（零样本基线）")

    def encode_documents(self, task):
        import torch
        import torch.nn.functional as F

        imgs = task.load_images()
        vecs = []
        with torch.no_grad():
            for i in range(0, len(imgs), self.batch_size):
                px = self._processor(images=imgs[i:i + self.batch_size],
                                     return_tensors="pt")["pixel_values"].to(self.device)
                # get_image_features 内部会过 vision_projection，保证与文本塔同空间
                feat = self._clip.get_image_features(pixel_values=px)
                vecs.append(F.normalize(feat.float(), dim=-1).cpu().numpy())
        return np.concatenate(vecs, axis=0).astype(np.float32)

    def score_query(self, query: str, docs, task=None):
        import torch
        import torch.nn.functional as F

        with torch.no_grad():
            enc = self._tokenizer(query, padding="max_length", max_length=77,
                                  truncation=True, return_tensors="pt").to(self.device)
            feat = self._clip.get_text_features(input_ids=enc["input_ids"],
                                                attention_mask=enc["attention_mask"])
            t = F.normalize(feat.float(), dim=-1)[0].cpu().numpy()
        return (docs @ t).astype(np.float32), {}


# --------------------------------------------------------------------------- #
# DSE（Document Screenshot Embedding, Ma et al. 2024）
# 审稿人 3 点名"最接近的已有工作"，论文里必须引用并对比
# --------------------------------------------------------------------------- #
class DseScorer(Scorer):
    """DSE 单向量稠密检索器：文档截图 -> 1 个向量，文本查询 -> 1 个向量。

    调用方式来自官方 README（Tevatron/dse-phi3-docmatix-v1）：
        查询:  "query: {q}</s>"，max_length=128
        文档:  "<|image_1|>\\nWhat is shown in this image?</s>" + 图像，max_length=4096
        向量:  取最后一层 hidden state 在"最后一个非 padding 位置"的值，再 L2 归一化
    """

    kind = "image_encoder"
    uses_images = True
    supports_precompute = True

    def __init__(self, name: str, model_id: str, batch_size: int = 1,
                 dtype: str = "bfloat16", attn: str = "sdpa",
                 device: str | None = None):
        super().__init__(name, {"model_id": model_id, "attn": attn})
        self.model_id = model_id
        self.batch_size = batch_size
        self.dtype_name = dtype
        self.attn = attn
        self.device = device
        self._model = None
        self._processor = None

    def prepare(self):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}[self.dtype_name]
        if self.device == "cpu":
            dtype = torch.float32          # CPU 上 bf16 很慢
        kw = dict(trust_remote_code=True, local_files_only=True)

        self._processor = AutoProcessor.from_pretrained(self.model_id, **kw)

        # DSE 的 config.json 里存着 "_attn_implementation": "flash_attention_2"，
        # 而 Phi3V 的自定义建模代码是**直接读 config._attn_implementation** 的，
        # 所以只传 attn_implementation= 参数不生效 —— 必须改写 config 本身。
        cfg = AutoConfig.from_pretrained(self.model_id, **kw)
        attempts = [self.attn] + [x for x in ("sdpa", "eager") if x != self.attn]
        last_exc = None
        for impl in attempts:
            try:
                cfg._attn_implementation = impl          # 关键：改写 config
                self._model = self._load_model(AutoModelForCausalLM, cfg, dtype, kw)
                if impl != self.attn:
                    print(f"  {self.name}: 注意力实现回退到 {impl}")
                return
            except (ValueError, TypeError, ImportError, AttributeError) as exc:
                last_exc = exc
                print(f"  {self.name}: attn={impl} 失败（{type(exc).__name__}: "
                      f"{str(exc)[:90]}），换下一种")
        raise SystemExit(
            f"❌ 加载 DSE 失败：{type(last_exc).__name__}: {last_exc}\n"
            "   排查顺序：\n"
            "     1) 若 TypeError 提到某个关键字参数 —— 是 transformers 版本与该模型"
            "保存格式不兼容（本项目已在 v5 上踩过 use_cache / torch_dtype 两个坑，"
            "换台机器或换版本可复现）；\n"
            f"     2) 若显存不足 —— 4.2B 模型 bf16 约需 8.4GB 显存（当前设备 {self.device}）；\n"
            "     3) 若权重缺失 —— models_local/dse-phi3/ 下应有 2 个 .bin 分片。")

    def _load_model(self, cls, cfg, dtype, kw):
        """兼容不同 transformers 版本的签名差异。

        踩过的两个坑（transformers v5 + trust_remote_code 模型）：
          * `torch_dtype` 被改名为 `dtype`；
          * `use_cache` 不再是构造参数（v5 从 config 读），传了会 TypeError。
        这里按组合逐个尝试，自动剔除不被接受的参数。
        """
        cfg.use_cache = False          # v5 的正确做法：写在 config 上
        last_exc = None
        for dtype_kw in ({"dtype": dtype}, {"torch_dtype": dtype}):
            for cache_kw in ({}, {"use_cache": False}):
                try:
                    model = cls.from_pretrained(self.model_id, config=cfg,
                                                **dtype_kw, **cache_kw, **kw)
                    model.config.use_cache = False
                    return model.to(self.device).eval()
                except TypeError as exc:
                    last_exc = exc
                    continue
        raise last_exc

    @staticmethod
    def _pool(hidden, mask):
        import torch
        import torch.nn.functional as F
        idx = mask.sum(dim=1) - 1                      # 最后一个非 padding 位置
        reps = hidden[torch.arange(hidden.shape[0], device=hidden.device), idx]
        return F.normalize(reps.float(), p=2, dim=-1)

    def _encode_images(self, imgs):
        import torch
        vecs = []
        with torch.no_grad():
            for i in range(0, len(imgs), max(1, self.batch_size)):
                chunk = imgs[i:i + max(1, self.batch_size)]
                prompts = ["<|image_1|>\nWhat is shown in this image?</s>"] * len(chunk)
                enc = self._processor(prompts, images=chunk, return_tensors="pt",
                                      padding="longest", max_length=4096,
                                      truncation=True).to(self.device)
                for k in ("input_ids", "attention_mask", "image_sizes"):
                    if k in enc and enc[k].dim() == 3:
                        enc[k] = enc[k].squeeze(0)     # 官方 README 里的 squeeze
                out = self._model(**enc, return_dict=True, output_hidden_states=True)
                vecs.append(self._pool(out.hidden_states[-1], enc["attention_mask"]).cpu().numpy())
        return np.concatenate(vecs, axis=0).astype(np.float32)

    def encode_documents(self, task):
        return self._encode_images(task.load_images())

    def score_query(self, query: str, docs, task=None):
        import torch
        with torch.no_grad():
            enc = self._processor([f"query: {query}</s>"], return_tensors="pt",
                                  max_length=128, truncation=True).to(self.device)
            out = self._model(**enc, return_dict=True, output_hidden_states=True)
            q = self._pool(out.hidden_states[-1], enc["attention_mask"])[0].cpu().numpy()
        return (docs @ q).astype(np.float32), {}


# --------------------------------------------------------------------------- #
# 工厂
# --------------------------------------------------------------------------- #
def build_scorer(spec, device: str | None = None) -> Scorer:
    """把 EvalSpec 变成具体的 Scorer 实例。"""
    extra = dict(getattr(spec, "extra", {}) or {})
    if spec.kind == "bm25":
        return Bm25Scorer(spec.name, tokenizer=spec.tokenizer)
    if spec.kind == "text_encoder":
        extra_te = dict(getattr(spec, "extra", {}) or {})
        return TextEncoderScorer(spec.name, model_id=spec.model_id, checkpoint=spec.checkpoint,
                                 pooling=spec.pooling, batch_size=spec.batch_size,
                                 device=device,
                                 encoder_class=extra_te.get("encoder_class", "auto"))
    if spec.kind == "ocr_text":
        # 先 OCR 再交给文本检索器：base 由 extra["base_kind"] / ["base_tokenizer"] 指定
        extra_ocr = dict(getattr(spec, "extra", {}) or {})
        engine = OcrEngine(cache_dir=extra_ocr.get("ocr_cache", "results/ocr_cache"),
                           scale=extra_ocr.get("ocr_scale"),   # None -> 按后端取默认
                           psm=extra_ocr.get("ocr_psm", 6),
                           backend=extra_ocr.get("ocr_backend", "auto"))
        base_kind = extra_ocr.get("base_kind", "bm25")
        if base_kind == "bm25":
            base = Bm25Scorer(f"{spec.name}-base", tokenizer=extra_ocr.get("base_tokenizer", "word"))
        elif base_kind == "text_encoder":
            base = TextEncoderScorer(f"{spec.name}-base",
                                     model_id=spec.model_id,
                                     checkpoint=extra_ocr.get("base_model_path"),
                                     pooling=spec.pooling or "cls",
                                     batch_size=spec.batch_size, device=device,
                                     encoder_class=extra_ocr.get("encoder_class", "auto"))
        else:
            raise ValueError(f"未知的 OCR 基座类型: {base_kind}")
        return OcrTextScorer(spec.name, base, engine)
    if spec.kind == "ravt":
        return RavtScorer(spec.name, clip_path=extra.get("clip_path", "clip_local_weights_224"),
                          checkpoint=spec.checkpoint, patch_size=spec.patch_size,
                          n_rows=spec.n_rows, row_height=spec.row_height,
                          aggregation=spec.aggregation, batch_size=spec.batch_size,
                          device=device, **extra.get("ravt_flags", {}))
    if spec.kind == "image_encoder":
        if "colqwen" in spec.model_id.lower() or extra.get("family") == "colqwen2":
            return ColQwen2Scorer(spec.name, spec.model_id,
                                  local_dir=extra.get("local_dir"),
                                  adapter_dir=extra.get("adapter_dir"),
                                  processor_dir=extra.get("processor_dir"),
                                  batch_size=spec.batch_size, dtype=spec.dtype,
                                  attn=extra.get("attn", "sdpa"), device=device)
        if "colpali" in spec.model_id.lower():
            return ColPaliScorer(spec.name, spec.model_id,
                                 base_model_dir=extra.get("base_model_dir"),
                                 adapter_dir=extra.get("adapter_dir"),
                                 batch_size=spec.batch_size, dtype=spec.dtype, device=device)
        if "dse" in spec.model_id.lower() or extra.get("family") == "dse":
            return DseScorer(spec.name, spec.model_id, batch_size=spec.batch_size,
                             dtype=spec.dtype, attn=extra.get("attn", "sdpa"),
                             device=device)
        return ImageTextDualEncoderScorer(spec.name, spec.model_id, checkpoint=spec.checkpoint,
                                          batch_size=spec.batch_size, device=device,
                                          image_size=extra.get("image_size"))
    if spec.kind == "clip_zeroshot":
        return ClipZeroShotScorer(spec.name, clip_path=extra.get("clip_path",
                                                                "clip_local_weights_224"),
                                  batch_size=spec.batch_size, device=device)
    raise ValueError(f"未知的 scorer 类型: {spec.kind}")
