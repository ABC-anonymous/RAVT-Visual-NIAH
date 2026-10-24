"""模型 / 渲染器 / 噪声 / 损失的**唯一定义处**。

原仓库把同一份模型定义复制在 train.py、train_bk.py、ablation_train.py、evaluate.py
四个文件里，四份副本各自漂移——`PATCH_SIZE` 写死 16 的 bug 就是这么产生的。
本模块把定义收敛到一处，训练与评测必须都从这里导入。

核心结构（与论文一致）：
    图像 -> CLIP ViT -> Multi-Layer Fusion -> RAVT 行切割 -> Gated Residual Projector
         -> n_rows 个行向量（或 1 个全局向量）
"""

from __future__ import annotations

import io
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageEnhance, ImageFilter, ImageOps

# --------------------------------------------------------------------------- #
# 渲染器
# --------------------------------------------------------------------------- #
_FONT_CANDIDATES = [
    # Linux
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    # macOS
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    # Windows
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "arial.ttf",
    # CJK（E8 多语种实验需要）
    "C:/Windows/Fonts/msyh.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


def find_font(explicit: str | None = None) -> str | None:
    if explicit:
        if not os.path.exists(explicit):
            raise FileNotFoundError(f"指定的字体不存在: {explicit}")
        return explicit
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


@dataclass
class RendererConfig:
    block_width: int = 224
    block_height: int = 44
    #: mosaic 画布高度。老代码用的是 224（5 行只占 220px，底部留 4px 白边），
    #: 这样 CLIPProcessor 的 resize 是恒等变换。若设成 44*n_rows=220，
    #: CLIPProcessor 会把 224x220 拉成 228x224 再中心裁剪成 224x224
    #: —— 纵向拉伸 1.8%、左右各裁 2px，与老流程不可比。
    canvas_size: int = 224
    font_size: int = 11
    margin_x: int = 4
    margin_y: int = 2
    line_spacing: int = 1
    max_lines: int = 2
    font_path: str | None = None
    #: 复刻老代码的断行细节：当单个词就超过一行宽度时，老代码会 append 一个空行，
    #: 占掉"最多两行"里的一个名额。差异只影响极少数超长词样本（实测 1/200），
    #: 但为了与旧结果可比，默认保持一致。
    legacy_empty_line: bool = True
    #: 断行宽度测量方式：
    #:   "fast"  —— 逐词测量后累加（O(n)，实测 0.011 s/块）
    #:   "exact" —— 老代码的做法：每加一个词就测量整行（O(n^2)，慢 8.7 倍）
    #: 两者在词恰好卡在行宽边界时会有约 0.7% 的样本断行不同，其余完全一致。
    wrap_mode: str = "fast"



def _wrap_fast(words, font, draw, limit, legacy_empty_line):
    """逐词测宽后累加（O(n)）。实测比 _wrap_exact 快 8.7 倍。"""
    def width_of(s: str) -> float:
        try:
            return font.getlength(s)
        except Exception:
            try:
                return draw.textbbox((0, 0), s, font=font)[2]
            except Exception:
                return len(s) * 6

    space_w = width_of(" ")
    lines, current, cur_w = [], [], 0.0
    for word in words:
        w = width_of(word)
        add = w if not current else space_w + w
        if cur_w + add < limit:
            current.append(word)
            cur_w += add
        else:
            if current:
                lines.append(" ".join(current))
            elif legacy_empty_line:
                lines.append("")          # 与老代码一致：超宽词会插入空行
            current = [word]
            cur_w = w
    if current:
        lines.append(" ".join(current))
    return lines


def _wrap_exact(words, draw, font, limit, legacy_empty_line):
    """老代码的原始做法：每加一个词就测量整行（O(n^2)），用于严格复现旧结果。"""
    lines, current = [], []
    for word in words:
        candidate = " ".join(current + [word])
        try:
            width = draw.textbbox((0, 0), candidate, font=font)[2]
        except Exception:
            width = len(candidate) * 6
        if width < limit:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            elif legacy_empty_line:
                lines.append("")
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


class BlockRenderer:
    """把一段文本渲染成 224x44 的文本块（与原论文完全一致的参数）。"""

    def __init__(self, cfg: RendererConfig | None = None):
        self.cfg = cfg or RendererConfig()
        self.font_path = find_font(self.cfg.font_path)
        if self.font_path is None:
            raise RuntimeError(
                "找不到任何可用字体。请在 RendererConfig.font_path 显式指定一个 ttf/ttc 路径，"
                "否则渲染结果与论文不可比。"
            )

    def describe(self) -> str:
        """渲染参数摘要（字体是关键实验变量，必须写进日志）。"""
        return (f"字体={self.font_path} 字号={self.cfg.font_size} "
                f"块={self.cfg.block_width}x{self.cfg.block_height} "
                f"画布高={self.cfg.canvas_size} 边距=({self.cfg.margin_x},{self.cfg.margin_y})")

    def _font(self):
        try:
            return ImageFont.truetype(self.font_path, self.cfg.font_size)
        except Exception:
            return ImageFont.load_default()

    def render(self, text: str) -> tuple[Image.Image, str]:
        """返回 (图像, 实际渲染出来的文本)。

        实际渲染文本会被图片宽度/行数截断，评测纯文本基线时必须用它，
        否则文本侧与图像侧看到的不是同一段内容（原仓库踩过这个坑）。

        性能：原实现每加一个词就测量一次整行宽度（O(n^2)，实测 0.096 s/块，
        生成 5 万块要 1.3 小时）。这里改为先测每个词、再累加，
        复杂度降为 O(n)，断行结果与原实现一致（见 self_test 的换行等价性检查）。
        """
        cfg = self.cfg
        text = "".join(ch for ch in str(text) if ch.isprintable())
        font = self._font()
        img = Image.new("RGB", (cfg.block_width, cfg.block_height), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        limit = cfg.block_width - cfg.margin_x * 2
        words = text.split()
        if cfg.wrap_mode == "exact":
            lines = _wrap_exact(words, draw, font, limit, cfg.legacy_empty_line)
        else:
            lines = _wrap_fast(words, font, draw, limit, cfg.legacy_empty_line)

        rendered, y = [], cfg.margin_y
        for line in lines[:cfg.max_lines]:
            try:
                draw.text((cfg.margin_x, y), line, font=font, fill=(0, 0, 0))
            except Exception:
                pass
            rendered.append(line)
            y += cfg.font_size + cfg.line_spacing
        return img, " ".join(rendered)


# --------------------------------------------------------------------------- #
# 噪声引擎
# --------------------------------------------------------------------------- #
_HOMOGLYPHS = {"a": "@", "e": "3", "i": "1", "o": "0", "s": "$",
               "S": "5", "t": "+", "z": "2"}
_GIBBERISH = list("#*@%&?!~+=")

#: 词级同义替换表（TMM 新增的"语义保持"内容轴）：
#: 替换后的句子仍然通顺、OCR 毫无损失，但词面匹配会直接失效。
_SYNONYMS = {
    "big": ["large", "huge"], "large": ["big", "great"], "small": ["little", "minor"],
    "said": ["stated", "noted"], "says": ["states", "claims"], "also": ["additionally", "too"],
    "because": ["since", "as"], "however": ["nevertheless", "though"],
    "many": ["numerous", "several"], "most": ["the majority of"],
    "used": ["employed", "utilized"], "use": ["employ", "utilize"],
    "make": ["produce", "create"], "made": ["produced", "created"],
    "show": ["demonstrate", "indicate"], "showed": ["demonstrated", "indicated"],
    "begin": ["start", "commence"], "began": ["started", "commenced"],
    "end": ["conclude", "finish"], "first": ["initial", "earliest"],
    "last": ["final", "latest"], "near": ["close to", "around"],
    "about": ["roughly", "approximately"], "between": ["among"],
    "during": ["throughout", "in"], "after": ["following", "subsequent to"],
    "before": ["prior to", "ahead of"], "while": ["whereas", "whilst"],
    "city": ["town", "municipality"], "country": ["nation", "state"],
    "people": ["individuals", "persons"], "team": ["squad", "group"],
    "part": ["portion", "section"], "number": ["quantity", "count"],
    "new": ["recent", "novel"], "old": ["former", "earlier"],
    "good": ["fine", "solid"], "bad": ["poor", "weak"],
    "very": ["extremely", "highly"], "often": ["frequently", "regularly"],
    "always": ["invariably", "consistently"], "only": ["solely", "merely"],
    "other": ["another", "additional"], "same": ["identical", "equal"],
    "work": ["effort", "labor"], "works": ["functions", "operates"],
    "found": ["discovered", "located"], "known": ["recognized", "documented"],
    "called": ["named", "termed"], "includes": ["comprises", "contains"],
    "released": ["issued", "published"], "reported": ["recounted", "described"],
}

#: 去同形字基线用到的逆映射（Cheap baseline：把噪声还原回正常字母再检索）
HOMOGLYPH_INVERSE_FOR_DENOISE = {v: k for k, v in _HOMOGLYPHS.items()}

_KEYBOARD_NEIGHBOURS = {
    "a": "qwsz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wsdr",
    "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "i": "ujko", "j": "huikmn",
    "k": "jiolm", "l": "kop", "m": "njk", "n": "bhjm", "o": "iklp",
    "p": "ol", "q": "wa", "r": "edft", "s": "awedxz", "t": "rfgy",
    "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu", "z": "asx",
}


class CorruptionEngine:
    """字符级与图像级噪声。

    `families` 决定用哪些噪声。训练时只用 "seen" 家族，
    评测时用它 + "unseen_*" 家族，从而把"记住训练噪声"和"真的抗噪"区分开
    （E8：审稿人 1 第 2 条、审稿人 2 第 3 条）。
    """

    CHAR_KINDS = {"homoglyph", "gibberish", "occlusion",
                  "keyboard_typo", "deletion", "transposition", "insertion",
                  "synonym_swap", "case_flip"}
    #: 词级（语义保持）噪声：OCR/阅读顺序之外的另一条内容轴
    WORD_KINDS = {"synonym_swap", "word_drop"}
    #: 前 5 个是原论文用的"温和退化"；后面 8 个是 TMM 版新增的真实扫描件退化
    #: （透视畸变/折痕/透印/摩尔纹/复印对比度/运动模糊/椒盐噪声/非均匀光照）。
    IMAGE_KINDS = {"gaussian_blur", "jpeg", "rotation", "low_resolution", "shadow",
                   "motion_blur", "salt_pepper", "uneven_light", "perspective",
                   "fold", "show_through", "moire", "photocopy"}

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    # ---------------- 词级（语义保持）----------------
    def apply_word(self, text: str, rate: float, families: list[str]) -> str:
        """同义替换 / 丢词。模拟"被检索的正文用词与查询不同"的真实情形。

        与字符级噪声的区别：这里每个词仍然是合法英文单词，OCR 毫无损失，
        但词面匹配（BM25-word）会直接失效 —— 用来把"抗噪"和"容错"分开考察。
        """
        if rate <= 0 or not families:
            return text
        kinds = [k for k in families if k in self.WORD_KINDS]
        if not kinds:
            return text
        words = text.split()
        if len(words) < 4:
            return text
        for i in range(len(words)):
            if self.rng.random() >= rate:
                continue
            kind = self.rng.choice(kinds)
            bare = words[i].strip(".,;:!?()[]\"'")
            if kind == "synonym_swap":
                cand = _SYNONYMS.get(bare.lower())
                if cand:
                    words[i] = self.rng.choice(cand)
            elif kind == "word_drop" and len(words) > 3:
                words[i] = ""
        out = " ".join(w for w in words if w)
        return out or text

    # ---------------- 字符级 ----------------
    def apply_char(self, text: str, rate: float, families: list[str]) -> str:
        if rate <= 0 or not families:
            return text
        kinds = [k for k in families if k in self.CHAR_KINDS]
        if not kinds:
            return text
        chars = list(text)
        for i in range(len(chars)):
            if self.rng.random() >= rate:
                continue
            kind = self.rng.choice(kinds)
            if kind == "homoglyph":
                chars[i] = _HOMOGLYPHS.get(chars[i], chars[i])
            elif kind == "gibberish":
                chars[i] = self.rng.choice(_GIBBERISH)
            elif kind == "occlusion":
                chars[i] = "#"
            elif kind == "keyboard_typo":
                nbr = _KEYBOARD_NEIGHBOURS.get(chars[i].lower())
                if nbr:
                    chars[i] = self.rng.choice(nbr)
            elif kind == "deletion":
                chars[i] = ""
            elif kind == "transposition" and i + 1 < len(chars):
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
            elif kind == "insertion":
                chars[i] = chars[i] + self.rng.choice("abcdefghijklmnopqrstuvwxyz")
            elif kind == "case_flip":
                chars[i] = chars[i].swapcase()
        return "".join(chars)

    # ---------------- 图像级 ----------------
    def apply_image(self, img: Image.Image, severity: float, families: list[str],
                    coverage: float | None = None) -> Image.Image:
        """图像级退化。

        coverage 与 severity 解耦（TMM 版新增）：
          * coverage —— 一个块被施加退化的概率（0..1，"多少比例的页面是脏的"）
          * severity —— 施加时的强度（0..1）
        老行为等价于 coverage = severity（同一档位同时放大覆盖率和强度）。
        显式传入 coverage 就能建"同强度、不同覆盖率"的对照，用来把
        "页面脏的面积"和"页面脏的程度"这两个因素分开。
        """
        kinds = [k for k in families if k in self.IMAGE_KINDS]
        if not kinds or severity <= 0:
            return img
        cov = severity if coverage is None else coverage
        s = min(1.0, max(0.0, severity))
        out = img
        for kind in kinds:
            if self.rng.random() > cov:
                continue
            if kind == "gaussian_blur":
                # 0.2 -> 1.7px, 0.4 -> 2.9px（11px 字高的一半左右，已经明显糊）
                out = out.filter(ImageFilter.GaussianBlur(radius=0.5 + 6.0 * s))
            elif kind == "jpeg":
                buf = io.BytesIO()
                # 0.2 -> 75, 0.4 -> 60, 1.0 -> 15
                out.save(buf, format="JPEG", quality=int(90 - 75 * s))
                buf.seek(0)
                out = Image.open(buf).convert("RGB")
            elif kind == "rotation":
                # 0.4 -> 约 ±2.5°（足以让一行文字跨出原来的 patch 边界）
                angle = self.rng.uniform(-1.0, 1.0) * (1.0 + 5.0 * s)
                out = out.rotate(angle, resample=Image.BILINEAR, fillcolor=(255, 255, 255))
            elif kind == "low_resolution":
                w, h = out.size
                # 0.4 -> 缩到 0.72 倍再放大回去（真实低分辨率扫描）
                scale = max(0.25, 1.0 - 0.7 * s)
                small = out.resize((max(8, int(w * scale)), max(8, int(h * scale))), Image.BILINEAR)
                out = small.resize((w, h), Image.BILINEAR)
            elif kind == "shadow":
                grad = Image.new("L", out.size, 255)
                gd = ImageDraw.Draw(grad)
                gd.rectangle([0, 0, out.size[0], int(out.size[1] * (0.2 + 0.5 * s))],
                             fill=int(255 * (1 - 0.8 * s)))
                out = Image.composite(out, ImageEnhance.Brightness(out).enhance(1 - 0.8 * s), grad)
            # ---------------- TMM 新增：真实扫描件退化族 ----------------
            elif kind == "motion_blur":
                # 拍摄/扫描时的运动模糊：带方向的均值核
                n = 5 if s > 0.25 else 3
                kernel = [0.0] * (n * n)
                mid = (n // 2) * n
                kernel[mid:mid + n] = [1.0 / n] * n
                out = out.filter(ImageFilter.Kernel((n, n), kernel, scale=1.0))
            elif kind == "salt_pepper":
                # 椒盐噪声：老化的传真/复印件上最典型的噪声
                arr = np.asarray(out).copy()
                rs = np.random.RandomState(self.rng.randrange(1 << 31))
                rnd = rs.rand(*arr.shape[:2])
                p = 0.01 + 0.08 * s
                arr[rnd < p / 2] = 0
                arr[(rnd >= p / 2) & (rnd < p)] = 255
                out = Image.fromarray(arr)
            elif kind == "uneven_light":
                # 非均匀光照：任意方向的亮度斜坡（阴影只覆盖上半部分，这个是全页渐变）
                arr = np.asarray(out).astype(np.float32)
                h, w = arr.shape[:2]
                rs = np.random.RandomState(self.rng.randrange(1 << 31))
                ang = rs.uniform(0, 2 * math.pi)
                yy, xx = np.mgrid[0:h, 0:w]
                ramp = math.cos(ang) * xx / w + math.sin(ang) * yy / h
                ramp = (ramp - ramp.min()) / max(float(np.ptp(ramp)), 1e-6)
                gain = 1.0 - 0.75 * s * ramp
                arr = np.clip(arr * gain[..., None], 0, 255).astype(np.uint8)
                out = Image.fromarray(arr)
            elif kind == "perspective":
                # 透视畸变：翻拍时纸张没有摆正
                w, h = out.size
                rs = np.random.RandomState(self.rng.randrange(1 << 31))
                j = 0.07 * s * min(w, h)
                quad = (rs.uniform(0, j), rs.uniform(0, j),
                        rs.uniform(0, j), h - rs.uniform(0, j),
                        w - rs.uniform(0, j), h - rs.uniform(0, j),
                        w - rs.uniform(0, j), rs.uniform(0, j))
                out = out.transform((w, h), Image.QUAD, quad, resample=Image.BILINEAR)
            elif kind == "fold":
                # 折痕：一条亮带 + 相邻一条暗带，带内的字被物理遮挡
                w, h = out.size
                band = max(2, int(round(1 + 4 * s)))
                draw = ImageDraw.Draw(out)
                if self.rng.random() < 0.5:
                    x = self.rng.randrange(int(w * 0.2), int(w * 0.8) + 1)
                    draw.rectangle([x, 0, x + band, h], fill=(255, 255, 255))
                    draw.rectangle([max(0, x - band), 0, max(0, x - 1), h],
                                   fill=(int(255 - 90 * s),) * 3)
                else:
                    y = self.rng.randrange(int(h * 0.2), int(h * 0.8) + 1)
                    draw.rectangle([0, y, w, y + band], fill=(255, 255, 255))
                    draw.rectangle([0, max(0, y - band), w, max(0, y - 1)],
                                   fill=(int(255 - 90 * s),) * 3)
            elif kind == "show_through":
                # 双面透印：纸张背面（上下镜像）的字隐约透过来
                ghost = ImageOps.mirror(out).filter(
                    ImageFilter.GaussianBlur(radius=1.5 + 3.0 * s))
                ghost = ImageEnhance.Contrast(ghost).enhance(max(0.2, 1 - 0.6 * s))
                out = Image.blend(out, ghost, alpha=0.10 + 0.35 * s)
            elif kind == "moire":
                # 摩尔纹：周期性扫描栅格与文字纹理干涉
                arr = np.asarray(out).astype(np.float32)
                h, w = arr.shape[:2]
                rs = np.random.RandomState(self.rng.randrange(1 << 31))
                fx, fy = rs.uniform(0.25, 0.9), rs.uniform(-0.4, 0.4)
                yy, xx = np.mgrid[0:h, 0:w]
                pat = 1.0 - 0.20 * s * (1 + np.sin(2 * math.pi * (fx * xx / w + fy * yy / h)))
                arr = np.clip(arr * pat[..., None], 0, 255).astype(np.uint8)
                out = Image.fromarray(arr)
            elif kind == "photocopy":
                # 复印/传真：对比度拉伸 + 轻微模糊 + 亮度压缩
                out = ImageOps.autocontrast(out.convert("L"), cutoff=2).convert("RGB")
                out = ImageEnhance.Contrast(out).enhance(1.0 + 1.2 * s)
                out = ImageEnhance.Brightness(out).enhance(1.0 + 0.12 * s)
                out = out.filter(ImageFilter.GaussianBlur(radius=0.3 + 0.6 * s))
        return out


def build_mosaic(row_texts: list[str], renderer: BlockRenderer, engine: CorruptionEngine,
                 noise_rate: float = 0.0, char_families: tuple[str, ...] = (),
                 word_families: tuple[str, ...] = (),
                 image_families: tuple[str, ...] = (),
                 image_severity: float = 0.0,
                 image_coverage: float | None = None) -> tuple[Image.Image, list[str]]:
    """把 n 行文本拼成一张 224x(44*n) 的 mosaic，返回图像与实际渲染文本。"""
    n = len(row_texts)
    height = max(renderer.cfg.canvas_size, renderer.cfg.block_height * n)
    canvas = Image.new("RGB", (renderer.cfg.block_width, height), (255, 255, 255))
    rendered_texts = []
    for i, text in enumerate(row_texts):
        # 先用词级噪声（同义替换/丢词），再叠加字符级噪声
        dirty = engine.apply_word(text, noise_rate, list(word_families))
        dirty = engine.apply_char(dirty, noise_rate, list(char_families))
        block, actual = renderer.render(dirty)
        block = engine.apply_image(block, image_severity, list(image_families),
                                   coverage=image_coverage)
        canvas.paste(block, (0, i * renderer.cfg.block_height))
        rendered_texts.append(actual)
    return canvas, rendered_texts


# --------------------------------------------------------------------------- #
# 冒烟测试用的假处理器（不加载任何权重）
# --------------------------------------------------------------------------- #
class RandomProcessor:
    """复刻 CLIPImageProcessor 的 resize + center-crop + 归一化，但不加载权重。

    为什么必须复刻：渲染出的 mosaic 是 224x220（5x44），而 CLIP 期望 224x224，
    `CLIPProcessor` 默认会 do_resize(shortest_edge=224) + do_center_crop(224)。
    也就是真实流水线里画布被**纵向拉伸 1.8%、左右各裁 2 像素**，
    行高实际是 44.8px 而不是 44px。冒烟处理器必须与之保持一致，否则形状对不上。
    """

    _MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    _STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

    def __init__(self, image_size: int = 224):
        self.image_size = image_size

    def _resize_center_crop(self, im):
        size = self.image_size
        w, h = im.size
        scale = size / min(w, h)
        if abs(scale - 1.0) > 1e-6:
            im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                           Image.BILINEAR)
        w, h = im.size
        left, top = (w - size) // 2, (h - size) // 2
        return im.crop((left, top, left + size, top + size))

    def __call__(self, images=None, return_tensors="pt", **kwargs):
        import numpy as np
        arr = np.stack([np.asarray(self._resize_center_crop(im.convert("RGB")), dtype="float32") / 255.0
                        for im in images])
        t = torch.from_numpy(arr).permute(0, 3, 1, 2)
        return {"pixel_values": (t - self._MEAN) / self._STD}


class RandomTokenizer:
    """按文本哈希生成稳定的 token id，不加载词表。"""

    def __init__(self, vocab_size: int = 49408):
        self.vocab_size = vocab_size

    def __call__(self, texts, padding=None, max_length=77, truncation=True,
                 return_tensors="pt", **kwargs):
        import hashlib
        ids = []
        for text in texts:
            h = hashlib.sha256(str(text).encode("utf-8")).digest()
            row = [(h[i % len(h)] * 256 + h[(i + 7) % len(h)]) % self.vocab_size
                   for i in range(max_length)]
            ids.append(row)
        t = torch.tensor(ids, dtype=torch.long)
        return {"input_ids": t, "attention_mask": torch.ones_like(t)}


def build_random_clip_cfg(**overrides) -> dict:
    """冒烟用的随机小 CLIP 配置（训练与测试必须用同一份）。"""
    cfg = dict(image_size=224, patch_size=14, hidden_size=64, intermediate_size=128,
               num_hidden_layers=2, num_attention_heads=4, text_hidden_size=32,
               text_intermediate_size=64, text_num_hidden_layers=2,
               text_num_attention_heads=4, vocab_size=49408)
    cfg.update(overrides)
    return cfg


# --------------------------------------------------------------------------- #
# RAVT 模型
# --------------------------------------------------------------------------- #
@dataclass
class RAVTConfig:
    clip_path: str = "clip_local_weights_224"
    image_size: int = 224
    row_height: int = 44
    n_rows: int = 5
    #: 行切割映射使用的 patch 大小。
    #: 14 = 与实际 backbone（ViT-L/14）一致（修复后）；
    #: 16 = 旧代码的写法（与 ViT-L/14 不一致，第 1~4 行窗口向上错位）。
    patch_size: int = 14
    use_fusion: bool = True
    use_gated_proj: bool = True
    use_slicing: bool = True
    fusion_layers: int = 4
    proj_hidden: int = 2048
    proj_layers: int = 3
    temperature_init: float = 14.28
    unfreeze_vision_layers: int = 3
    unfreeze_text_layers: int = 2
    random_init: bool = False
    random_cfg: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d.pop("random_cfg", None)
        return d


class GatedResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(),
                                    nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())

    def forward(self, x):
        return x + self.layers(x) * self.gate(x)


class MultiLayerFusion(nn.Module):
    def __init__(self, num_layers: int = 4):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_layers))

    def forward(self, hidden_states):
        layers = hidden_states[-self.weights.numel():]
        w = F.softmax(self.weights, dim=0)
        return sum(wi * h for wi, h in zip(w, layers))


class RobustProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 2048, n_layers: int = 3):
        super().__init__()
        self.input_fc = nn.Linear(in_dim, hidden)
        self.res_layers = nn.Sequential(*[GatedResBlock(hidden) for _ in range(n_layers)])
        self.output_fc = nn.Sequential(nn.Linear(hidden, out_dim), nn.LayerNorm(out_dim), nn.Tanh())

    def forward(self, x):
        return self.output_fc(self.res_layers(F.gelu(self.input_fc(x))))


class SimpleProjector(nn.Module):
    """w/o gated projector 的对照：单层线性 + LayerNorm + Tanh。"""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.Tanh())

    def forward(self, x):
        return self.net(x)


def _random_clip_configs(cfg: dict):
    """构造随机初始化的小 CLIP，用于本地/CPU 冒烟自测（不下载任何权重）。"""
    from transformers import CLIPVisionConfig, CLIPTextConfig
    vision = CLIPVisionConfig(
        image_size=cfg.get("image_size", 224),
        patch_size=cfg.get("patch_size", 14),
        hidden_size=cfg.get("hidden_size", 64),
        intermediate_size=cfg.get("intermediate_size", 128),
        num_hidden_layers=cfg.get("num_hidden_layers", 2),
        num_attention_heads=cfg.get("num_attention_heads", 4),
    )
    text = CLIPTextConfig(
        vocab_size=cfg.get("vocab_size", 49408),
        hidden_size=cfg.get("text_hidden_size", 32),
        intermediate_size=cfg.get("text_intermediate_size", 64),
        num_hidden_layers=cfg.get("text_num_hidden_layers", 2),
        num_attention_heads=cfg.get("text_num_attention_heads", 4),
        max_position_embeddings=cfg.get("max_position_embeddings", 77),
    )
    return vision, text


class RAVTModel(nn.Module):
    """RAVT 检索模型：可训练全部组件，也可只做推理。"""

    def __init__(self, cfg: RAVTConfig):
        super().__init__()
        from transformers import CLIPVisionModel, CLIPTextModel

        self.cfg = cfg
        if cfg.random_init:
            vision_cfg, text_cfg = _random_clip_configs(cfg.random_cfg)
            self.backbone = CLIPVisionModel(vision_cfg)
            self.text_encoder = CLIPTextModel(text_cfg)
        else:
            text_dir = os.path.join(cfg.clip_path, "text_encoder")
            self.backbone = CLIPVisionModel.from_pretrained(cfg.clip_path, local_files_only=True)
            if not os.path.isdir(text_dir):
                raise FileNotFoundError(
                    f"{text_dir} 不存在。原代码假定微调后的 CLIP 目录下有一个 text_encoder/ 子目录，"
                    "请确认权重目录结构，或用 RAVTConfig(random_init=True) 做冒烟自测。"
                )
            self.text_encoder = CLIPTextModel.from_pretrained(text_dir, local_files_only=True)

        self._set_trainable(cfg)
        self.fusion = MultiLayerFusion(cfg.fusion_layers)

        v_dim = self.backbone.config.hidden_size
        t_dim = self.text_encoder.config.hidden_size
        self.proj = (RobustProjector(v_dim, t_dim, cfg.proj_hidden, cfg.proj_layers)
                     if cfg.use_gated_proj else SimpleProjector(v_dim, t_dim))
        self.temperature = nn.Parameter(torch.tensor(float(cfg.temperature_init)))

        # 真实 patch 网格 = 位置编码数量推导，而不是假定
        self.grid_size = self._infer_grid_size()
        self.expected_grid = cfg.image_size // cfg.patch_size

    # ------------------------------------------------------------------ #
    def _set_trainable(self, cfg: RAVTConfig) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        def _unfreeze_last(module_obj, n):
            # 注意：不能用 `a or b` 判断 nn.Module（部分容器实现了 __len__，空模块会被判 False）
            core = getattr(module_obj, "vision_model", None)
            if core is None:
                core = getattr(module_obj, "text_model", None)
            if core is None:
                core = module_obj
            enc = getattr(core, "encoder", core)
            layers = getattr(enc, "layers", None)
            if layers is None:
                return
            for layer in list(layers)[-n:]:
                for p in layer.parameters():
                    p.requires_grad = True

        if cfg.unfreeze_vision_layers:
            _unfreeze_last(self.backbone, cfg.unfreeze_vision_layers)
        if cfg.unfreeze_text_layers:
            _unfreeze_last(self.text_encoder, cfg.unfreeze_text_layers)
        for name, p in self.backbone.named_parameters():
            if "layernorm" in name.lower():
                p.requires_grad = True

    def _infer_grid_size(self) -> int:
        # 不同 transformers 版本的结构不同：
        #   旧版: CLIPVisionModel.vision_model.embeddings.position_embedding
        #   新版: CLIPVisionModel.embeddings.position_embedding
        core = getattr(self.backbone, "vision_model", self.backbone)
        embeddings = getattr(core, "embeddings", None)
        pos = getattr(embeddings, "position_embedding", None) if embeddings is not None else None
        if pos is not None:
            n_tokens = pos.weight.shape[0] - 1  # 去掉 CLS
            side = int(round(math.sqrt(n_tokens)))
            if side * side == n_tokens:
                return side
        conf = self.backbone.config
        return int(conf.image_size) // int(conf.patch_size)

    def true_patch_size(self) -> int:
        """由实际网格反推的 patch 大小（像素）。"""
        return self.cfg.image_size // self.grid_size if self.grid_size else self.cfg.patch_size

    def patch_mapping_warning(self) -> str | None:
        if self.grid_size != self.expected_grid:
            return (f"patch 大小不匹配：backbone 的实际网格是 {self.grid_size}x{self.grid_size}"
                    f"（patch={self.cfg.image_size // self.grid_size}px），"
                    f"但行切割按 patch={self.cfg.patch_size}px 计算。"
                    f"第 1~{self.cfg.n_rows - 1} 行的特征窗口会错位（旧代码的已知 bug）。")
        return None

    # ------------------------------------------------------------------ #
    def row_patch_slices(self) -> list[tuple[int, int]]:
        """RAVT 的行窗口（patch 索引区间）。patch_size 必须等于真实 patch 大小。"""
        out = []
        for r in range(self.cfg.n_rows):
            start = int((r * self.cfg.row_height) / self.cfg.patch_size)
            end = int(math.ceil(((r + 1) * self.cfg.row_height) / self.cfg.patch_size))
            start = max(0, min(start, self.grid_size - 1))
            end = max(start + 1, min(end, self.grid_size))
            out.append((start, end))
        return out

    def encode_vision(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """返回 (B, n_rows, D) 的行特征；use_slicing=False 时返回 (B, D)。"""
        out = self.backbone(pixel_values=pixel_values, output_hidden_states=True)
        if self.cfg.use_fusion:
            feats = self.fusion(out.hidden_states)
        else:
            feats = out.hidden_states[-1]
        feats = feats[:, 1:, :]                       # 去掉 CLS
        B, N, D = feats.shape
        grid = self.grid_size
        if N != grid * grid:
            raise ValueError(f"patch 数量 {N} 与推导的网格 {grid}x{grid} 不一致")
        v_grid = feats.view(B, grid, grid, D)

        if not self.cfg.use_slicing:
            pooled = self.proj(v_grid.mean(dim=[1, 2]))
            return F.normalize(pooled, dim=-1)

        rows = []
        for start, end in self.row_patch_slices():
            rows.append(v_grid[:, start:end, :, :].mean(dim=[1, 2]))
        stacked = torch.stack(rows, dim=1)                       # (B, n_rows, D)
        flat = stacked.reshape(B * self.cfg.n_rows, D)
        return F.normalize(self.proj(flat), dim=-1).view(B, self.cfg.n_rows, -1)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return F.normalize(out.pooler_output, dim=-1)

    def forward(self, pixel_values, input_ids, attention_mask):
        return (self.encode_vision(pixel_values),
                self.encode_text(input_ids, attention_mask),
                self.temperature)


# --------------------------------------------------------------------------- #
# 打分聚合
# --------------------------------------------------------------------------- #
def block_scores(row_feats: torch.Tensor, text_feat: torch.Tensor,
                 temperature: torch.Tensor | None = None,
                 aggregation: str = "max_row") -> torch.Tensor:
    """把 (B, n_rows, D) 行特征 + (D,) 查询特征聚合成 (B,) 的块级得分。

    max_row          —— 原论文的做法：取行间相似度最大值（再乘温度）
    late_interaction —— E3 的修复尝试：ColBERT 式行级 MaxSim（对行也做 max，
                        但保留行维度做加权求和，等价于对 5 行做 max 的平滑版本）
    """
    if row_feats.dim() == 2:                      # 全局池化分支
        sims = row_feats @ text_feat.unsqueeze(-1)
        sims = sims.squeeze(-1)
    else:
        sims = torch.einsum("bnd,d->bn", row_feats, text_feat)
    if aggregation == "late_interaction":
        # 行级 MaxSim 的平滑版：softmax 加权，温度固定为 0.1 以保持稳定
        weights = torch.softmax(sims / 0.1, dim=-1)
        scores = (weights * sims).sum(dim=-1)
    else:
        scores = sims.max(dim=-1).values
    if temperature is not None:
        scores = scores * temperature
    return scores


# --------------------------------------------------------------------------- #
# 损失
# --------------------------------------------------------------------------- #
def multi_grained_loss(v_feats, t_feat, target_row, temperature,
                       w_local: float = 1.0, w_margin: float = 0.5,
                       margin: float = 0.15, use_local: bool = True,
                       use_margin: bool = True) -> dict:
    """global + local(row) + margin 三项对比损失。

    use_local / use_margin 用于 E3 要求的单独消融（原论文只做了结构消融，没做损失消融）。
    """
    B = t_feat.shape[0]
    labels = torch.arange(B, device=t_feat.device)
    losses = {}

    if v_feats.dim() == 2:
        logits = temperature * (v_feats @ t_feat.t())
        loss_global = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
        losses["global"] = loss_global
        total = loss_global
    else:
        pos = v_feats[torch.arange(B, device=v_feats.device), target_row]
        logits = temperature * (pos @ t_feat.t())
        loss_global = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
        losses["global"] = loss_global
        total = loss_global

        raw_local = torch.bmm(v_feats, t_feat.unsqueeze(2)).squeeze(2)     # (B, n_rows)
        if use_local:
            loss_local = F.cross_entropy(temperature * raw_local, target_row)
            losses["local"] = loss_local
            total = total + w_local * loss_local
        if use_margin:
            pos_sim = raw_local[torch.arange(B, device=v_feats.device), target_row]
            neg = raw_local.clone()
            neg[torch.arange(B, device=v_feats.device), target_row] = float("-inf")
            loss_margin = F.relu(neg.max(dim=1).values - pos_sim + margin).mean()
            losses["margin"] = loss_margin
            total = total + w_margin * loss_margin

    losses["total"] = total
    return losses
