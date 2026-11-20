"""量化对照图渲染：四联图（输入 / 浮点判定 / 整数判定 / 叠加差异）。

每个面板的每个输出网格单元按 CELL 像素等比放大，输出网格行列与输入图像的
对照关系由页面文字说明；仅使用 Pillow 绘图，不引入额外字体依赖。
颜色约定：
  * 输入面板：原始灰度；
  * 浮点/整数面板：裂缝=深红，非裂缝=浅灰；
  * 叠加面板：两路都判裂缝=绿，量化新增=红，量化消失=蓝，一致非裂缝=浅灰。
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

CELL = 12          # 每个输出网格单元的边长（像素）
GAP = 4            # 面板间留白
BG = (255, 255, 255)
GRAY = (226, 230, 235)
CRACK_RED = (185, 28, 28)
AGREE_GREEN = (21, 128, 61)
NEW_RED = (220, 38, 38)
GONE_BLUE = (37, 99, 235)


def _mask_panel(mask: np.ndarray) -> Image.Image:
    h, w = mask.shape
    img = Image.new("RGB", (w * CELL, h * CELL), GRAY)
    px = img.load()
    rows, cols = np.nonzero(mask)
    for r, c in zip(rows.tolist(), cols.tolist()):
        for dy in range(CELL):
            for dx in range(CELL):
                px[c * CELL + dx, r * CELL + dy] = CRACK_RED
    return img


def _overlay_panel(float_mask: np.ndarray, quant_mask: np.ndarray,
                   new_mask: np.ndarray, gone_mask: np.ndarray) -> Image.Image:
    h, w = float_mask.shape
    img = Image.new("RGB", (w * CELL, h * CELL), GRAY)
    px = img.load()

    def paint(mask, color):
        rows, cols = np.nonzero(mask)
        for r, c in zip(rows.tolist(), cols.tolist()):
            for dy in range(CELL):
                for dx in range(CELL):
                    px[c * CELL + dx, r * CELL + dy] = color

    paint(float_mask & quant_mask, AGREE_GREEN)
    paint(new_mask, NEW_RED)
    paint(gone_mask, GONE_BLUE)
    return img


def render_comparison_png(image_u8: np.ndarray, result) -> bytes:
    """渲染四联对照图，返回 PNG 字节。"""
    ho, wo = result.float_crack.shape

    # 输入按输出网格尺寸放大（最近邻），保证四个面板严格对齐
    input_panel = Image.fromarray(image_u8, mode="L").convert("RGB").resize(
        (wo * CELL, ho * CELL), Image.Resampling.NEAREST)
    float_panel = _mask_panel(result.float_crack)
    quant_panel = _mask_panel(result.quant_crack)
    overlay_panel = _overlay_panel(
        result.float_crack, result.quant_crack,
        result.new_mask, result.gone_mask)

    panels = [input_panel, float_panel, quant_panel, overlay_panel]
    pw, ph = panels[0].size
    total_w = 4 * pw + 5 * GAP
    total_h = ph + 2 * GAP
    canvas = Image.new("RGB", (total_w, total_h), BG)
    for i, panel in enumerate(panels):
        x = GAP + i * (pw + GAP)
        canvas.paste(panel, (x, GAP))

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()
