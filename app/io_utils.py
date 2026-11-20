"""上传内容解析与校验。"""

from __future__ import annotations

import io
import json

import numpy as np
from PIL import Image

MAX_SIDE = 512


def load_grayscale_png(data: bytes) -> np.ndarray:
    """读取上传的 PNG，要求灰度模式，每边 <= 512，返回 float64 二维数组。

    像素值归一化到 [0,1]。拒绝多通道（RGB/RGBA/P）等非灰度图像。
    """
    if not data:
        raise ValueError("图像文件为空")
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception as exc:  # Pillow 的异常类型较杂
        raise ValueError(f"无法解析为 PNG 图像：{exc}") from exc
    if im.format != "PNG":
        raise ValueError(f"只接受 PNG 图像，收到 {im.format}")
    if im.mode != "L":
        raise ValueError(f"只接受灰度 PNG（mode=L），收到 mode={im.mode}")
    if im.width > MAX_SIDE or im.height > MAX_SIDE:
        raise ValueError(
            f"图像每边至多 {MAX_SIDE} 像素，收到 {im.width}x{im.height}")
    arr = np.asarray(im, dtype=np.float64) / 255.0
    return arr


def load_weights_json(data: bytes) -> dict:
    if not data:
        raise ValueError("权重 JSON 文件为空")
    try:
        obj = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"权重文件不是合法 JSON：{exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError("权重 JSON 顶层必须是对象")
    return obj
