"""整数（定点）推理参考实现，并与整图浮点推理做裂缝判定对照。

网络结构与 conv.py 固定一致：
  Layer 1: 1 -> 4 通道, 3x3 核, stride=2, dilation=1, padding=1, ReLU
  Layer 2: 4 -> 1 通道, 3x3 核, stride=1, dilation=2, padding=2, 无激活

量化口径（按需求固定）：
  * 首层输入直接取灰度字节 uint8，输入尺度 s_x = 1/255；
  * 权重按给定正尺度量化：q = clip(round_half_even(w / s_w), -128, 127)，
    即饱和到 int8；
  * 偏置尺度 = 该层输入尺度 × 该层权重尺度，偏置按最近整数、遇半取偶量化，
    量化结果必须落在 int32 内；
  * 首层卷积的带偏置点积累加器必须落在 int32 内；ReLU 后按给定的激活尺度
    重定标并饱和到 uint8（round half to even），次层直接读取这些字节；
  * 次层卷积的带偏置点积累加器同样必须落在 int32 内，其整数累加值即该路
    的“分数”，>= 0 判为裂缝。

任何一步出现非有限值、量化偏置或带偏置点积超出 int32，都抛 QuantError，
由上层只展示原因、不保留任何旧结果图。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .conv import (
    KERNEL,
    L1_OUT_CH,
    L1_PAD,
    L1_STRIDE,
    L2_DILATION,
    L2_IN_CH,
    L2_PAD,
    layer1_out_size,
    run_full_baseline,
    validate_weights,
)

INT8_MIN, INT8_MAX = -128, 127
UINT8_MIN, UINT8_MAX = 0, 255
INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1

INPUT_SCALE = 1.0 / 255.0  # 首层灰度字节输入尺度，固定


class QuantError(ValueError):
    """量化/整数推理过程中的硬性失败（上溢、非有限值等）。"""


# --------------------------------------------------------------------------
# 基础量化原语
# --------------------------------------------------------------------------
def round_half_even(x) -> np.ndarray:
    """最近整数舍入，遇半取偶（banker's rounding），向量化实现。

    基于 floor 实现，因此对负数同样成立（如 -2.5 -> -2，-3.5 -> -4）。
    输入含 NaN/Inf 时抛 QuantError。
    """
    arr = np.asarray(x, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise QuantError("舍入输入中出现非有限值（NaN/Inf），计算中止")
    n = np.floor(arr)
    frac = arr - n
    is_even = np.mod(n, 2.0) == 0.0
    rounded = np.where(
        frac < 0.5, n,
        np.where(frac > 0.5, n + 1.0, np.where(is_even, n, n + 1.0)),
    )
    return rounded


def _check_finite(arr: np.ndarray, what: str) -> None:
    if not np.isfinite(arr).all():
        raise QuantError(f"{what}中出现非有限值（NaN/Inf），计算中止")


def quantize_int8(weight: np.ndarray, scale: float, name: str
                  ) -> tuple[np.ndarray, int]:
    """权重量化到 int8 并饱和，返回 (int8 数组, 被饱和裁剪的元素数)。"""
    scaled = weight / scale
    _check_finite(scaled, f"{name} 权重除以尺度后的中间值")
    raw = round_half_even(scaled).astype(np.int64)
    clipped = np.clip(raw, INT8_MIN, INT8_MAX)
    saturated = int(np.count_nonzero(raw != clipped))
    return clipped.astype(np.int8), saturated


def quantize_bias(bias: np.ndarray, scale: float, name: str) -> np.ndarray:
    """偏置按给定尺度（= 输入尺度 × 权重尺度）量化，硬性要求落在 int32 内。"""
    scaled = bias / scale
    _check_finite(scaled, f"{name} 偏置除以尺度后的中间值")
    qb = round_half_even(scaled)
    if qb.size and (float(qb.min()) < INT32_MIN or float(qb.max()) > INT32_MAX):
        raise QuantError(
            f"{name} 量化偏置超出 int32 范围 "
            f"[{INT32_MIN}, {INT32_MAX}]：实际范围 "
            f"[{int(qb.min())}, {int(qb.max())}]，计算中止")
    return qb.astype(np.int64)


def _check_acc_int32(acc: np.ndarray, name: str) -> None:
    lo, hi = int(acc.min()), int(acc.max())
    if lo < INT32_MIN or hi > INT32_MAX:
        idx = np.unravel_index(int(np.argmax(np.abs(acc).astype(np.float64))),
                               acc.shape)
        where = ",".join(str(int(i)) for i in idx)
        raise QuantError(
            f"{name} 带偏置点积超出 int32 范围 [{INT32_MIN}, {INT32_MAX}]："
            f"实际范围 [{lo}, {hi}]，首个极值位置（数组索引）=({where})，"
            f"计算中止")


# --------------------------------------------------------------------------
# 结果容器
# --------------------------------------------------------------------------
@dataclass
class QuantComparison:
    output_shape: list[int]
    float_output: np.ndarray          # (1,ho,wo) float64，整图浮点分数
    quant_acc: np.ndarray             # (1,ho,wo) int64，整数路分数（累加器）
    quant_activation: np.ndarray      # (4,ho,wo) uint8，首层 ReLU 重定标字节
    float_crack: np.ndarray           # (ho,wo) bool
    quant_crack: np.ndarray           # (ho,wo) bool
    new_mask: np.ndarray              # 量化新增：quant 有、float 无
    gone_mask: np.ndarray             # 量化消失：float 有、quant 无
    new_coords: list[list[int]]       # [row, col]
    gone_coords: list[list[int]]
    weight_saturated: dict[str, int]
    scales: dict[str, float]
    counts: dict[str, int] = field(default_factory=dict)


def _mask_coords(mask: np.ndarray) -> list[list[int]]:
    rows, cols = np.nonzero(mask)
    return [[int(r), int(c)] for r, c in zip(rows, cols)]


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------
def run_quant_comparison(image_u8: np.ndarray, weights: dict,
                         scale_w1: float, scale_w2: float,
                         scale_a1: float) -> QuantComparison:
    """两路推理 + 判定对照。image_u8 为 uint8 二维灰度字节图。"""
    for label, s in (("layer1 权重尺度", scale_w1),
                     ("layer2 权重尺度", scale_w2),
                     ("首层 ReLU 输出尺度", scale_a1)):
        if not np.isfinite(s) or s <= 0:
            raise QuantError(f"{label}必须是正的有限数，收到 {s!r}")
    if image_u8.dtype != np.uint8 or image_u8.ndim != 2:
        raise QuantError("整数推理输入必须是二维 uint8 灰度字节数组")

    w1, b1, w2, b2 = validate_weights(weights)
    h, w = image_u8.shape
    ho, wo = layer1_out_size(h, w)

    # ---- 权重量化（饱和到 int8）----
    qw1, sat1 = quantize_int8(w1, scale_w1, "layer1")
    qw2, sat2 = quantize_int8(w2, scale_w2, "layer2")

    # ---- 偏置尺度 = 该层输入尺度 × 权重尺度 ----
    scale_b1 = INPUT_SCALE * scale_w1
    scale_b2 = scale_a1 * scale_w2
    _check_finite(np.array([scale_b1, scale_b2]), "偏置尺度")
    qb1 = quantize_bias(b1, scale_b1, "layer1")
    qb2 = quantize_bias(b2, scale_b2, "layer2")

    # ==================================================================
    # Layer 1：uint8 输入（零填充）× int8 权重 + int32 偏置，int64 累加
    # ==================================================================
    x = image_u8.astype(np.int64)
    xp = np.pad(x, ((L1_PAD, L1_PAD), (L1_PAD, L1_PAD)), mode="constant")
    acc1 = np.zeros((L1_OUT_CH, ho, wo), dtype=np.int64)
    acc1 += qb1.astype(np.int64)[:, None, None]
    for oc in range(L1_OUT_CH):
        for kh in range(KERNEL):
            for kw in range(KERNEL):
                # stride=2, dilation=1：起点即 (kh, kw)，步长 2
                patch = xp[kh:kh + L1_STRIDE * ho:L1_STRIDE,
                           kw:kw + L1_STRIDE * wo:L1_STRIDE]
                acc1[oc] += int(qw1[oc, 0, kh, kw]) * patch
    _check_acc_int32(acc1, "layer1")

    # ---- ReLU（在实数域）后按 scale_a1 重定标，饱和到 uint8 ----
    acc1_relu = np.maximum(acc1, 0)
    a1_real = acc1_relu * scale_b1          # 反量化后的 ReLU 输出
    _check_finite(a1_real, "layer1 ReLU 输出")
    qa1_raw = round_half_even(a1_real / scale_a1)
    _check_finite(qa1_raw, "layer1 重定标中间值")
    qa1 = np.clip(qa1_raw, UINT8_MIN, UINT8_MAX).astype(np.uint8)

    # ==================================================================
    # Layer 2：直接读取 uint8 激活字节（零填充），dilation=2
    # ==================================================================
    a1p = np.pad(qa1.astype(np.int64),
                 ((0, 0), (L2_PAD, L2_PAD), (L2_PAD, L2_PAD)),
                 mode="constant")
    offs = [kh * L2_DILATION for kh in range(KERNEL)]
    acc2 = np.full((ho, wo), int(qb2[0]), dtype=np.int64)
    for ic in range(L2_IN_CH):
        plane = a1p[ic]
        for kh in range(KERNEL):
            r0 = offs[kh]
            for kw in range(KERNEL):
                c0 = offs[kw]
                acc2 += (int(qw2[0, ic, kh, kw])
                         * plane[r0:r0 + ho, c0:c0 + wo])
    _check_acc_int32(acc2, "layer2")

    # ==================================================================
    # 浮点整图基准与判定对照（分数 >= 0 判裂缝）
    # ==================================================================
    float_img = image_u8.astype(np.float64) / 255.0
    float_out = run_full_baseline(float_img, w1, b1, w2, b2)
    _check_finite(float_out, "浮点推理输出")

    float_score = float_out[0]
    quant_score = acc2
    float_crack = float_score >= 0.0
    quant_crack = quant_score >= 0
    new_mask = quant_crack & ~float_crack
    gone_mask = float_crack & ~quant_crack

    counts = {
        "grid_cells": int(ho * wo),
        "float_crack": int(np.count_nonzero(float_crack)),
        "quant_crack": int(np.count_nonzero(quant_crack)),
        "agreed_crack": int(np.count_nonzero(float_crack & quant_crack)),
        "new_crack": int(np.count_nonzero(new_mask)),
        "gone_crack": int(np.count_nonzero(gone_mask)),
    }

    return QuantComparison(
        output_shape=[1, ho, wo],
        float_output=float_out,
        quant_acc=acc2[None, :, :],
        quant_activation=qa1,
        float_crack=float_crack,
        quant_crack=quant_crack,
        new_mask=new_mask,
        gone_mask=gone_mask,
        new_coords=_mask_coords(new_mask),
        gone_coords=_mask_coords(gone_mask),
        weight_saturated={"layer1": sat1, "layer2": sat2},
        scales={"input": INPUT_SCALE, "weight1": float(scale_w1),
                "weight2": float(scale_w2), "activation1": float(scale_a1),
                "bias1": float(scale_b1), "bias2": float(scale_b2)},
        counts=counts,
    )
