"""两层卷积的 NumPy 参考实现（全部 float64）。

网络结构（固定）：
  Layer 1: 1 -> 4 通道, 3x3 核, stride=2, padding=1, ReLU
  Layer 2: 4 -> 1 通道, 3x3 核, stride=1, dilation=2, padding=2

本模块提供两套独立实现：
  * run_full_baseline : 整图推理，显式零填充后整体向量化计算；
  * TiledRunner       : 分块推理，每块只截取覆盖它所需的最小图内激活
                        矩形窗口，在局部窗口上重算两层卷积。

两套实现的索引与求和顺序刻意不同（baseline 用滑窗 + einsum，
tiled 用掩码 gather + 逐 tap 矩阵乘），用于交叉验证：差异应仅为浮点
舍入量级（远小于 1e-8）。
"""

from __future__ import annotations

import numpy as np

# ---- 固定网络超参 ----
L1_IN_CH = 1
L1_OUT_CH = 4
L2_IN_CH = L1_OUT_CH
L2_OUT_CH = 1
KERNEL = 3
L1_STRIDE = 2
L1_PAD = 1
L2_DILATION = 2
L2_PAD = 2
MAX_TILE_SIDE = 64

DTYPE = np.float64


def layer1_out_size(h: int, w: int) -> tuple[int, int]:
    """第一层（也是网络最终）输出空间尺寸。"""
    ho = (h + 2 * L1_PAD - KERNEL) // L1_STRIDE + 1
    wo = (w + 2 * L1_PAD - KERNEL) // L1_STRIDE + 1
    return ho, wo


def validate_weights(weights: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """校验并返回 (w1, b1, w2, b2)，全部转为 float64。

    接受的 JSON 结构：
      {"layer1": {"weight": [4][1][3][3], "bias": [4]},
       "layer2": {"weight": [1][4][3][3], "bias": [1]}}
    也兼容顶层直接给出 layer1_weight / layer1_bias / layer2_weight / layer2_bias。
    """
    if not isinstance(weights, dict):
        raise ValueError("权重 JSON 必须是对象")

    l1 = weights.get("layer1")
    l2 = weights.get("layer2")
    if isinstance(l1, dict) and isinstance(l2, dict):
        w1, b1 = l1.get("weight"), l1.get("bias")
        w2, b2 = l2.get("weight"), l2.get("bias")
    else:
        w1 = weights.get("layer1_weight", weights.get("w1"))
        b1 = weights.get("layer1_bias", weights.get("b1"))
        w2 = weights.get("layer2_weight", weights.get("w2"))
        b2 = weights.get("layer2_bias", weights.get("b2"))

    if any(x is None for x in (w1, b1, w2, b2)):
        raise ValueError("缺少权重或偏置字段，需要 layer1/layer2 各自的 weight 与 bias")

    w1 = np.asarray(w1, dtype=DTYPE)
    b1 = np.asarray(b1, dtype=DTYPE)
    w2 = np.asarray(w2, dtype=DTYPE)
    b2 = np.asarray(b2, dtype=DTYPE)

    if w1.shape != (L1_OUT_CH, L1_IN_CH, KERNEL, KERNEL):
        raise ValueError(f"layer1.weight 形状应为 (4,1,3,3)，实际 {w1.shape}")
    if b1.shape != (L1_OUT_CH,):
        raise ValueError(f"layer1.bias 形状应为 (4,)，实际 {b1.shape}")
    if w2.shape != (L2_OUT_CH, L2_IN_CH, KERNEL, KERNEL):
        raise ValueError(f"layer2.weight 形状应为 (1,4,3,3)，实际 {w2.shape}")
    if b2.shape != (L2_OUT_CH,):
        raise ValueError(f"layer2.bias 形状应为 (1,)，实际 {b2.shape}")
    if not all(np.isfinite(x).all() for x in (w1, b1, w2, b2)):
        raise ValueError("权重/偏置中存在 NaN 或 Inf")
    return w1, b1, w2, b2


# --------------------------------------------------------------------------
# 整图基准：显式零填充 + 滑窗 + einsum
# --------------------------------------------------------------------------
def run_full_baseline(image: np.ndarray, w1, b1, w2, b2) -> np.ndarray:
    """独立整图基准推理，返回 (1, ho, wo) float64。"""
    if image.ndim != 2:
        raise ValueError("图像必须是二维灰度数组")
    x = np.asarray(image, dtype=DTYPE)[None, :, :]  # (1,H,W)
    h, w = image.shape
    ho, wo = layer1_out_size(h, w)

    # ---- Layer 1: pad -> 滑窗 -> einsum -> bias -> ReLU ----
    xp = np.pad(x, ((0, 0), (L1_PAD, L1_PAD), (L1_PAD, L1_PAD)), mode="constant")
    wins = np.lib.stride_tricks.sliding_window_view(xp, (KERNEL, KERNEL),
                                                    axis=(1, 2))
    wins = wins[:, ::L1_STRIDE, ::L1_STRIDE]       # (1,ho,wo,3,3)
    a1 = np.einsum("ihwxy,oixy->ohw", wins, w1)    # (4,ho,wo)
    a1 = np.maximum(a1 + b1[:, None, None], 0.0)

    # ---- Layer 2: dilation=2, pad=2，按膨胀偏移切片求和 ----
    a1p = np.pad(a1, ((0, 0), (L2_PAD, L2_PAD), (L2_PAD, L2_PAD)),
                 mode="constant")
    offs = [kh * L2_DILATION for kh in range(KERNEL)]
    acc = np.zeros((ho, wo), dtype=DTYPE)
    for ic in range(L2_IN_CH):
        plane = a1p[ic]
        for kh in range(KERNEL):
            r0 = offs[kh]
            for kw in range(KERNEL):
                c0 = offs[kw]
                acc += w2[0, ic, kh, kw] * plane[r0:r0 + ho, c0:c0 + wo]
    return (acc + b2[0])[None, :, :]


# --------------------------------------------------------------------------
# 分块推理
# --------------------------------------------------------------------------
def required_input_window(r0: int, r1: int, c0: int, c1: int,
                          h: int, w: int) -> tuple[int, int, int, int]:
    """输出块（layer2/layer1 输出坐标，半开区间）反推所需的最小图内激活窗口。

    感受野链路（layer1 stride=2，layer2 dilation=2）：
      layer2 输出 (i,j) 读 layer1 输出坐标 i + 2*(kh-1)，即 [i-2, i+2]；
      layer1 输出 k 读输入坐标 2k + (kh-1)，即 [2k-1, 2k+1]。
      合并后单像素输入行窗口为 [2i-5, 2i+5]（11 行）；
      输出块 r0..r1-1 的行窗口为 [2*r0-5, 2*(r1-1)+5]，列同理。
    只保留落在图内 [0,H) x [0,W) 的部分（图外为零填充，无需物化）。
    """
    ir0 = max(2 * r0 - 5, 0)
    ir1 = min(2 * (r1 - 1) + 6, h)
    ic0 = max(2 * c0 - 5, 0)
    ic1 = min(2 * (c1 - 1) + 6, w)
    return ir0, ir1, ic0, ic1


def _gather_valid(plane: np.ndarray, gr: np.ndarray, gc: np.ndarray,
                  base_r: int, base_c: int, dim_r: int, dim_c: int):
    """按全局坐标从 plane gather 成 (th,tw,3,3)，越界位置按零填充处理。

    gr 形状 (th,3)：每个输出行、每个核行 tap 的全局行坐标；
    gc 形状 (tw,3)：每个输出列、每个核列 tap 的全局列坐标。
    不构造任何填充数组。
    """
    lr = (gr - base_r)[:, None, :, None]   # (th,1,3,1)
    lc = (gc - base_c)[None, :, None, :]   # (1,tw,1,3)
    valid = (lr >= 0) & (lr < dim_r) & (lc >= 0) & (lc < dim_c)
    lr = np.broadcast_to(np.clip(lr, 0, dim_r - 1), valid.shape)
    lc = np.broadcast_to(np.clip(lc, 0, dim_c - 1), valid.shape)
    return plane[lr, lc] * valid


class TiledRunner:
    """在固定方形分块边长下执行分块推理，并记录图内激活窗口峰值用量。"""

    def __init__(self, image: np.ndarray, w1, b1, w2, b2, tile: int):
        if not 1 <= tile <= MAX_TILE_SIDE:
            raise ValueError(f"分块边长须在 1..{MAX_TILE_SIDE} 之间")
        self.image = np.asarray(image, dtype=DTYPE)
        self.h, self.w = self.image.shape
        self.w1, self.b1, self.w2, self.b2 = w1, b1, w2, b2
        self.tile = tile
        self.ho, self.wo = layer1_out_size(self.h, self.w)
        self.out = np.zeros((1, self.ho, self.wo), dtype=DTYPE)
        # 图内激活元素数峰值（输入通道数计入；本网络输入 1 通道）
        self.peak_input_elems = 0
        # 含全部通道的层1中间张量峰值（参考值，不计入预算判定）
        self.peak_layer1_elems = 0
        self._tap = np.arange(KERNEL)

    def _conv_tile(self, ir0, ir1, ic0, ic1, r0, r1, c0, c1):
        """在局部输入窗口上重算两层卷积，写回输出块。

        驻留的图内激活只有 patch 这一个最小矩形窗口；图外位置经掩码按 0
        处理，不物化零填充。layer2 的膨胀感受野会读到块外 ±2 的 layer1
        激活，因此先在同一输入窗口上重算带 halo 的局部 layer1 区域
        （坐标裁到合法输出界 [0,ho)x[0,wo)），再在其上做 layer2。
        索引数组为整型簿记量，不计激活预算。
        """
        patch = self.image[ir0:ir1, ic0:ic1]  # (ph, pw)
        ph, pw = patch.shape
        th, tw = r1 - r0, c1 - c0

        # layer2 需要的 layer1 输出坐标范围（含 halo），裁到合法输出界
        R0 = max(r0 - 2, 0)
        R1 = min(r1 + 2, self.ho)
        C0 = max(c0 - 2, 0)
        C1 = min(c1 + 2, self.wo)
        lh, lw = R1 - R0, C1 - C0

        # ---- Layer 1（stride=2, pad=1），在 halo 区域上重算 ----
        ar_h = np.arange(lh)[:, None]
        ac_h = np.arange(lw)[:, None]
        gr1 = 2 * R0 + 2 * ar_h + (self._tap - 1)[None, :]  # (lh,3)
        gc1 = 2 * C0 + 2 * ac_h + (self._tap - 1)[None, :]  # (lw,3)
        gathered = _gather_valid(patch, gr1, gc1, ir0, ic0, ph, pw)
        a1 = np.einsum("xykh,qkh->qxy", gathered, self.w1[:, 0, :, :])
        a1 = np.maximum(a1 + self.b1[:, None, None], 0.0)  # (4,lh,lw)

        # ---- Layer 2（dilation=2, pad=2），在局部 a1 上 gather ----
        ar = np.arange(th)[:, None]
        ac = np.arange(tw)[:, None]
        gr2 = r0 + ar + 2 * (self._tap - 1)[None, :]        # (th,3)
        gc2 = c0 + ac + 2 * (self._tap - 1)[None, :]        # (tw,3)
        acc = np.full((th, tw), self.b2[0], dtype=DTYPE)
        for ic in range(L2_IN_CH):
            vals = _gather_valid(a1[ic], gr2, gc2, R0, C0, lh, lw)
            acc += np.einsum("xykh,kh->xy", vals, self.w2[0, ic])
        self.out[0, r0:r1, c0:c1] = acc

        self.peak_input_elems = max(self.peak_input_elems, ph * pw * L1_IN_CH)
        self.peak_layer1_elems = max(self.peak_layer1_elems, a1.size)

    def run(self) -> np.ndarray:
        t = self.tile
        for r0 in range(0, self.ho, t):
            r1 = min(r0 + t, self.ho)
            for c0 in range(0, self.wo, t):
                c1 = min(c0 + t, self.wo)
                ir0, ir1, ic0, ic1 = required_input_window(
                    r0, r1, c0, c1, self.h, self.w)
                self._conv_tile(ir0, ir1, ic0, ic1, r0, r1, c0, c1)
        return self.out
