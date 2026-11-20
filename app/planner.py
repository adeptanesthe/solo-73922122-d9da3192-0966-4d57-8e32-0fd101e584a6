"""分块尺寸选择：在激活预算内选 1..64 中最大的方形输出块边长。

预算口径（按题目）：
  对每个输出分块（边缘块按实际输出裁短），沿感受野链路反推出覆盖该块
  所需的 *图内* 激活最小矩形窗口，元素数 = 窗口高 × 窗口宽 × 输入通道数，
  取所有分块中的最大值作为该边长下的峰值用量。图外零填充不物化、不计入。
  实跑时 TiledRunner 截取的正是同一个窗口，因此运行期峰值与此处预算一致。

窗口链路：layer2 dilation2 使单输出在 layer1 输出上跨度 [i-2,i+2]，
layer1 stride2 再映射到输入 [2r0-5, 2(r1-1)+6)，列同理，最后裁到图界。
"""

from __future__ import annotations

from dataclasses import dataclass

from .conv import (
    L1_IN_CH,
    MAX_TILE_SIDE,
    layer1_out_size,
    required_input_window,
)


@dataclass
class TilePlan:
    feasible: bool
    tile: int | None            # 选中的方形边长（不可行时为 None）
    budget: int                 # 用户预算（元素数）
    minimum_budget: int         # s=1 时所需的最低预算
    peak_input_elems: int       # 选中边长下的实际峰值图内激活元素数
    tile_grid: tuple[int, int]  # (行块数, 列块数)
    per_side_peaks: list[tuple[int, int]]  # 各候选边长 -> 峰值，供页面展示
    output_size: tuple[int, int]


def _peak_for_side(h: int, w: int, ho: int, wo: int, s: int) -> int:
    """边长 s 下，所有分块（含边缘裁短块）所需图内激活元素数的最大值。

    窗口面积关于块高、块宽单调不减，而贴图界的裁剪只会缩小窗口。
    全网格的峰值必落在以下三类候选块之一：
      * 四角块（裁短且贴边，行/列方向都被裁剪）；
      * 严格内部块（不贴任何图界，窗口不被裁剪，通常给出全局最大）；
      * 边中块（仅行或仅列方向贴边）。
    因此只需常数个候选块，无需枚举所有分块。
    """
    row_starts = list(range(0, ho, s))
    col_starts = list(range(0, wo, s))

    def block(r0, c0):
        r1, c1 = min(r0 + s, ho), min(c0 + s, wo)
        ir0, ir1, ic0, ic1 = required_input_window(r0, r1, c0, c1, h, w)
        return (ir1 - ir0) * (ic1 - ic0) * L1_IN_CH

    # 每个轴取“起点序列”的首、中、末三个代表（中部为不贴边的完整块）
    def picks(starts):
        mid = starts[len(starts) // 2]
        out = [starts[0], mid, starts[-1]]
        return sorted(set(out))

    peak = 0
    for r0 in picks(row_starts):
        for c0 in picks(col_starts):
            peak = max(peak, block(r0, c0))
    return peak


def plan_tiles(h: int, w: int, budget: int) -> TilePlan:
    if budget < 1:
        raise ValueError("激活预算必须是正整数（元素数）")
    ho, wo = layer1_out_size(h, w)

    # 候选边长不超过输出本身的尺寸，也不超过 64
    upper = min(MAX_TILE_SIDE, ho, wo)
    per_side: list[tuple[int, int]] = []
    side_peaks: dict[int, int] = {}
    for s in range(1, upper + 1):
        p = _peak_for_side(h, w, ho, wo, s)
        side_peaks[s] = p
        per_side.append((s, p))

    minimum_budget = side_peaks[1]
    if budget < minimum_budget:
        # 单像素块仍超限：不执行推理，只报最低预算
        return TilePlan(
            feasible=False, tile=None, budget=budget,
            minimum_budget=minimum_budget, peak_input_elems=minimum_budget,
            tile_grid=(ho, wo), per_side_peaks=per_side, output_size=(ho, wo),
        )

    # 从大到小选预算内最大的方形边长
    chosen = next(s for s in range(upper, 0, -1) if side_peaks[s] <= budget)
    grid = ((ho + chosen - 1) // chosen, (wo + chosen - 1) // chosen)
    return TilePlan(
        feasible=True, tile=chosen, budget=budget,
        minimum_budget=minimum_budget,
        peak_input_elems=side_peaks[chosen],
        tile_grid=grid, per_side_peaks=per_side, output_size=(ho, wo),
    )
