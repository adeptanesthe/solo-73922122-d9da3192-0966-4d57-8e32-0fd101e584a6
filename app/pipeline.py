"""推理流水线：选型 -> 分块推理 -> 整图基准 -> 指标对比。"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict

import numpy as np

from .conv import (
    TiledRunner,
    run_full_baseline,
    validate_weights,
)
from .planner import TilePlan, plan_tiles

ERROR_TOL = 1e-8


@dataclass
class InferResult:
    feasible: bool
    plan: dict
    metrics: dict | None
    error: str | None = None


def run_pipeline(image: np.ndarray, weights: dict, budget: int) -> InferResult:
    w1, b1, w2, b2 = validate_weights(weights)
    h, w = image.shape
    plan: TilePlan = plan_tiles(h, w, budget)

    plan_dict = asdict(plan)
    if not plan.feasible:
        # 单像素块仍超预算：只报最低预算，不执行推理
        return InferResult(feasible=False, plan=plan_dict, metrics=None)

    # ---- 分块推理（float64）----
    runner = TiledRunner(image, w1, b1, w2, b2, plan.tile)
    t0 = time.perf_counter()
    out_tiled = runner.run()
    tiled_ms = (time.perf_counter() - t0) * 1000.0

    # 实跑守住预算的硬性断言（规划与实跑共用同一窗口公式）
    if runner.peak_input_elems > budget:
        raise RuntimeError(
            f"实跑峰值 {runner.peak_input_elems} 超过预算 {budget}")
    if runner.peak_input_elems != plan.peak_input_elems:
        raise RuntimeError(
            f"实跑峰值 {runner.peak_input_elems} 与规划峰值 "
            f"{plan.peak_input_elems} 不一致")

    # ---- 独立整图基准（float64）----
    t0 = time.perf_counter()
    out_full = run_full_baseline(image, w1, b1, w2, b2)
    full_ms = (time.perf_counter() - t0) * 1000.0

    diff = np.abs(out_tiled - out_full)
    max_err = float(diff.max())
    idx = np.unravel_index(int(diff.argmax()), diff.shape)
    # 坐标按 (通道, 行, 列) 报告，行列从 0 起
    max_err_coord = {"channel": int(idx[0]), "row": int(idx[1]),
                     "col": int(idx[2])}
    passed = max_err <= ERROR_TOL

    metrics = {
        "tiled_ms": round(tiled_ms, 3),
        "full_ms": round(full_ms, 3),
        "runtime_peak_input_elems": runner.peak_input_elems,
        "planned_peak_input_elems": plan.peak_input_elems,
        "peak_layer1_elems_reference": runner.peak_layer1_elems,
        "max_abs_error": max_err,
        "max_abs_error_coord": max_err_coord,
        "tolerance": ERROR_TOL,
        "passed": passed,
        "output_shape": list(out_tiled.shape),
        "tiled_output_sample": out_tiled.ravel()[:8].tolist(),
        "full_output_sample": out_full.ravel()[:8].tolist(),
    }
    return InferResult(feasible=True, plan=plan_dict, metrics=metrics)
