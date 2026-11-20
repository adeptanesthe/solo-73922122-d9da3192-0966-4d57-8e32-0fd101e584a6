"""FastAPI 网页：

1. /            分块尺寸选型（float64，原有功能）；
2. /quant       整数量化推理 vs 整图浮点推理的裂缝判定对照（新增）。
"""

from __future__ import annotations

import base64
import html
import math

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from .io_utils import (
    load_grayscale_png,
    load_grayscale_png_u8,
    load_weights_json,
)
from .pipeline import run_pipeline
from .quant import QuantError, run_quant_comparison
from .visualize import render_comparison_png

app = FastAPI(title="路面裂缝两层卷积边缘演示", version="2.0.0")

# ==================================================================
# 通用样式
# ==================================================================
STYLE = """
 body{font-family:system-ui,"PingFang SC","Microsoft YaHei",sans-serif;
   margin:2rem auto;max-width:1100px;padding:0 1rem;color:#1f2933}
 h1{font-size:1.4rem} h2{font-size:1.1rem;margin-top:1.6rem}
 nav{margin:.5rem 0 1.2rem;font-size:.95rem}
 nav a{margin-right:1.2rem}
 form{border:1px solid #cbd2d9;border-radius:8px;padding:1rem 1.2rem;background:#f8fafc}
 label{display:block;margin:.7rem 0 .25rem;font-weight:600}
 input{margin:.2rem 0} button{margin-top:1rem;padding:.5rem 1.4rem;
   background:#1d4ed8;color:#fff;border:0;border-radius:6px;cursor:pointer}
 table{border-collapse:collapse;margin:.6rem 0;font-size:.92rem}
 th,td{border:1px solid #cbd2d9;padding:.3rem .7rem;text-align:right}
 th{background:#eef2f7} .ok{color:#15803d;font-weight:700}
 .bad{color:#b91c1c;font-weight:700} .note{color:#52606d;font-size:.85rem}
 .err{background:#fef2f2;border:1px solid #fca5a5;color:#991b1b;
   padding:.7rem 1rem;border-radius:6px;white-space:pre-wrap}
 code{background:#eef2f7;padding:.1rem .35rem;border-radius:4px}
 .legend span{display:inline-block;margin-right:1.1rem;font-size:.9rem}
 .sw{display:inline-block;width:.9rem;height:.9rem;border:1px solid #94a3b8;
   vertical-align:middle;margin-right:.3rem}
 img{max-width:100%;border:1px solid #cbd2d9;border-radius:6px;background:#fff}
"""

NAV = ('<nav><a href="/">分块尺寸选型（float64）</a>'
       '<a href="/quant">量化对照（整数 vs 浮点）</a></nav>')

# ==================================================================
# 页面 1：分块选型（原有）
# ==================================================================
PAGE = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>两层卷积分块选型演示</title>
<style>{STYLE}</style></head><body>
{NAV}
<h1>路面裂缝两层卷积 · 边缘分块尺寸选型（NumPy float64）</h1>
<p class="note">结构固定：Conv 1→4，3×3，stride 2，pad 1，ReLU；Conv 4→1，3×3，
stride 1，dilation 2，pad 2。图像为每边 ≤512 的灰度 PNG。
预算按覆盖每个输出块所需的<b>图内激活最小矩形窗口</b>（含通道）计算，
边缘块裁短；分块边长从 1～64 中取预算内最大者。</p>
<form action="/infer" method="post" enctype="multipart/form-data">
 <label>灰度 PNG 图像（mode=L，边长 ≤512）</label>
 <input type="file" name="image" accept="image/png" required>
 <label>两层权重/偏置 JSON</label>
 <input type="file" name="weights" accept=".json,application/json" required>
 <label>第一层输入激活预算（可同时保留的激活元素数，正整数）</label>
 <input type="number" name="budget" min="1" step="1" value="20000" required>
 <label class="note"><input type="checkbox" name="as_json" value="1">
   以 JSON 返回（便于脚本调用）</label>
 <button type="submit">运行分块推理与整图基准对比</button>
</form>
__RESULT__
</body></html>"""


def _kv_table(rows: list[tuple[str, str]]) -> str:
    body = "".join(f"<tr><th style='text-align:left'>{html.escape(k)}</th>"
                   f"<td>{v}</td></tr>" for k, v in rows)
    return f"<table>{body}</table>"


def _render_error(msg: str) -> str:
    return f"<div class='err'>{html.escape(msg)}</div>"


def _render_result(payload: dict) -> str:
    plan = payload["plan"]
    parts = ["<h2>选型结果</h2>"]
    if not payload["feasible"]:
        parts.append(_kv_table([
            ("是否可行", "<span class='bad'>否：单像素块仍超预算，未执行推理</span>"),
            ("给定预算（元素数）", str(plan["budget"])),
            ("最低所需预算（s=1 峰值）", f"<b>{plan['minimum_budget']}</b>"),
            ("输出尺寸 (行,列)", str(tuple(plan["output_size"]))),
        ]))
    else:
        m = payload["metrics"]
        verdict = ("通过（≤ 1e-8）" if m["passed"]
                   else "失败（&gt; 1e-8）")
        vcls = "ok" if m["passed"] else "bad"
        parts.append(_kv_table([
            ("选中的方形分块边长 s", f"<b>{plan['tile']}</b>（1～64 内预算允许的最大值）"),
            ("分块网格 (行块数, 列块数)", str(tuple(plan["tile_grid"]))),
            ("给定预算（元素数）", str(plan["budget"])),
            ("规划峰值图内激活元素数", str(plan["peak_input_elems"])),
            ("实跑峰值图内激活元素数", str(m["runtime_peak_input_elems"])),
            ("参考：层1中间张量峰值元素数（4 通道，不计入预算）",
             str(m["peak_layer1_elems_reference"])),
            ("最低所需预算（s=1 峰值）", str(plan["minimum_budget"])),
            ("输出尺寸 (通道,行,列)", str(tuple(m["output_shape"]))),
            ("分块推理耗时 (ms)", str(m["tiled_ms"])),
            ("整图基准耗时 (ms)", str(m["full_ms"])),
            ("最大绝对误差", f"{m['max_abs_error']:.3e}"),
            ("最大误差坐标 (通道,行,列)", str(tuple(m["max_abs_error_coord"].values()))),
            ("判定（容差 1e-8）", f"<span class='{vcls}'>{verdict}</span>"),
        ]))
        parts.append("<p class='note'>分块输出前 8 个值："
                     + html.escape(", ".join(f"{x:.6g}" for x in m["tiled_output_sample"]))
                     + "<br>整图输出前 8 个值："
                     + html.escape(", ".join(f"{x:.6g}" for x in m["full_output_sample"]))
                     + "</p>")

    peaks = plan["per_side_peaks"]
    rows = "".join(
        f"<tr><td>{s}</td><td>{p}</td>"
        f"<td>{'✔ 预算内' if p <= plan['budget'] else '— 超预算'}</td></tr>"
        for s, p in peaks)
    parts.append("<h2>各候选边长的峰值用量</h2><table><tr><th>边长 s</th>"
                 "<th>峰值图内激活元素数</th><th>对给定预算</th></tr>"
                 f"{rows}</table>")
    parts.append("<p class='note'>窗口公式：输出块 (r0:r1, c0:c1) 对应图内行窗口 "
                 "[2·r0−5, 2·(r1−1)+6)、列窗口同理，再裁到图界；"
                 "元素数 = 高 × 宽 × 输入通道数（本网络输入为 1 通道）。</p>")
    return "".join(parts)


# ==================================================================
# 页面 2：量化对照（新增）
# ==================================================================
PAGE_QUANT = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>整数量化 vs 浮点推理 · 裂缝判定对照</title>
<style>{STYLE}</style></head><body>
{NAV}
<h1>路面裂缝两层卷积 · 整数量化推理 vs 整图浮点推理对照</h1>
<p class="note">结构固定：Conv 1→4，3×3，stride 2，dilation 1，pad 1，ReLU；
Conv 4→1，3×3，stride 1，dilation 2，pad 2。上传每边 ≤128 的灰度 PNG 与
原权重 JSON，并填写三个<b>正量化尺度</b>：两层权重尺度、首层 ReLU 输出尺度。
首层输入直接取灰度字节（尺度 1/255）；量化取最近整数、遇半取偶；权重饱和到
int8；偏置按该层输入尺度 × 权重尺度量化并要求落在 int32 内；首层 ReLU 后按
给定尺度重定标并饱和到 uint8，次层直接读取这些字节。两路均以分数 ≥ 0 判裂缝，
在输出网格上分别标出新增（红）与消失（蓝）的裂缝像素。</p>
<form action="/quant/infer" method="post" enctype="multipart/form-data">
 <label>灰度 PNG 图像（mode=L，边长 ≤128）</label>
 <input type="file" name="image" accept="image/png" required>
 <label>原权重/偏置 JSON（与分块页同一格式）</label>
 <input type="file" name="weights" accept=".json,application/json" required>
 <label>layer1 权重正量化尺度 s_w1</label>
 <input type="text" name="scale_w1" value="0.05" required>
 <label>layer2 权重正量化尺度 s_w2</label>
 <input type="text" name="scale_w2" value="0.05" required>
 <label>首层 ReLU 输出正量化尺度 s_a1（次层输入尺度）</label>
 <input type="text" name="scale_a1" value="0.02" required>
 <label class="note"><input type="checkbox" name="as_json" value="1">
   以 JSON 返回（便于脚本调用）</label>
 <button type="submit">运行整数 / 浮点对照</button>
</form>
__RESULT__
</body></html>"""

COORD_PREVIEW = 20  # 页面上坐标清单最多展示多少条


def _parse_scale(raw: str, label: str) -> float:
    try:
        s = float(raw)
    except (TypeError, ValueError):
        raise QuantError(f"{label}必须是正的有限数，收到 {raw!r}")
    if not math.isfinite(s) or s <= 0:
        raise QuantError(f"{label}必须是正的有限数，收到 {raw!r}")
    return s


def _coord_preview(coords: list[list[int]]) -> str:
    head = coords[:COORD_PREVIEW]
    text = ", ".join(f"({r},{c})" for r, c in head)
    if len(coords) > COORD_PREVIEW:
        text += f" …… 仅显示前 {COORD_PREVIEW} 条，共 {len(coords)} 条（JSON 含全量）"
    return html.escape(text or "（无）")


def _render_quant_result(payload: dict, png_b64: str) -> str:
    c = payload["counts"]
    sc = payload["scales"]
    sat = payload["weight_saturated"]
    parts = ["<h2>对照结果</h2>", _kv_table([
        ("输出网格尺寸 (通道,行,列)", str(tuple(payload["output_shape"]))),
        ("输入尺度（灰度字节，固定）", f"{sc['input']:.8g}"),
        ("layer1 权重尺度 s_w1", f"{sc['weight1']:.8g}"),
        ("layer2 权重尺度 s_w2", f"{sc['weight2']:.8g}"),
        ("首层 ReLU 输出尺度 s_a1", f"{sc['activation1']:.8g}"),
        ("派生 layer1 偏置尺度 s_x·s_w1", f"{sc['bias1']:.8g}"),
        ("派生 layer2 偏置尺度 s_a1·s_w2", f"{sc['bias2']:.8g}"),
        ("layer1 权重 int8 饱和元素数", str(sat["layer1"])),
        ("layer2 权重 int8 饱和元素数", str(sat["layer2"])),
        ("浮点路裂缝像素数（分数 ≥ 0）", str(c["float_crack"])),
        ("整数路裂缝像素数（累加器 ≥ 0）", str(c["quant_crack"])),
        ("两路一致裂缝像素数", str(c["agreed_crack"])),
        ("量化新增裂缝像素数", f"<span class='bad'>{c['new_crack']}</span>"),
        ("量化消失裂缝像素数", f"<span class='bad'>{c['gone_crack']}</span>"),
    ])]
    parts.append(
        "<p class='note'>整数路输出累加器前 8 个值："
        + html.escape(", ".join(str(x) for x in payload["quant_acc_sample"]))
        + "<br>浮点路输出分数前 8 个值："
        + html.escape(", ".join(f"{x:.6g}" for x in payload["float_output_sample"]))
        + "</p>")
    parts.append(
        "<div class='legend'><span><i class='sw' style='background:#15803d'></i>"
        "两路都判裂缝</span><span><i class='sw' style='background:#dc2626'></i>"
        "量化新增</span><span><i class='sw' style='background:#2563eb'></i>"
        "量化消失</span><span><i class='sw' style='background:#e2e6eb'></i>"
        "一致非裂缝</span></div>")
    parts.append("<h2>四联对照图（原图 / 浮点判定 / 整数判定 / 叠加差异）</h2>")
    parts.append(f"<img alt='量化对照四联图' src='data:image/png;base64,{png_b64}'>")
    parts.append("<h2>差异坐标（输出网格，行/列从 0 起）</h2>")
    parts.append("<p><b>量化新增：</b>" + _coord_preview(payload["new_coords"]) + "</p>")
    parts.append("<p><b>量化消失：</b>" + _coord_preview(payload["gone_coords"]) + "</p>")
    return "".join(parts)


# ==================================================================
# 路由
# ==================================================================
@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE.replace("__RESULT__", "")


@app.get("/quant", response_class=HTMLResponse)
def quant_page() -> str:
    return PAGE_QUANT.replace("__RESULT__", "")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/infer", response_model=None)
async def infer(image: UploadFile = File(...),
                weights: UploadFile = File(...),
                budget: int = Form(...),
                as_json: str = Form("")) -> HTMLResponse | JSONResponse:
    try:
        if budget < 1:
            raise ValueError("预算必须是正整数")
        img = load_grayscale_png(await image.read())
        wts = load_weights_json(await weights.read())
        result = run_pipeline(img, wts, budget)
        payload = {"feasible": result.feasible, "plan": result.plan,
                   "metrics": result.metrics}
    except (ValueError, RuntimeError) as exc:
        if as_json == "1":
            return JSONResponse({"error": str(exc)}, status_code=400)
        # 出错时结果区只渲染原因，不携带任何旧图
        return HTMLResponse(PAGE.replace("__RESULT__", _render_error(str(exc))),
                            status_code=400)

    if as_json == "1":
        return JSONResponse(payload)
    return HTMLResponse(PAGE.replace("__RESULT__", _render_result(payload)))


@app.post("/quant/infer", response_model=None)
async def quant_infer(image: UploadFile = File(...),
                      weights: UploadFile = File(...),
                      scale_w1: str = Form(...),
                      scale_w2: str = Form(...),
                      scale_a1: str = Form(...),
                      as_json: str = Form("")) -> HTMLResponse | JSONResponse:
    try:
        sw1 = _parse_scale(scale_w1, "layer1 权重尺度")
        sw2 = _parse_scale(scale_w2, "layer2 权重尺度")
        sa1 = _parse_scale(scale_a1, "首层 ReLU 输出尺度")
        img_u8 = load_grayscale_png_u8(await image.read())
        wts = load_weights_json(await weights.read())
        result = run_quant_comparison(img_u8, wts, sw1, sw2, sa1)
        png_bytes = render_comparison_png(img_u8, result)
        payload = {
            "output_shape": result.output_shape,
            "scales": result.scales,
            "weight_saturated": result.weight_saturated,
            "counts": result.counts,
            "new_coords": result.new_coords,
            "gone_coords": result.gone_coords,
            "float_crack_mask": result.float_crack.astype(int).tolist(),
            "quant_crack_mask": result.quant_crack.astype(int).tolist(),
            "new_mask": result.new_mask.astype(int).tolist(),
            "gone_mask": result.gone_mask.astype(int).tolist(),
            "quant_activation_shape": list(result.quant_activation.shape),
            "quant_acc_sample": result.quant_acc.ravel()[:8].tolist(),
            "float_output_sample": result.float_output.ravel()[:8].tolist(),
        }
    except (ValueError, RuntimeError) as exc:
        # 硬性失败：只显示原因，清掉旧图（结果区不含任何图像元素）
        if as_json == "1":
            return JSONResponse({"error": str(exc)}, status_code=400)
        return HTMLResponse(
            PAGE_QUANT.replace("__RESULT__", _render_error(str(exc))),
            status_code=400)

    if as_json == "1":
        return JSONResponse(payload)
    png_b64 = base64.b64encode(png_bytes).decode("ascii")
    return HTMLResponse(
        PAGE_QUANT.replace("__RESULT__", _render_quant_result(payload, png_b64)))
