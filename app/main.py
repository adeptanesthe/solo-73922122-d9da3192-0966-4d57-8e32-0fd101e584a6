"""FastAPI 网页：上传灰度 PNG + 两层权重 JSON + 激活预算，返回分块选型与对比指标。"""

from __future__ import annotations

import html

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from .io_utils import load_grayscale_png, load_weights_json
from .pipeline import run_pipeline

app = FastAPI(title="路面裂缝两层卷积边缘分块演示", version="1.0.0")

PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>两层卷积分块选型演示</title>
<style>
 body{font-family:system-ui,"PingFang SC","Microsoft YaHei",sans-serif;
   margin:2rem auto;max-width:980px;padding:0 1rem;color:#1f2933}
 h1{font-size:1.4rem} h2{font-size:1.1rem;margin-top:1.6rem}
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
</style></head><body>
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


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE.replace("__RESULT__", "")


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
        return HTMLResponse(PAGE.replace("__RESULT__", _render_error(str(exc))),
                            status_code=400)

    if as_json == "1":
        return JSONResponse(payload)
    return HTMLResponse(PAGE.replace("__RESULT__", _render_result(payload)))
