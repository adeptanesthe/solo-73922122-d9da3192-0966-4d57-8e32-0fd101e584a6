"""核心功能自测：窗口公式、选型、预算约束、两套实现一致性、HTTP 接口。"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
from PIL import Image

from app.conv import (
    L1_OUT_CH,
    TiledRunner,
    layer1_out_size,
    required_input_window,
    run_full_baseline,
    validate_weights,
)
from app.io_utils import load_grayscale_png, load_weights_json
from app.pipeline import ERROR_TOL, run_pipeline
from app.planner import plan_tiles


# --------------------------------------------------------------------- helpers
def make_weights(rng: np.random.Generator):
    w1 = rng.normal(0, 0.2, (4, 1, 3, 3))
    b1 = rng.normal(0, 0.1, 4)
    w2 = rng.normal(0, 0.2, (1, 4, 3, 3))
    b2 = rng.normal(0, 0.1, 1)
    return {"layer1": {"weight": w1.tolist(), "bias": b1.tolist()},
            "layer2": {"weight": w2.tolist(), "bias": b2.tolist()}}


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8), mode="L").save(buf, format="PNG")
    return buf.getvalue()


# ----------------------------------------------------------- 1. 感受野窗口公式
class TestWindow:
    def test_interior_single_pixel_11x11(self):
        # 内部单输出像素：感受野在输入上是 11x11（两层 3x3、dilation2、stride2）
        # 8x8 图，输出 4x4；内部输出 (1,1) 对应输入行 [-1,9]，裁到图内 [0,8)
        r = required_input_window(1, 2, 1, 2, 8, 8)
        assert r == (0, 8, 0, 8)

    def test_corner_single_pixel_clipped(self):
        # 左上角输出 (0,0)：输入行 [-5,6) -> 裁到 [0,6)
        r = required_input_window(0, 1, 0, 1, 8, 8)
        assert r == (0, 6, 0, 6)

    def test_tile_window_contains_each_member_pixel_window(self):
        # 块窗口必须覆盖块内每个像素各自的窗口
        h = w = 64
        r0, r1, c0, c1 = 3, 7, 5, 9
        ir0, ir1, ic0, ic1 = required_input_window(r0, r1, c0, c1, h, w)
        for r in range(r0, r1):
            for c in range(c0, c1):
                a, b, d, e = required_input_window(r, r + 1, c, c + 1, h, w)
                assert ir0 <= a and ir1 >= b and ic0 <= d and ic1 >= e

    def test_minimality(self):
        # 最小性：窗口上下左右边界至少由块内某个像素的窗口触及（内部块）
        h = w = 64
        r0, r1, c0, c1 = 10, 14, 10, 14
        ir0, ir1, ic0, ic1 = required_input_window(r0, r1, c0, c1, h, w)
        top = required_input_window(r0, r0 + 1, c0, c0 + 1, h, w)
        bot = required_input_window(r1 - 1, r1, c0, c0 + 1, h, w)
        assert ir0 == top[0] and ir1 == bot[1]


# ----------------------------------------------------------- 2. 选型与预算
class TestPlanner:
    def test_largest_feasible_side_selected(self):
        # 64x64 -> 输出 32x32；所有分块都是边缘块（窗口被图界裁短）。
        # s=32 单块窗口即整图 64x64=4096
        plan = plan_tiles(64, 64, budget=10_000)
        assert plan.feasible
        assert plan.tile == 32
        assert plan.peak_input_elems == 64 * 64

    def test_budget_picks_smaller_side(self):
        # s=8 角块窗口 21x21=441，边缘非角块（如 8x16）窗口 21x37=777；
        # s=7 对应边缘块峰值 23x23=529，故预算 600 选 7
        plan = plan_tiles(64, 64, budget=600)
        assert plan.feasible and plan.tile == 7
        assert plan.peak_input_elems == 529

    def test_side_capped_at_64(self):
        plan = plan_tiles(512, 512, budget=10 ** 9)
        assert plan.feasible and plan.tile == 64
        # 内部 64x64 块窗口为 [2r0-5, 2(r1-1)+6) = 137x137，即全网格峰值
        assert plan.peak_input_elems == 137 * 137

    def test_monotonic_peak_with_side(self):
        plan = plan_tiles(512, 512, budget=10 ** 9)
        peaks = dict(plan.per_side_peaks)
        vals = [peaks[s] for s in sorted(peaks)]
        assert all(b >= a for a, b in zip(vals, vals[1:]))

    def test_infeasible_reports_minimum_only(self):
        # 512 图 s=1 内部窗口 11x11 = 121
        plan = plan_tiles(512, 512, budget=120)
        assert not plan.feasible and plan.tile is None
        assert plan.minimum_budget == 121
        assert plan.peak_input_elems == 121

    def test_small_image_minimum_budget(self):
        # 8x8 图 s=1：角块窗口 6x6=36，中心块窗口 8x8=64（边界已裁短）
        plan = plan_tiles(8, 8, budget=10)
        assert not plan.feasible and plan.minimum_budget == 64

    def test_fast_peak_matches_full_enumeration(self):
        # 快速峰值算法（只查角块+内部块）必须与全网格枚举完全一致
        from app.planner import _peak_for_side
        from app.conv import required_input_window

        def brute(h, w, s):
            ho, wo = layer1_out_size(h, w)
            peak = 0
            for r0 in range(0, ho, s):
                for c0 in range(0, wo, s):
                    a, b, d, e = required_input_window(
                        r0, min(r0 + s, ho), c0, min(c0 + s, wo), h, w)
                    peak = max(peak, (b - a) * (e - d))
            return peak

        for h, w in [(8, 8), (64, 64), (100, 70), (512, 512), (513, 300)]:
            ho, wo = layer1_out_size(h, w)
            for s in (1, 2, 3, 7, 8, 32, 64):
                if s <= min(ho, wo):
                    assert _peak_for_side(h, w, ho, wo, s) == brute(h, w, s)

    def test_runtime_never_exceeds_budget(self, rng=np.random.default_rng(1)):
        img = rng.random((512, 512))
        wts = make_weights(rng)
        w1, b1, w2, b2 = validate_weights(wts)
        for budget in (200, 500, 5000, 30000):
            plan = plan_tiles(512, 512, budget)
            assert plan.feasible
            r = TiledRunner(img, w1, b1, w2, b2, plan.tile)
            r.run()
            assert r.peak_input_elems <= budget
            assert r.peak_input_elems == plan.peak_input_elems


# ------------------------------------------------- 3. 分块 vs 整图基准（float64）
class TestEquivalence:
    @pytest.mark.parametrize("size,tile", [(8, 1), (8, 4), (64, 1), (64, 8),
                                           (64, 32), (100, 7), (128, 1),
                                           (512, 16), (512, 64)])
    def test_matches_baseline(self, size, tile):
        rng = np.random.default_rng(seed=size * 100 + tile)
        img = rng.random((size, size))
        wts = make_weights(rng)
        w1, b1, w2, b2 = validate_weights(wts)
        a = TiledRunner(img, w1, b1, w2, b2, tile).run()
        b = run_full_baseline(img, w1, b1, w2, b2)
        assert a.shape == b.shape == (1, *layer1_out_size(size, size))
        assert a.dtype == np.float64 and b.dtype == np.float64
        err = np.abs(a - b).max()
        assert err <= ERROR_TOL, (size, tile, err)

    def test_non_square_image(self):
        rng = np.random.default_rng(7)
        img = rng.random((40, 96))
        w1, b1, w2, b2 = validate_weights(make_weights(rng))
        a = TiledRunner(img, w1, b1, w2, b2, 8).run()
        b = run_full_baseline(img, w1, b1, w2, b2)
        assert np.abs(a - b).max() <= ERROR_TOL

    def test_padding_zeros_not_garbage(self):
        # 全零图：layer1 处处输出 ReLU(b1)。在远离图界的内部点，layer2 的
        # 9 个 tap 全部落在图内，输出为确定常数；边缘点受零填充影响另算。
        rng = np.random.default_rng(3)
        img = np.zeros((32, 32))
        w1, b1, w2, b2 = validate_weights(make_weights(rng))
        out = run_full_baseline(img, w1, b1, w2, b2)
        relu_b1 = np.maximum(b1, 0.0)
        expected = float((w2[0] * relu_b1[None, :, None, None]).sum() + b2[0])
        # 输出 16x16，layer2 pad2 触及 ±2，取中心 4x4 必为纯图内 tap
        assert np.allclose(out[0, 6:10, 6:10], expected, atol=1e-12)
        # 角点必然包含零填充贡献，不应等于内部常数
        assert not np.isclose(out[0, 0, 0], expected, atol=1e-12)


# ------------------------------------------------- 4. 流水线与输入校验
class TestPipelineAndIO:
    def test_end_to_end_pass(self):
        rng = np.random.default_rng(11)
        img_u8 = (rng.random((64, 64)) * 255).astype(np.uint8)
        img = load_grayscale_png(png_bytes(img_u8))
        res = run_pipeline(img, make_weights(rng), budget=1000)
        assert res.feasible and res.metrics["passed"]
        assert res.metrics["max_abs_error"] <= 1e-8
        coord = res.metrics["max_abs_error_coord"]
        assert set(coord) == {"channel", "row", "col"}

    def test_infeasible_pipeline_has_no_metrics(self):
        rng = np.random.default_rng(12)
        res = run_pipeline(rng.random((512, 512)), make_weights(rng), budget=1)
        assert not res.feasible and res.metrics is None
        assert res.plan["minimum_budget"] == 121

    def test_failure_when_error_exceeds_tolerance(self, monkeypatch):
        # 构造 2e-8 的人为偏差，验证 >1e-8 判失败且坐标被正确报告
        import app.pipeline as pl
        original = pl.run_full_baseline

        def fake_baseline(image, w1, b1, w2, b2):
            out = original(image, w1, b1, w2, b2).copy()
            out[0, 1, 2] += 2e-8
            return out

        monkeypatch.setattr(pl, "run_full_baseline", fake_baseline)
        rng = np.random.default_rng(13)
        res = run_pipeline(rng.random((32, 32)), make_weights(rng), budget=10_000)
        m = res.metrics
        assert m["passed"] is False
        assert m["max_abs_error"] > 1e-8
        assert m["max_abs_error_coord"] == {"channel": 0, "row": 1, "col": 2}

    def test_reject_rgb_and_oversize(self):
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="灰度"):
            load_grayscale_png(self._png(rgb, "RGB"))
        big = np.zeros((513, 10), dtype=np.uint8)
        with pytest.raises(ValueError, match="512"):
            load_grayscale_png(png_bytes(big))

    @staticmethod
    def _png(arr, mode):
        buf = io.BytesIO()
        Image.fromarray(arr, mode=mode).save(buf, format="PNG")
        return buf.getvalue()

    def test_reject_bad_json_and_shapes(self):
        with pytest.raises(ValueError):
            load_weights_json(b"not json")
        rng = np.random.default_rng(4)
        bad = make_weights(rng)
        bad["layer1"]["weight"] = rng.normal(0, 1, (2, 1, 3, 3)).tolist()
        with pytest.raises(ValueError, match="4,1,3,3"):
            validate_weights(bad)

    def test_weights_roundtrip_file(self):
        rng = np.random.default_rng(5)
        raw = json.dumps(make_weights(rng)).encode()
        parsed = load_weights_json(raw)
        w1, b1, w2, b2 = validate_weights(parsed)
        assert w1.shape == (4, 1, 3, 3) and w2.shape == (1, 4, 3, 3)
        assert all(x.dtype == np.float64 for x in (w1, b1, w2, b2))


# ----------------------------------------------------------------- 5. HTTP 接口
class TestHTTP:
    def test_index_health_and_infer(self):
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)
        assert client.get("/").status_code == 200
        assert client.get("/health").json() == {"status": "ok"}

        rng = np.random.default_rng(21)
        img = (rng.random((64, 64)) * 255).astype(np.uint8)
        wts = json.dumps(make_weights(rng))
        r = client.post(
            "/infer",
            files={"image": ("a.png", png_bytes(img), "image/png"),
                   "weights": ("w.json", wts, "application/json")},
            data={"budget": "1000", "as_json": "1"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["feasible"] and body["metrics"]["passed"]

    def test_http_html_success_and_error_pages(self):
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)
        rng = np.random.default_rng(24)
        img = (rng.random((64, 64)) * 255).astype(np.uint8)
        r = client.post(
            "/infer",
            files={"image": ("a.png", png_bytes(img), "image/png"),
                   "weights": ("w.json", json.dumps(make_weights(rng)),
                               "application/json")},
            data={"budget": "1000"})
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "选中的方形分块边长" in r.text and "最大绝对误差" in r.text

    def test_http_infeasible(self):
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)
        rng = np.random.default_rng(22)
        img = (rng.random((512, 512)) * 255).astype(np.uint8)
        r = client.post(
            "/infer",
            files={"image": ("a.png", png_bytes(img), "image/png"),
                   "weights": ("w.json", json.dumps(make_weights(rng)),
                               "application/json")},
            data={"budget": "50", "as_json": "1"})
        assert r.status_code == 200
        body = r.json()
        assert body["feasible"] is False
        assert body["plan"]["minimum_budget"] == 121
        assert body["metrics"] is None

    def test_http_bad_request(self):
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)
        rng = np.random.default_rng(23)
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        r = client.post(
            "/infer",
            files={"image": ("a.png", TestPipelineAndIO._png(img, "RGB"),
                             "image/png"),
                   "weights": ("w.json", json.dumps(make_weights(rng)),
                               "application/json")},
            data={"budget": "1000"})
        assert r.status_code == 400 and "灰度" in r.text
