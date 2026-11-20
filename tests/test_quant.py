"""量化对照功能自测：舍入原语、量化边界、整数推理正确性、硬性失败与 HTTP。"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
from PIL import Image

from app.conv import validate_weights
from app.io_utils import load_grayscale_png_u8, load_weights_json
from app.main import app
from app.quant import (
    INT32_MAX,
    INT32_MIN,
    QuantError,
    round_half_even,
    run_quant_comparison,
)
from app.visualize import render_comparison_png


# --------------------------------------------------------------------- helpers
def make_weights(rng: np.random.Generator, scale: float = 0.2):
    w1 = rng.normal(0, scale, (4, 1, 3, 3))
    b1 = rng.normal(0, 0.05, 4)
    w2 = rng.normal(0, scale, (1, 4, 3, 3))
    b2 = rng.normal(0, 0.05, 1)
    return {"layer1": {"weight": w1.tolist(), "bias": b1.tolist()},
            "layer2": {"weight": w2.tolist(), "bias": b2.tolist()}}


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8), mode="L").save(buf, format="PNG")
    return buf.getvalue()


def run_default(img_u8, rng_seed=1, sw1=0.05, sw2=0.05, sa1=0.02):
    rng = np.random.default_rng(rng_seed)
    return run_quant_comparison(img_u8, make_weights(rng), sw1, sw2, sa1)


# ----------------------------------------------------------- 1. 遇半取偶舍入
class TestRoundHalfEven:
    def test_basic_ties_even(self):
        out = round_half_even(np.array([0.5, 1.5, 2.5, 3.5, -0.5,
                                        -1.5, -2.5, -3.5]))
        assert out.tolist() == [0, 2, 2, 4, 0, -2, -2, -4]

    def test_non_ties_nearest(self):
        out = round_half_even(np.array([0.49, 0.51, -0.49, -0.51,
                                        2.4, 2.6, -2.4, -2.6]))
        assert out.tolist() == [0, 1, 0, -1, 2, 3, -2, -3]

    def test_integers_unchanged(self):
        x = np.array([-3.0, -0.0, 0.0, 7.0, 127.0])
        assert round_half_even(x).tolist() == [-3, 0, 0, 7, 127]

    def test_matches_numpy_banker(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 5, 1000)
        # 显式实现须与 np.round（同为遇半取偶）一致
        assert np.array_equal(round_half_even(x), np.round(x))

    def test_non_finite_raises(self):
        with pytest.raises(QuantError, match="非有限"):
            round_half_even(np.array([1.0, np.nan]))
        with pytest.raises(QuantError, match="非有限"):
            round_half_even(np.array([np.inf]))


# ----------------------------------------------------- 2. 量化规则与边界
class TestQuantization:
    def test_weight_saturates_to_int8(self):
        rng = np.random.default_rng(2)
        wts = make_weights(rng)
        # 极大权重 + 小尺度，必然触发饱和；饱和后值严格在 int8 内
        wts["layer1"]["weight"][0][0][0][0] = 100.0
        res = run_quant_comparison(
            (rng.random((32, 32)) * 255).astype(np.uint8),
            wts, 1e-6, 0.05, 0.02)
        assert res.weight_saturated["layer1"] >= 1
        # 结果结构完整
        assert res.quant_acc.dtype == np.int64
        assert res.quant_activation.dtype == np.uint8
        assert res.quant_activation.min() >= 0
        assert res.quant_activation.max() <= 255

    def test_activation_saturates_to_uint8(self):
        # 大偏置经 ReLU 后产生巨大激活，重定标后必须被夹到 255
        rng = np.random.default_rng(3)
        wts = make_weights(rng)
        wts["layer1"]["bias"] = [50.0, 0.0, 0.0, 0.0]
        res = run_quant_comparison(
            np.zeros((16, 16), dtype=np.uint8), wts, 0.05, 0.05, 0.02)
        assert int(res.quant_activation[0].max()) == 255

    def test_bias_quantization_scale_derivation(self):
        # 手工构造：b1 = 3 * (1/255 * 0.05)，量化偏置须恰好为 3
        rng = np.random.default_rng(4)
        wts = make_weights(rng)
        sb1 = (1.0 / 255.0) * 0.05
        wts["layer1"]["bias"] = [3 * sb1, 0.0, 0.0, 0.0]
        img = np.zeros((16, 16), dtype=np.uint8)
        res = run_quant_comparison(img, wts, 0.05, 0.05, 0.02)
        # 全零图上 layer1 累加器（ReLU 前）即量化偏置本身
        from app.quant import quantize_bias
        qb1 = quantize_bias(np.array(wts["layer1"]["bias"]), sb1, "layer1")
        assert qb1.tolist() == [3, 0, 0, 0]
        # ReLU 后重定标字节有限且可跑完全程
        assert res.counts["grid_cells"] > 0

    def test_bias_exceeding_int32_rejected(self):
        rng = np.random.default_rng(5)
        wts = make_weights(rng)
        # 极小偏置尺度使量化偏置爆 int32
        wts["layer1"]["bias"] = [1e20, 0.0, 0.0, 0.0]
        with pytest.raises(QuantError, match="量化偏置超出 int32"):
            run_quant_comparison(np.zeros((8, 8), dtype=np.uint8),
                                 wts, 1e-12, 0.05, 0.02)

    def test_non_positive_scales_rejected(self):
        rng = np.random.default_rng(6)
        wts = make_weights(rng)
        img = np.zeros((8, 8), dtype=np.uint8)
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with pytest.raises(QuantError):
                run_quant_comparison(img, wts, bad, 0.05, 0.02)


# --------------------------------------------- 3. 整数推理的手工核对与确定性
class TestIntegerInference:
    def test_hand_computed_8x8_all_zero(self):
        # 全零输入：layer1 累加器恒为量化偏置；手工核对一个输出格的两层链路
        rng = np.random.default_rng(7)
        wts = make_weights(rng)
        img = np.zeros((8, 8), dtype=np.uint8)
        res = run_quant_comparison(img, wts, 0.05, 0.05, 0.02)
        w1, b1, w2, b2 = validate_weights(wts)

        from app.conv import layer1_out_size
        ho, wo = layer1_out_size(8, 8)
        assert (ho, wo) == (4, 4)

        sb1 = (1.0 / 255.0) * 0.05
        qb1 = round_half_even(b1 / sb1).astype(np.int64)
        # 全零输入下 layer1 每个空间位置的激活相同，扩展回 (4,ho,wo) 便于按坐标核对
        qa1_flat = np.clip(round_half_even(np.maximum(qb1, 0) * sb1 / 0.02),
                           0, 255).astype(np.int64)
        qa1 = np.broadcast_to(qa1_flat[:, None, None], (4, ho, wo))
        sb2 = 0.02 * 0.05
        qb2 = int(round_half_even(np.array([b2[0] / sb2]))[0])
        qw2 = np.clip(np.round(w2[0] / 0.05), -128, 127).astype(np.int64)

        # 输出 (0,0)：layer2 pad2 裁掉图外 tap，只统计合法的 layer1 位置
        expected = qb2
        for ic in range(4):
            for kh in range(3):
                for kw in range(3):
                    r = 0 + 2 * (kh - 1)
                    c = 0 + 2 * (kw - 1)
                    if 0 <= r < ho and 0 <= c < wo:
                        expected += qw2[ic, kh, kw] * qa1[ic, r, c]
        assert int(res.quant_acc[0, 0, 0]) == int(expected)

    def test_deterministic_across_calls(self):
        rng = np.random.default_rng(8)
        img = (rng.random((48, 48)) * 255).astype(np.uint8)
        wts = make_weights(rng)
        a = run_quant_comparison(img, wts, 0.05, 0.05, 0.02)
        b = run_quant_comparison(img, wts, 0.05, 0.05, 0.02)
        assert np.array_equal(a.quant_acc, b.quant_acc)
        assert np.array_equal(a.quant_activation, b.quant_activation)
        assert np.array_equal(a.new_mask, b.new_mask)

    def test_output_shape_matches_float_grid(self):
        img = (np.random.default_rng(9).random((64, 64)) * 255).astype(np.uint8)
        res = run_default(img, rng_seed=9)
        assert res.output_shape == [1, 32, 32]
        assert res.float_crack.shape == (32, 32)
        assert res.quant_crack.shape == (32, 32)
        assert res.quant_activation.shape == (4, 32, 32)

    def test_non_square_image(self):
        # 40x96 -> layer1 输出 20x48；两层切片与掩码形状都须自洽
        rng = np.random.default_rng(15)
        img = (rng.random((40, 96)) * 255).astype(np.uint8)
        res = run_default(img, rng_seed=15)
        assert res.output_shape == [1, 20, 48]
        assert res.float_crack.shape == (20, 48)
        assert res.quant_activation.shape == (4, 20, 48)
        assert res.new_mask.shape == res.gone_mask.shape == (20, 48)

    def test_masks_partition_consistency(self):
        rng = np.random.default_rng(10)
        img = (rng.random((64, 64)) * 255).astype(np.uint8)
        res = run_default(img, rng_seed=10)
        f, q = res.float_crack, res.quant_crack
        assert np.array_equal(res.new_mask, q & ~f)
        assert np.array_equal(res.gone_mask, f & ~q)
        # new 与 gone 互斥；并集恰为判定不一致集合
        assert not np.any(res.new_mask & res.gone_mask)
        assert np.array_equal(res.new_mask | res.gone_mask, f != q)
        assert res.counts["new_crack"] == len(res.new_coords)
        assert res.counts["gone_crack"] == len(res.gone_coords)

    def test_same_scales_change_judgements_on_sample(self):
        # 粗尺度（强量化）相对细尺度应产生差异（新增或消失至少其一非零）
        rng = np.random.default_rng(11)
        img = (rng.random((64, 64)) * 255).astype(np.uint8)
        wts = make_weights(rng)
        fine = run_quant_comparison(img, wts, 0.001, 0.001, 0.001)
        coarse = run_quant_comparison(img, wts, 0.2, 0.2, 0.2)
        diff_fine = fine.counts["new_crack"] + fine.counts["gone_crack"]
        diff_coarse = coarse.counts["new_crack"] + coarse.counts["gone_crack"]
        assert diff_coarse >= diff_fine
        assert diff_coarse > 0


# ----------------------------------------------------- 4. int32 硬性失败路径
class TestOverflow:
    def test_layer2_accumulator_overflow_rejected(self, monkeypatch):
        # layer1 零权重 + 小正偏置 -> 累加器仅为个位数偏置（通过检查），
        # 但 ReLU 后产生非零 uint8 激活；layer2 权重全部饱和到 127，
        # 36 个 tap 累加大幅超过人为缩小的 int32 上界。
        rng = np.random.default_rng(12)
        wts = make_weights(rng)
        wts["layer1"]["weight"] = np.zeros((4, 1, 3, 3)).tolist()
        sb1 = (1.0 / 255.0) * 0.001
        wts["layer1"]["bias"] = [5 * sb1] * 4
        wts["layer2"]["weight"] = np.ones((1, 4, 3, 3)).tolist()
        wts["layer2"]["bias"] = [0.0]
        img = np.full((8, 8), 255, dtype=np.uint8)
        import app.quant as qmod
        monkeypatch.setattr(qmod, "INT32_MAX", 1000)
        monkeypatch.setattr(qmod, "INT32_MIN", -1000)
        with pytest.raises(QuantError, match="layer2 带偏置点积超出 int32"):
            run_quant_comparison(img, wts, 0.001, 0.001, 1e-6)

    def test_layer1_accumulator_overflow_rejected(self, monkeypatch):
        rng = np.random.default_rng(13)
        wts = make_weights(rng)
        wts["layer1"]["weight"] = np.ones((4, 1, 3, 3)).tolist()
        wts["layer1"]["bias"] = [0.0] * 4
        img = np.full((8, 8), 255, dtype=np.uint8)
        import app.quant as qmod
        monkeypatch.setattr(qmod, "INT32_MAX", 100)
        monkeypatch.setattr(qmod, "INT32_MIN", -100)
        with pytest.raises(QuantError, match="超出 int32"):
            run_quant_comparison(img, wts, 0.001, 0.05, 0.02)

    def test_bounds_constants_sane(self):
        assert INT32_MIN == -(2**31) and INT32_MAX == 2**31 - 1


# ----------------------------------------------------------- 5. 输入与渲染
class TestIOAndRender:
    def test_u8_loader_limit_128(self):
        ok = np.zeros((128, 128), dtype=np.uint8)
        assert load_grayscale_png_u8(png_bytes(ok)).dtype == np.uint8
        big = np.zeros((129, 64), dtype=np.uint8)
        with pytest.raises(ValueError, match="128"):
            load_grayscale_png_u8(png_bytes(big))

    def test_reject_rgb(self):
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
        with pytest.raises(ValueError, match="灰度"):
            load_grayscale_png_u8(buf.getvalue())

    def test_render_png_returns_valid_image(self):
        rng = np.random.default_rng(14)
        img = (rng.random((32, 32)) * 255).astype(np.uint8)
        res = run_default(img, rng_seed=14)
        data = render_comparison_png(img, res)
        im = Image.open(io.BytesIO(data))
        assert im.format == "PNG" and im.mode == "RGB"

    def test_bad_json_rejected(self):
        with pytest.raises(ValueError):
            load_weights_json(b"{not json")


# ----------------------------------------------------------------- 6. HTTP
class TestHTTPQuant:
    def _post(self, client, img, wts, data=None):
        return client.post(
            "/quant/infer",
            files={"image": ("a.png", png_bytes(img), "image/png"),
                   "weights": ("w.json", json.dumps(wts), "application/json")},
            data=data or {"scale_w1": "0.05", "scale_w2": "0.05",
                          "scale_a1": "0.02"})

    def test_quant_page_renders(self):
        from fastapi.testclient import TestClient
        client = TestClient(app)
        r = client.get("/quant")
        assert r.status_code == 200 and "量化对照" in r.text

    def test_success_html_and_json(self):
        from fastapi.testclient import TestClient
        client = TestClient(app)
        rng = np.random.default_rng(21)
        img = (rng.random((64, 64)) * 255).astype(np.uint8)
        wts = make_weights(rng)

        r = self._post(client, img, wts)
        assert r.status_code == 200, r.text
        assert "data:image/png;base64" in r.text
        assert "量化新增" in r.text and "量化消失" in r.text

        rj = self._post(client, img, wts,
                        {"scale_w1": "0.05", "scale_w2": "0.05",
                         "scale_a1": "0.02", "as_json": "1"})
        assert rj.status_code == 200
        body = rj.json()
        assert body["output_shape"] == [1, 32, 32]
        assert set(body["counts"]) == {
            "grid_cells", "float_crack", "quant_crack", "agreed_crack",
            "new_crack", "gone_crack"}
        assert len(body["new_mask"]) == 32

    def test_error_page_has_no_image_and_shows_reason(self):
        from fastapi.testclient import TestClient
        client = TestClient(app)
        rng = np.random.default_rng(22)
        img = (rng.random((32, 32)) * 255).astype(np.uint8)
        wts = make_weights(rng)
        wts["layer1"]["bias"] = [1e20, 0, 0, 0]
        r = self._post(client, img, wts,
                       {"scale_w1": "1e-12", "scale_w2": "0.05",
                        "scale_a1": "0.02"})
        assert r.status_code == 400
        assert "量化偏置超出 int32" in r.text
        assert "data:image" not in r.text  # 旧图被清掉，只显示原因

    def test_bad_scale_returns_400(self):
        from fastapi.testclient import TestClient
        client = TestClient(app)
        rng = np.random.default_rng(23)
        img = (rng.random((16, 16)) * 255).astype(np.uint8)
        r = self._post(client, img, make_weights(rng),
                       {"scale_w1": "0", "scale_w2": "0.05",
                        "scale_a1": "0.02", "as_json": "1"})
        assert r.status_code == 400 and "正" in r.json()["error"]

    def test_oversize_image_400(self):
        from fastapi.testclient import TestClient
        client = TestClient(app)
        rng = np.random.default_rng(24)
        img = (rng.random((129, 64)) * 255).astype(np.uint8)
        r = self._post(client, img, make_weights(rng))
        assert r.status_code == 400 and "128" in r.text
