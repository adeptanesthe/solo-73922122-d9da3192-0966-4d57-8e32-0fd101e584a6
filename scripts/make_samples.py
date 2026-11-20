#!/usr/bin/env python3
"""生成样例灰度 PNG 与两层权重 JSON（确定性，可重复）。

用法：
    python3 scripts/make_samples.py [输出目录]
默认输出到 samples/。请用 `python3 -I` 运行。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261119)

    # ---- 64x64 灰度“路面”图：暗色背景 + 亮色裂缝状线条 + 噪声 ----
    h = w = 64
    yy, xx = np.mgrid[0:h, 0:w]
    crack = np.exp(-((xx - (0.6 * yy + 8.0)) ** 2) / 6.0)
    crack += 0.8 * np.exp(-((xx - (1.15 * yy - 20.0)) ** 2) / 4.0)
    noise = rng.normal(0.0, 0.04, size=(h, w))
    img = np.clip(0.18 + 0.55 * crack + noise, 0.0, 1.0)
    Image.fromarray((img * 255.0).round().astype(np.uint8), mode="L").save(
        out / "sample_crack_64.png")

    # ---- 小图样例（8x8），便于手工核对窗口/预算 ----
    small = rng.random((8, 8))
    Image.fromarray((small * 255.0).round().astype(np.uint8), mode="L").save(
        out / "sample_random_8.png")

    # ---- 权重：刻意取小尺度，避免输出值过大掩盖浮点误差 ----
    w1 = (rng.normal(0.0, 0.25, size=(4, 1, 3, 3))).tolist()
    b1 = rng.normal(0.0, 0.1, size=(4,)).tolist()
    w2 = (rng.normal(0.0, 0.25, size=(1, 4, 3, 3))).tolist()
    b2 = rng.normal(0.0, 0.1, size=(1,)).tolist()
    weights = {
        "layer1": {"weight": w1, "bias": b1},
        "layer2": {"weight": w2, "bias": b2},
    }
    (out / "weights.json").write_text(
        json.dumps(weights, indent=2), encoding="utf-8")

    # ---- 量化对照专用零均值权重：浮点输出有正有负，默认尺度即可看到
    #      量化新增/消失的裂缝判定差异 ----
    qrng = np.random.default_rng(20261120)
    qw1 = qrng.normal(0.0, 0.3, size=(4, 1, 3, 3)).tolist()
    qb1 = [0.0, 0.0, 0.0, 0.0]
    qw2 = qrng.normal(0.0, 0.3, size=(1, 4, 3, 3)).tolist()
    qb2 = [0.0]
    quant_weights = {
        "layer1": {"weight": qw1, "bias": qb1},
        "layer2": {"weight": qw2, "bias": qb2},
    }
    (out / "weights_quant_demo.json").write_text(
        json.dumps(quant_weights, indent=2), encoding="utf-8")

    print(f"已生成样例：{out}/sample_crack_64.png, "
          f"{out}/sample_random_8.png, {out}/weights.json, "
          f"{out}/weights_quant_demo.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "samples")
