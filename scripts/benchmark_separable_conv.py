"""Compare standard and depthwise-separable convolution on the local CPU."""
from __future__ import annotations

import argparse
import time

import numpy as np

from puredl.layers import Conv2D, DepthwiseSeparableConv2D


def parameter_count(module) -> int:
    return sum(parameter.data.size for parameter in module.parameters())


def measure(layer, x, repeats: int) -> float:
    y = layer.forward(x)
    layer.backward(np.ones_like(y))
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        y = layer.forward(x)
        layer.backward(np.ones_like(y))
        samples.append(time.perf_counter() - start)
    return float(np.median(samples))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--in-channels", type=int, default=16)
    parser.add_argument("--out-channels", type=int, default=32)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(99)
    x = rng.normal(
        size=(args.batch_size, args.in_channels, args.size, args.size)
    ).astype(np.float32)
    standard = Conv2D(
        args.in_channels,
        args.out_channels,
        3,
        padding=1,
        bias=False,
        rng=rng,
    )
    separable = DepthwiseSeparableConv2D(
        args.in_channels,
        args.out_channels,
        3,
        padding=1,
        bias=False,
        rng=rng,
    )

    standard_time = measure(standard, x, args.repeats)
    separable_time = measure(separable, x, args.repeats)
    standard_params = parameter_count(standard)
    separable_params = parameter_count(separable)
    reduction = 100.0 * (1.0 - separable_params / standard_params)

    print(
        f"standard params={standard_params:,} median_forward_backward={standard_time:.6f}s"
    )
    print(
        f"separable params={separable_params:,} median_forward_backward={separable_time:.6f}s"
    )
    print(f"parameter_reduction={reduction:.2f}%")
    print(f"runtime_ratio_separable_over_standard={separable_time / standard_time:.3f}")


if __name__ == "__main__":
    main()
