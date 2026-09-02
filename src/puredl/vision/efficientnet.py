from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from puredl.core import Module, Sequential
from puredl.layers import (
    BatchNorm2D,
    Conv2D,
    DepthwiseConv2D,
    GlobalAveragePool2D,
    Linear,
    PointwiseConv2D,
    SiLU,
    SqueezeExcitation,
    StochasticDepth,
)


def _round_channels(channels: int, width: float, divisor: int = 8) -> int:
    scaled = channels * width
    rounded = max(divisor, int(scaled + divisor / 2) // divisor * divisor)
    if rounded < 0.9 * scaled:
        rounded += divisor
    return int(rounded)


class ConvBNSiLU(Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        rng: np.random.Generator,
        dtype=np.float32,
    ) -> None:
        super().__init__(
            Conv2D(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
                rng=rng,
                dtype=dtype,
            ),
            BatchNorm2D(out_channels, dtype=dtype),
            SiLU(),
        )


class MBConv(Module):
    """Mobile inverted bottleneck with depthwise convolution and SE gating."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        expand_ratio: int = 6,
        se_ratio: float = 0.25,
        drop_prob: float = 0.0,
        rng: np.random.Generator | None = None,
        dtype=np.float32,
    ) -> None:
        super().__init__()
        if kernel_size not in (3, 5):
            raise ValueError("MBConv kernel_size must be 3 or 5")
        if stride not in (1, 2):
            raise ValueError("MBConv stride must be 1 or 2")
        if expand_ratio <= 0:
            raise ValueError("expand_ratio must be positive")
        if not 0.0 < se_ratio <= 1.0:
            raise ValueError("se_ratio must be in (0, 1]")

        rng = rng or np.random.default_rng()
        expanded_channels = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        self.expand = None
        if expand_ratio != 1:
            self.expand = Sequential(
                PointwiseConv2D(
                    in_channels,
                    expanded_channels,
                    bias=False,
                    rng=rng,
                    dtype=dtype,
                ),
                BatchNorm2D(expanded_channels, dtype=dtype),
                SiLU(),
            )

        self.depthwise = Sequential(
            DepthwiseConv2D(
                expanded_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=False,
                rng=rng,
                dtype=dtype,
            ),
            BatchNorm2D(expanded_channels, dtype=dtype),
            SiLU(),
        )
        squeeze_channels = max(1, int(in_channels * se_ratio))
        self.se = SqueezeExcitation(
            expanded_channels,
            squeeze_channels=squeeze_channels,
            rng=rng,
            dtype=dtype,
        )
        self.project = Sequential(
            PointwiseConv2D(
                expanded_channels,
                out_channels,
                bias=False,
                rng=rng,
                dtype=dtype,
            ),
            BatchNorm2D(out_channels, dtype=dtype),
        )
        self.stochastic_depth = StochasticDepth(drop_prob, rng=rng)

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x
        if self.expand is not None:
            out = self.expand.forward(out)
        out = self.depthwise.forward(out)
        out = self.se.forward(out)
        out = self.project.forward(out)
        if self.use_residual:
            out = self.stochastic_depth.forward(out)
            out = out + x
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        grad_branch = grad_out
        if self.use_residual:
            grad_branch = self.stochastic_depth.backward(grad_branch)
        grad_branch = self.project.backward(grad_branch)
        grad_branch = self.se.backward(grad_branch)
        grad_branch = self.depthwise.backward(grad_branch)
        if self.expand is not None:
            grad_branch = self.expand.backward(grad_branch)
        if self.use_residual:
            return grad_branch + grad_out
        return grad_branch


@dataclass(frozen=True)
class MiniMBConvConfig:
    out_channels: int
    kernel_size: int
    stride: int
    expand_ratio: int
    repeats: int


class MiniEfficientNet(Module):
    """CIFAR-scale EfficientNet-style model built from real MBConv components.

    This is intentionally smaller than EfficientNet-B0 so the entire training
    loop remains practical on a CPU with NumPy only.
    """

    def __init__(
        self,
        num_classes: int = 10,
        *,
        width: float = 0.5,
        drop_path_rate: float = 0.10,
        rng: np.random.Generator | None = None,
        dtype=np.float32,
    ) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("width must be positive")
        rng = rng or np.random.default_rng()

        stem_channels = _round_channels(24, width)
        self.stem = ConvBNSiLU(
            3,
            stem_channels,
            3,
            padding=1,
            rng=rng,
            dtype=dtype,
        )

        configs = (
            MiniMBConvConfig(24, 3, 1, 1, 1),
            MiniMBConvConfig(32, 3, 2, 4, 2),
            MiniMBConvConfig(48, 5, 2, 4, 2),
            MiniMBConvConfig(64, 3, 2, 4, 2),
        )
        total_blocks = sum(config.repeats for config in configs)
        blocks: list[MBConv] = []
        in_channels = stem_channels
        block_index = 0
        for config in configs:
            out_channels = _round_channels(config.out_channels, width)
            for repeat in range(config.repeats):
                stride = config.stride if repeat == 0 else 1
                progress = block_index / max(1, total_blocks - 1)
                block_drop = drop_path_rate * progress
                block = MBConv(
                    in_channels,
                    out_channels,
                    kernel_size=config.kernel_size,
                    stride=stride,
                    expand_ratio=config.expand_ratio,
                    se_ratio=0.25,
                    drop_prob=block_drop,
                    rng=rng,
                    dtype=dtype,
                )
                blocks.append(block)
                in_channels = out_channels
                block_index += 1
        self.blocks = blocks

        head_channels = _round_channels(128, width)
        self.head = ConvBNSiLU(
            in_channels,
            head_channels,
            1,
            rng=rng,
            dtype=dtype,
        )
        self.pool = GlobalAveragePool2D()
        self.classifier = Linear(
            head_channels,
            num_classes,
            rng=rng,
            dtype=dtype,
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = self.stem.forward(x)
        for block in self.blocks:
            out = block.forward(out)
        out = self.head.forward(out)
        out = self.pool.forward(out)
        return self.classifier.forward(out)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        grad = self.classifier.backward(grad_out)
        grad = self.pool.backward(grad)
        grad = self.head.backward(grad)
        for block in reversed(self.blocks):
            grad = block.backward(grad)
        return self.stem.backward(grad)
