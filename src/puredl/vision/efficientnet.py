from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from puredl.core import Module, Sequential
from puredl.layers import (
    BatchNorm2D,
    Conv2D,
    DepthwiseConv2D,
    Dropout,
    GlobalAveragePool2D,
    Linear,
    PointwiseConv2D,
    SiLU,
    SqueezeExcitation,
    StochasticDepth,
)


def round_channels(channels: int, width_mult: float, divisor: int = 8) -> int:
    """Scale channel count while keeping hardware-friendly divisibility."""
    if width_mult <= 0:
        raise ValueError("width_mult must be positive")
    scaled = channels * width_mult
    rounded = max(divisor, int(scaled + divisor / 2) // divisor * divisor)
    if rounded < 0.9 * scaled:
        rounded += divisor
    return int(rounded)


def round_repeats(repeats: int, depth_mult: float) -> int:
    """Scale the number of repeated blocks in a stage."""
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if depth_mult <= 0:
        raise ValueError("depth_mult must be positive")
    return int(math.ceil(repeats * depth_mult))


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
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.expand_ratio = int(expand_ratio)
        self.drop_prob = float(drop_prob)
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
class EfficientNetStageConfig:
    expand_ratio: int
    kernel_size: int
    stride: int
    in_channels: int
    out_channels: int
    repeats: int


EFFICIENTNET_B0_STAGES = (
    EfficientNetStageConfig(1, 3, 1, 32, 16, 1),
    EfficientNetStageConfig(6, 3, 2, 16, 24, 2),
    EfficientNetStageConfig(6, 5, 2, 24, 40, 2),
    EfficientNetStageConfig(6, 3, 2, 40, 80, 3),
    EfficientNetStageConfig(6, 5, 1, 80, 112, 3),
    EfficientNetStageConfig(6, 5, 2, 112, 192, 4),
    EfficientNetStageConfig(6, 3, 1, 192, 320, 1),
)


class EfficientNet(Module):
    """Config-driven EfficientNet with explicit NumPy forward and backward passes."""

    def __init__(
        self,
        num_classes: int = 1000,
        *,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        dropout_rate: float = 0.2,
        drop_path_rate: float = 0.2,
        stem_stride: int = 2,
        stages: tuple[EfficientNetStageConfig, ...] = EFFICIENTNET_B0_STAGES,
        rng: np.random.Generator | None = None,
        dtype=np.float32,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if stem_stride not in (1, 2):
            raise ValueError("stem_stride must be 1 or 2")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in [0, 1)")
        if not 0.0 <= drop_path_rate < 1.0:
            raise ValueError("drop_path_rate must be in [0, 1)")
        if not stages:
            raise ValueError("stages must not be empty")

        rng = rng or np.random.default_rng()
        self.num_classes = int(num_classes)
        self.width_mult = float(width_mult)
        self.depth_mult = float(depth_mult)
        self.dropout_rate = float(dropout_rate)
        self.drop_path_rate = float(drop_path_rate)
        self.stem_stride = int(stem_stride)
        self.stage_configs = tuple(stages)

        stem_channels = round_channels(32, width_mult)
        self.stem = ConvBNSiLU(
            3,
            stem_channels,
            3,
            stride=stem_stride,
            padding=1,
            rng=rng,
            dtype=dtype,
        )

        scaled_repeats = [round_repeats(config.repeats, depth_mult) for config in stages]
        total_blocks = sum(scaled_repeats)
        self.total_blocks = total_blocks
        self.stages: list[Sequential] = []
        self.blocks: list[MBConv] = []

        in_channels = stem_channels
        block_index = 0
        for config, repeats in zip(stages, scaled_repeats):
            out_channels = round_channels(config.out_channels, width_mult)
            stage_blocks: list[MBConv] = []
            for repeat_index in range(repeats):
                stride = config.stride if repeat_index == 0 else 1
                block_drop = drop_path_rate * block_index / max(1, total_blocks)
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
                stage_blocks.append(block)
                self.blocks.append(block)
                in_channels = out_channels
                block_index += 1
            self.stages.append(Sequential(*stage_blocks))

        head_channels = round_channels(1280, width_mult)
        self.head = ConvBNSiLU(
            in_channels,
            head_channels,
            1,
            rng=rng,
            dtype=dtype,
        )
        self.pool = GlobalAveragePool2D()
        self.dropout = Dropout(dropout_rate, rng=rng)
        self.classifier = Linear(head_channels, num_classes, rng=rng, dtype=dtype)

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = self.stem.forward(x)
        for stage in self.stages:
            out = stage.forward(out)
        out = self.head.forward(out)
        out = self.pool.forward(out)
        out = self.dropout.forward(out)
        return self.classifier.forward(out)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        grad = self.classifier.backward(grad_out)
        grad = self.dropout.backward(grad)
        grad = self.pool.backward(grad)
        grad = self.head.backward(grad)
        for stage in reversed(self.stages):
            grad = stage.backward(grad)
        return self.stem.backward(grad)


class EfficientNetB0(EfficientNet):
    """Canonical B0 stage layout with an optional CIFAR-friendly stem."""

    def __init__(
        self,
        num_classes: int = 1000,
        *,
        cifar_stem: bool = False,
        dropout_rate: float = 0.2,
        drop_path_rate: float = 0.2,
        rng: np.random.Generator | None = None,
        dtype=np.float32,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            width_mult=1.0,
            depth_mult=1.0,
            dropout_rate=dropout_rate,
            drop_path_rate=drop_path_rate,
            stem_stride=1 if cifar_stem else 2,
            stages=EFFICIENTNET_B0_STAGES,
            rng=rng,
            dtype=dtype,
        )


@dataclass(frozen=True)
class MiniMBConvConfig:
    out_channels: int
    kernel_size: int
    stride: int
    expand_ratio: int
    repeats: int


class MiniEfficientNet(Module):
    """Small EfficientNet-style model used for practical CPU experiments."""

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

        stem_channels = round_channels(24, width)
        self.stem = ConvBNSiLU(3, stem_channels, 3, padding=1, rng=rng, dtype=dtype)
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
            out_channels = round_channels(config.out_channels, width)
            for repeat in range(config.repeats):
                stride = config.stride if repeat == 0 else 1
                progress = block_index / max(1, total_blocks - 1)
                block = MBConv(
                    in_channels,
                    out_channels,
                    kernel_size=config.kernel_size,
                    stride=stride,
                    expand_ratio=config.expand_ratio,
                    se_ratio=0.25,
                    drop_prob=drop_path_rate * progress,
                    rng=rng,
                    dtype=dtype,
                )
                blocks.append(block)
                in_channels = out_channels
                block_index += 1
        self.blocks = blocks
        head_channels = round_channels(128, width)
        self.head = ConvBNSiLU(in_channels, head_channels, 1, rng=rng, dtype=dtype)
        self.pool = GlobalAveragePool2D()
        self.classifier = Linear(head_channels, num_classes, rng=rng, dtype=dtype)

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
