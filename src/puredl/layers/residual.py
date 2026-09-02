from __future__ import annotations

import numpy as np

from puredl.core import Module, Sequential
from .activations import ReLU
from .conv import Conv2D
from .normalization import BatchNorm2D


class BasicResidualBlock(Module):
    """Two-convolution residual block with optional projection shortcut."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        rng: np.random.Generator | None = None,
        dtype=np.float32,
    ) -> None:
        super().__init__()
        if stride <= 0:
            raise ValueError("stride must be positive")
        rng = rng or np.random.default_rng()
        self.conv1 = Conv2D(
            in_channels,
            out_channels,
            3,
            stride=stride,
            padding=1,
            bias=False,
            rng=rng,
            dtype=dtype,
        )
        self.bn1 = BatchNorm2D(out_channels, dtype=dtype)
        self.act1 = ReLU()
        self.conv2 = Conv2D(
            out_channels,
            out_channels,
            3,
            padding=1,
            bias=False,
            rng=rng,
            dtype=dtype,
        )
        self.bn2 = BatchNorm2D(out_channels, dtype=dtype)
        self.projection = None
        if stride != 1 or in_channels != out_channels:
            self.projection = Sequential(
                Conv2D(
                    in_channels,
                    out_channels,
                    1,
                    stride=stride,
                    bias=False,
                    rng=rng,
                    dtype=dtype,
                ),
                BatchNorm2D(out_channels, dtype=dtype),
            )
        self.out_act = ReLU()

    def forward(self, x: np.ndarray) -> np.ndarray:
        identity = x if self.projection is None else self.projection.forward(x)
        out = self.conv1.forward(x)
        out = self.bn1.forward(out)
        out = self.act1.forward(out)
        out = self.conv2.forward(out)
        out = self.bn2.forward(out)
        return self.out_act.forward(out + identity)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        grad_sum = self.out_act.backward(grad_out)

        grad_main = self.bn2.backward(grad_sum)
        grad_main = self.conv2.backward(grad_main)
        grad_main = self.act1.backward(grad_main)
        grad_main = self.bn1.backward(grad_main)
        grad_main = self.conv1.backward(grad_main)

        grad_skip = grad_sum
        if self.projection is not None:
            grad_skip = self.projection.backward(grad_skip)
        return grad_main + grad_skip
