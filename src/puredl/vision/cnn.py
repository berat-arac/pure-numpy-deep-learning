from __future__ import annotations

import numpy as np

from puredl.core import Module
from puredl.layers import Conv2D, Flatten, Linear, MaxPool2D, ReLU


class SmallCNN(Module):
    """Small image classifier used as the first trainable convolution checkpoint."""

    def __init__(
        self,
        in_channels: int,
        input_size: int,
        num_classes: int,
        *,
        rng: np.random.Generator | None = None,
        dtype=np.float32,
    ) -> None:
        super().__init__()
        rng = rng or np.random.default_rng()
        self.conv1 = Conv2D(in_channels, 8, 3, padding=1, rng=rng, dtype=dtype)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(2)
        self.conv2 = Conv2D(8, 16, 3, padding=1, rng=rng, dtype=dtype)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(2)
        self.flatten = Flatten()
        spatial = input_size // 4
        self.fc = Linear(16 * spatial * spatial, num_classes, rng=rng, dtype=dtype)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)
        x = self.flatten.forward(x)
        return self.fc.forward(x)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        g = self.fc.backward(grad_out)
        g = self.flatten.backward(g)
        g = self.pool2.backward(g)
        g = self.relu2.backward(g)
        g = self.conv2.backward(g)
        g = self.pool1.backward(g)
        g = self.relu1.backward(g)
        return self.conv1.backward(g)
