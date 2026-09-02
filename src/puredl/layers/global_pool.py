from __future__ import annotations

import numpy as np

from puredl.core import Module


class GlobalAveragePool2D(Module):
    def __init__(self, keepdims: bool = False) -> None:
        super().__init__()
        self.keepdims = keepdims
        self._input_shape: tuple[int, ...] | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 4:
            raise ValueError("GlobalAveragePool2D expects NCHW input")
        self._input_shape = x.shape
        return x.mean(axis=(2, 3), keepdims=self.keepdims)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._input_shape is None:
            raise RuntimeError("forward must be called before backward")
        n, c, h, w = self._input_shape
        grad = grad_out if self.keepdims else grad_out[:, :, None, None]
        return np.broadcast_to(grad / (h * w), (n, c, h, w)).copy()
