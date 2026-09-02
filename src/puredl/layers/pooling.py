from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from puredl.core import Module
from .conv import _pair


class MaxPool2D(Module):
    def __init__(
        self,
        kernel_size: int | Sequence[int],
        *,
        stride: int | Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride if stride is not None else kernel_size)
        self._input_shape: tuple[int, ...] | None = None
        self._argmax: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 4:
            raise ValueError("MaxPool2D expects NCHW input")
        n, c, h, w = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh = (h - kh) // sh + 1
        ow = (w - kw) // sw + 1
        if oh <= 0 or ow <= 0:
            raise ValueError("Pooling kernel is larger than input")

        out = np.empty((n, c, oh, ow), dtype=x.dtype)
        argmax = np.empty((n, c, oh, ow), dtype=np.int32)

        for oy in range(oh):
            y0 = oy * sh
            for ox in range(ow):
                x0 = ox * sw
                patch = x[:, :, y0 : y0 + kh, x0 : x0 + kw].reshape(n, c, -1)
                idx = patch.argmax(axis=2)
                argmax[:, :, oy, ox] = idx
                out[:, :, oy, ox] = np.take_along_axis(
                    patch, idx[:, :, None], axis=2
                )[:, :, 0]

        self._input_shape = x.shape
        self._argmax = argmax
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._input_shape is None or self._argmax is None:
            raise RuntimeError("forward must be called before backward")
        n, c, h, w = self._input_shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh, ow = grad_out.shape[2:]
        grad_x = np.zeros((n, c, h, w), dtype=grad_out.dtype)

        n_idx, c_idx = np.indices((n, c))
        for oy in range(oh):
            y0 = oy * sh
            for ox in range(ow):
                x0 = ox * sw
                flat_idx = self._argmax[:, :, oy, ox]
                iy = flat_idx // kw
                ix = flat_idx % kw
                np.add.at(
                    grad_x,
                    (n_idx, c_idx, y0 + iy, x0 + ix),
                    grad_out[:, :, oy, ox],
                )
        return grad_x
