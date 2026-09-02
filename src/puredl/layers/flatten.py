from __future__ import annotations

import numpy as np

from puredl.core import Module


class Flatten(Module):
    def __init__(self, start_dim: int = 1) -> None:
        super().__init__()
        if start_dim != 1:
            raise NotImplementedError("v0.1 supports start_dim=1 only")
        self.start_dim = start_dim
        self._input_shape: tuple[int, ...] | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._input_shape is None:
            raise RuntimeError("forward must be called before backward")
        return grad_out.reshape(self._input_shape)
