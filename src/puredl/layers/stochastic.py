from __future__ import annotations

import numpy as np

from puredl.core import Module


class Dropout(Module):
    """Elementwise dropout used by classifier heads."""

    def __init__(
        self,
        drop_prob: float = 0.5,
        *,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError("drop_prob must be in [0, 1)")
        self.drop_prob = float(drop_prob)
        self.rng = rng or np.random.default_rng()
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training or self.drop_prob == 0.0:
            self._mask = None
            return x
        keep_prob = 1.0 - self.drop_prob
        mask = (self.rng.random(x.shape) < keep_prob).astype(x.dtype)
        self._mask = mask
        return x * mask / keep_prob

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if not self.training or self.drop_prob == 0.0:
            return grad_out
        if self._mask is None:
            raise RuntimeError("training forward must be called before backward")
        return grad_out * self._mask / (1.0 - self.drop_prob)


class StochasticDepth(Module):
    """Drop complete residual branches per sample during training."""

    def __init__(
        self,
        drop_prob: float = 0.0,
        *,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError("drop_prob must be in [0, 1)")
        self.drop_prob = float(drop_prob)
        self.rng = rng or np.random.default_rng()
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training or self.drop_prob == 0.0:
            self._mask = None
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = (self.rng.random(shape) < keep_prob).astype(x.dtype)
        self._mask = mask
        return x * mask / keep_prob

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if not self.training or self.drop_prob == 0.0:
            return grad_out
        if self._mask is None:
            raise RuntimeError("training forward must be called before backward")
        return grad_out * self._mask / (1.0 - self.drop_prob)
