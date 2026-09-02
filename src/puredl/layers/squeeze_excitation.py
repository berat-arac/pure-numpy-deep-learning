from __future__ import annotations

import numpy as np

from puredl.core import Module
from .activations import SiLU
from .global_pool import GlobalAveragePool2D
from .linear import Linear


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.result_type(x, np.float32))
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


class SqueezeExcitation(Module):
    """Channel recalibration block with global pooling and two linear layers."""

    def __init__(
        self,
        channels: int,
        *,
        squeeze_channels: int | None = None,
        squeeze_ratio: float = 0.25,
        rng: np.random.Generator | None = None,
        dtype=np.float32,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if squeeze_channels is None:
            if squeeze_ratio <= 0:
                raise ValueError("squeeze_ratio must be positive")
            squeeze_channels = max(1, int(channels * squeeze_ratio))
        rng = rng or np.random.default_rng()
        self.channels = int(channels)
        self.pool = GlobalAveragePool2D()
        self.reduce = Linear(channels, squeeze_channels, rng=rng, dtype=dtype)
        self.act = SiLU()
        self.expand = Linear(squeeze_channels, channels, rng=rng, dtype=dtype)
        self._input: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._gate: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError("SqueezeExcitation expects NCHW input with matching channels")
        pooled = self.pool.forward(x)
        hidden = self.reduce.forward(pooled)
        hidden = self.act.forward(hidden)
        logits = self.expand.forward(hidden)
        gate = _sigmoid(logits)
        scale = gate[:, :, None, None]
        self._input = x
        self._gate = gate
        self._scale = scale
        return x * scale

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._input is None or self._scale is None or self._gate is None:
            raise RuntimeError("forward must be called before backward")
        if grad_out.shape != self._input.shape:
            raise ValueError("grad_out shape mismatch")

        direct = grad_out * self._scale
        grad_scale = (grad_out * self._input).sum(axis=(2, 3))
        grad_logits = grad_scale * self._gate * (1.0 - self._gate)
        grad_hidden = self.expand.backward(grad_logits)
        grad_hidden = self.act.backward(grad_hidden)
        grad_pooled = self.reduce.backward(grad_hidden)
        through_gate = self.pool.backward(grad_pooled)
        return direct + through_gate
