from __future__ import annotations

import numpy as np

from puredl.core import Module, Parameter


class Linear(Module):
    """Fully connected layer using explicit forward and backward passes."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        rng: np.random.Generator | None = None,
        dtype=np.float32,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        self.in_features = in_features
        self.out_features = out_features
        rng = rng or np.random.default_rng()
        std = np.sqrt(2.0 / in_features)
        weight = rng.normal(0.0, std, size=(out_features, in_features)).astype(dtype)
        self.weight = Parameter(weight)
        self.bias = Parameter(np.zeros(out_features, dtype=dtype)) if bias else None
        self._input: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected last dimension {self.in_features}, got {x.shape[-1]}"
            )
        self._input = x
        y = x @ self.weight.data.T
        if self.bias is not None:
            y = y + self.bias.data
        return y

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._input is None:
            raise RuntimeError("forward must be called before backward")
        x = self._input
        if grad_out.shape[:-1] != x.shape[:-1] or grad_out.shape[-1] != self.out_features:
            raise ValueError("grad_out shape is incompatible with the cached input")

        x2 = x.reshape(-1, self.in_features)
        g2 = grad_out.reshape(-1, self.out_features)
        self.weight.grad[...] = g2.T @ x2
        if self.bias is not None:
            self.bias.grad[...] = g2.sum(axis=0)
        grad_input = g2 @ self.weight.data
        return grad_input.reshape(x.shape)
