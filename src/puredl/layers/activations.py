from __future__ import annotations

import numpy as np

from puredl.core import Module


class ReLU(Module):
    def __init__(self) -> None:
        super().__init__()
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0
        return np.maximum(x, 0)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._mask is None:
            raise RuntimeError("forward must be called before backward")
        return grad_out * self._mask


class SiLU(Module):
    def __init__(self) -> None:
        super().__init__()
        self._input: np.ndarray | None = None
        self._sigmoid: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        sigmoid = np.empty_like(x, dtype=np.result_type(x, np.float32))
        positive = x >= 0
        sigmoid[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
        exp_x = np.exp(x[~positive])
        sigmoid[~positive] = exp_x / (1.0 + exp_x)
        self._input = x
        self._sigmoid = sigmoid
        return x * sigmoid

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._input is None or self._sigmoid is None:
            raise RuntimeError("forward must be called before backward")
        x = self._input
        s = self._sigmoid
        return grad_out * (s + x * s * (1.0 - s))


class GELU(Module):
    """Tanh-approximate GELU with an explicit derivative."""

    def __init__(self) -> None:
        super().__init__()
        self._input: np.ndarray | None = None
        self._tanh_term: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        coeff = np.sqrt(2.0 / np.pi)
        inner = coeff * (x + 0.044715 * x * x * x)
        tanh_term = np.tanh(inner)
        self._input = x
        self._tanh_term = tanh_term
        return 0.5 * x * (1.0 + tanh_term)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._input is None or self._tanh_term is None:
            raise RuntimeError("forward must be called before backward")
        x = self._input
        t = self._tanh_term
        coeff = np.sqrt(2.0 / np.pi)
        inner_grad = coeff * (1.0 + 3.0 * 0.044715 * x * x)
        derivative = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * inner_grad
        return grad_out * derivative
