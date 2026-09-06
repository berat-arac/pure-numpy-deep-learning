from __future__ import annotations

import numpy as np

from puredl.core import Module, Parameter


class BatchNorm2D(Module):
    """Batch normalization for NCHW tensors with explicit backward math."""

    def __init__(
        self,
        num_features: int,
        *,
        eps: float = 1e-5,
        momentum: float = 0.1,
        dtype=np.float32,
    ) -> None:
        super().__init__()
        if num_features <= 0:
            raise ValueError("num_features must be positive")
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.gamma = Parameter(np.ones(num_features, dtype=dtype))
        self.beta = Parameter(np.zeros(num_features, dtype=dtype))
        self.register_buffer("running_mean", np.zeros(num_features, dtype=dtype))
        self.register_buffer("running_var", np.ones(num_features, dtype=dtype))
        self._x_hat: np.ndarray | None = None
        self._inv_std: np.ndarray | None = None
        self._sample_count: int | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 4 or x.shape[1] != self.num_features:
            raise ValueError("BatchNorm2D expects NCHW input with matching channels")
        axes = (0, 2, 3)
        shape = (1, self.num_features, 1, 1)

        if self.training:
            mean = x.mean(axis=axes, keepdims=True)
            var = x.var(axis=axes, keepdims=True)
            inv_std = 1.0 / np.sqrt(var + self.eps)
            x_hat = (x - mean) * inv_std

            m = x.shape[0] * x.shape[2] * x.shape[3]
            self.running_mean[...] = (1.0 - self.momentum) * self.running_mean + self.momentum * mean.reshape(-1)
            unbiased_var = var.reshape(-1)
            if m > 1:
                unbiased_var = unbiased_var * (m / (m - 1))
            self.running_var[...] = (1.0 - self.momentum) * self.running_var + self.momentum * unbiased_var

            self._x_hat = x_hat
            self._inv_std = inv_std
            self._sample_count = m
        else:
            mean = self.running_mean.reshape(shape)
            var = self.running_var.reshape(shape)
            x_hat = (x - mean) / np.sqrt(var + self.eps)

        return x_hat * self.gamma.data.reshape(shape) + self.beta.data.reshape(shape)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._x_hat is None or self._inv_std is None or self._sample_count is None:
            raise RuntimeError("training forward must be called before backward")
        if grad_out.shape != self._x_hat.shape:
            raise ValueError("grad_out shape must match the training input shape")

        axes = (0, 2, 3)
        shape = (1, self.num_features, 1, 1)
        x_hat = self._x_hat
        m = self._sample_count

        self.beta.grad[...] = grad_out.sum(axis=axes)
        self.gamma.grad[...] = (grad_out * x_hat).sum(axis=axes)

        grad_xhat = grad_out * self.gamma.data.reshape(shape)
        sum_grad = grad_xhat.sum(axis=axes, keepdims=True)
        sum_grad_xhat = (grad_xhat * x_hat).sum(axis=axes, keepdims=True)
        grad_x = (self._inv_std / m) * (
            m * grad_xhat - sum_grad - x_hat * sum_grad_xhat
        )
        return grad_x


class LayerNorm(Module):
    """Layer normalization over the final feature dimension."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5, *, dtype=np.float32) -> None:
        super().__init__()
        if normalized_shape <= 0:
            raise ValueError("normalized_shape must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.normalized_shape = int(normalized_shape)
        self.eps = float(eps)
        self.weight = Parameter(np.ones(self.normalized_shape, dtype=dtype))
        self.bias = Parameter(np.zeros(self.normalized_shape, dtype=dtype))
        self._xhat: np.ndarray | None = None
        self._inv_std: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.shape[-1] != self.normalized_shape:
            raise ValueError(
                f"expected last dimension {self.normalized_shape}, got {x.shape[-1]}"
            )
        mean = x.mean(axis=-1, keepdims=True)
        centered = x - mean
        var = np.mean(centered * centered, axis=-1, keepdims=True)
        inv_std = 1.0 / np.sqrt(var + self.eps)
        xhat = centered * inv_std
        self._xhat = xhat
        self._inv_std = inv_std
        return xhat * self.weight.data + self.bias.data

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._xhat is None or self._inv_std is None:
            raise RuntimeError("forward must be called before backward")
        if grad_out.shape != self._xhat.shape:
            raise ValueError("grad_out shape mismatch")

        reduce_axes = tuple(range(grad_out.ndim - 1))
        self.weight.grad[...] = np.sum(grad_out * self._xhat, axis=reduce_axes)
        self.bias.grad[...] = np.sum(grad_out, axis=reduce_axes)

        grad_norm = grad_out * self.weight.data
        features = self.normalized_shape
        sum_grad = np.sum(grad_norm, axis=-1, keepdims=True)
        sum_grad_xhat = np.sum(grad_norm * self._xhat, axis=-1, keepdims=True)
        return (
            self._inv_std
            * (features * grad_norm - sum_grad - self._xhat * sum_grad_xhat)
            / features
        )
