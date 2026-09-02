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
