from __future__ import annotations

import numpy as np

from puredl.losses import CrossEntropyLoss


def clip_grad_norm(parameters, max_norm: float, eps: float = 1e-12) -> float:
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    params = list(parameters)
    total_sq = 0.0
    for parameter in params:
        grad = parameter.grad.astype(np.float64, copy=False)
        total_sq += float(np.sum(grad * grad))
    total_norm = float(np.sqrt(total_sq))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + eps)
        for parameter in params:
            parameter.grad *= scale
    return total_norm


def language_model_loss(
    loss_fn: CrossEntropyLoss,
    logits: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, np.ndarray]:
    if logits.ndim != 3 or targets.ndim != 2:
        raise ValueError("expected logits [batch,time,vocab] and targets [batch,time]")
    batch, time, vocab = logits.shape
    if targets.shape != (batch, time):
        raise ValueError("target shape mismatch")
    loss = loss_fn.forward(logits.reshape(batch * time, vocab), targets.reshape(-1))
    grad = loss_fn.backward().reshape(batch, time, vocab)
    return loss, grad
