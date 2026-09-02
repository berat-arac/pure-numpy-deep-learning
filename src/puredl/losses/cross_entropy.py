from __future__ import annotations

import numpy as np


class CrossEntropyLoss:
    """Mean cross entropy for integer class targets."""

    def __init__(self) -> None:
        self._probs: np.ndarray | None = None
        self._targets: np.ndarray | None = None

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        if logits.ndim != 2:
            raise ValueError("logits must have shape [batch, classes]")
        if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
            raise ValueError("targets must have shape [batch]")
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        self._probs = probs
        self._targets = targets.astype(np.int64, copy=False)
        batch = logits.shape[0]
        chosen = probs[np.arange(batch), self._targets]
        return float(-np.log(np.clip(chosen, 1e-12, None)).mean())

    __call__ = forward

    def backward(self) -> np.ndarray:
        if self._probs is None or self._targets is None:
            raise RuntimeError("forward must be called before backward")
        grad = self._probs.copy()
        batch = grad.shape[0]
        grad[np.arange(batch), self._targets] -= 1.0
        grad /= batch
        return grad
