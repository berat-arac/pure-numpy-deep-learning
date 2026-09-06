from __future__ import annotations

import numpy as np

from puredl.core import Module, Parameter


class Embedding(Module):
    """Integer token lookup table with explicit gradient accumulation."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        rng: np.random.Generator | None = None,
        dtype=np.float32,
    ) -> None:
        super().__init__()
        if num_embeddings <= 0 or embedding_dim <= 0:
            raise ValueError("num_embeddings and embedding_dim must be positive")
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        rng = rng or np.random.default_rng()
        scale = 1.0 / np.sqrt(embedding_dim)
        weight = rng.normal(
            0.0,
            scale,
            size=(num_embeddings, embedding_dim),
        ).astype(dtype)
        self.weight = Parameter(weight)
        self._indices: np.ndarray | None = None

    def forward(self, indices: np.ndarray) -> np.ndarray:
        ids = np.asarray(indices)
        if not np.issubdtype(ids.dtype, np.integer):
            raise TypeError("embedding indices must be integers")
        if ids.size and (ids.min() < 0 or ids.max() >= self.num_embeddings):
            raise ValueError("embedding index out of range")
        self._indices = ids.astype(np.int64, copy=False)
        return self.weight.data[self._indices]

    def backward(self, grad_out: np.ndarray) -> None:
        if self._indices is None:
            raise RuntimeError("forward must be called before backward")
        expected = self._indices.shape + (self.embedding_dim,)
        if grad_out.shape != expected:
            raise ValueError(f"expected grad_out shape {expected}, got {grad_out.shape}")
        self.weight.grad.fill(0)
        flat_ids = self._indices.reshape(-1)
        flat_grad = grad_out.reshape(-1, self.embedding_dim)
        np.add.at(self.weight.grad, flat_ids, flat_grad)
