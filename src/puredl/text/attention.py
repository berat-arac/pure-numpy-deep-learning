from __future__ import annotations

import numpy as np

from puredl.core import Module
from puredl.layers import Linear


class ScaledDotProductAttention(Module):
    """Scaled dot-product attention with optional causal masking."""

    def __init__(self, *, causal: bool = True) -> None:
        super().__init__()
        self.causal = bool(causal)
        self._q: np.ndarray | None = None
        self._k: np.ndarray | None = None
        self._v: np.ndarray | None = None
        self._weights: np.ndarray | None = None

    def forward(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("q, k, and v must have shape [batch, heads, time, head_dim]")
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError("self-attention expects q, k, and v with matching shapes")
        head_dim = q.shape[-1]
        scale = 1.0 / np.sqrt(float(head_dim))
        scores = (q @ np.swapaxes(k, -1, -2)) * scale
        if self.causal:
            time = q.shape[-2]
            mask = np.triu(np.ones((time, time), dtype=bool), k=1)
            scores = np.where(mask, -np.inf, scores)
        row_max = np.max(scores, axis=-1, keepdims=True)
        shifted = scores - row_max
        exp_scores = np.exp(shifted)
        weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        self._q = q
        self._k = k
        self._v = v
        self._weights = weights
        return weights @ v

    def backward(self, grad_out: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._q is None or self._k is None or self._v is None or self._weights is None:
            raise RuntimeError("forward must be called before backward")
        q, k, v, weights = self._q, self._k, self._v, self._weights
        if grad_out.shape != q.shape:
            raise ValueError("grad_out shape mismatch")

        grad_weights = grad_out @ np.swapaxes(v, -1, -2)
        grad_v = np.swapaxes(weights, -1, -2) @ grad_out

        dot = np.sum(grad_weights * weights, axis=-1, keepdims=True)
        grad_scores = weights * (grad_weights - dot)
        scale = 1.0 / np.sqrt(float(q.shape[-1]))
        grad_q = (grad_scores @ k) * scale
        grad_k = (np.swapaxes(grad_scores, -1, -2) @ q) * scale
        return grad_q, grad_k, grad_v


class MultiHeadSelfAttention(Module):
    """Multi-head causal self-attention built from explicit NumPy projections."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        causal: bool = True,
        rng: np.random.Generator | None = None,
        dtype=np.float32,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_heads <= 0:
            raise ValueError("d_model and num_heads must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads
        rng = rng or np.random.default_rng()
        self.q_proj = Linear(d_model, d_model, rng=rng, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, rng=rng, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, rng=rng, dtype=dtype)
        self.attention = ScaledDotProductAttention(causal=causal)
        self.out_proj = Linear(d_model, d_model, rng=rng, dtype=dtype)
        self._shape: tuple[int, int] | None = None

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        batch, time, _ = x.shape
        return x.reshape(batch, time, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        batch, heads, time, dim = x.shape
        if heads != self.num_heads or dim != self.head_dim:
            raise ValueError("head shape mismatch")
        return x.transpose(0, 2, 1, 3).reshape(batch, time, self.d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError("x must have shape [batch, time, d_model]")
        q = self._split_heads(self.q_proj.forward(x))
        k = self._split_heads(self.k_proj.forward(x))
        v = self._split_heads(self.v_proj.forward(x))
        attended = self.attention.forward(q, k, v)
        merged = self._merge_heads(attended)
        self._shape = (x.shape[0], x.shape[1])
        return self.out_proj.forward(merged)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._shape is None:
            raise RuntimeError("forward must be called before backward")
        batch, time = self._shape
        if grad_out.shape != (batch, time, self.d_model):
            raise ValueError("grad_out shape mismatch")
        grad_merged = self.out_proj.backward(grad_out)
        grad_heads = self._split_heads(grad_merged)
        grad_q, grad_k, grad_v = self.attention.backward(grad_heads)
        grad_q = self._merge_heads(grad_q)
        grad_k = self._merge_heads(grad_k)
        grad_v = self._merge_heads(grad_v)
        return (
            self.q_proj.backward(grad_q)
            + self.k_proj.backward(grad_k)
            + self.v_proj.backward(grad_v)
        )
