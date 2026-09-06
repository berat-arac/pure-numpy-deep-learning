from __future__ import annotations

import numpy as np

from puredl.core import Module, Parameter
from puredl.layers import Dropout, Embedding, GELU, LayerNorm, Linear

from .attention import MultiHeadSelfAttention


class FeedForward(Module):
    """Transformer feed-forward sublayer."""

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        *,
        rng: np.random.Generator | None = None,
        dtype=np.float32,
    ) -> None:
        super().__init__()
        rng = rng or np.random.default_rng()
        self.fc1 = Linear(d_model, hidden_dim, rng=rng, dtype=dtype)
        self.activation = GELU()
        self.fc2 = Linear(hidden_dim, d_model, rng=rng, dtype=dtype)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.fc2.forward(self.activation.forward(self.fc1.forward(x)))

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        grad = self.fc2.backward(grad_out)
        grad = self.activation.backward(grad)
        return self.fc1.backward(grad)


class DecoderBlock(Module):
    """Pre-normalized causal Transformer decoder block."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        *,
        dropout: float = 0.0,
        rng: np.random.Generator | None = None,
        dtype=np.float32,
    ) -> None:
        super().__init__()
        rng = rng or np.random.default_rng()
        self.norm1 = LayerNorm(d_model, dtype=dtype)
        self.attention = MultiHeadSelfAttention(
            d_model,
            num_heads,
            causal=True,
            rng=rng,
            dtype=dtype,
        )
        self.dropout1 = Dropout(dropout, rng=rng)
        self.norm2 = LayerNorm(d_model, dtype=dtype)
        self.ff = FeedForward(d_model, ff_dim, rng=rng, dtype=dtype)
        self.dropout2 = Dropout(dropout, rng=rng)

    def forward(self, x: np.ndarray) -> np.ndarray:
        attn_branch = self.attention.forward(self.norm1.forward(x))
        after_attn = x + self.dropout1.forward(attn_branch)
        ff_branch = self.ff.forward(self.norm2.forward(after_attn))
        return after_attn + self.dropout2.forward(ff_branch)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        grad_after_attn = grad_out.copy()
        grad_ff = self.dropout2.backward(grad_out)
        grad_norm2 = self.ff.backward(grad_ff)
        grad_after_attn += self.norm2.backward(grad_norm2)

        grad_x = grad_after_attn.copy()
        grad_attn = self.dropout1.backward(grad_after_attn)
        grad_norm1 = self.attention.backward(grad_attn)
        grad_x += self.norm1.backward(grad_norm1)
        return grad_x


class TinyTransformerLM(Module):
    """Decoder-only character language model with trainable positional embeddings."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        *,
        dropout: float = 0.0,
        rng: np.random.Generator | None = None,
        dtype=np.float32,
    ) -> None:
        super().__init__()
        if vocab_size <= 0 or context_length <= 0 or num_layers <= 0:
            raise ValueError("vocab_size, context_length, and num_layers must be positive")
        rng = rng or np.random.default_rng()
        self.vocab_size = int(vocab_size)
        self.context_length = int(context_length)
        self.d_model = int(d_model)
        self.token_embedding = Embedding(vocab_size, d_model, rng=rng, dtype=dtype)
        pos_scale = 1.0 / np.sqrt(float(d_model))
        self.position_embedding = Parameter(
            rng.normal(0.0, pos_scale, size=(context_length, d_model)).astype(dtype)
        )
        self.input_dropout = Dropout(dropout, rng=rng)
        self.blocks = [
            DecoderBlock(
                d_model,
                num_heads,
                ff_dim,
                dropout=dropout,
                rng=rng,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ]
        self.final_norm = LayerNorm(d_model, dtype=dtype)
        self.head = Linear(d_model, vocab_size, rng=rng, dtype=dtype)
        self._time: int | None = None

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(token_ids)
        if ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, time]")
        time = ids.shape[1]
        if time > self.context_length:
            raise ValueError("sequence exceeds context length")
        x = self.token_embedding.forward(ids)
        x = x + self.position_embedding.data[:time][None, :, :]
        x = self.input_dropout.forward(x)
        for block in self.blocks:
            x = block.forward(x)
        x = self.final_norm.forward(x)
        self._time = time
        return self.head.forward(x)

    def backward(self, grad_logits: np.ndarray) -> None:
        if self._time is None:
            raise RuntimeError("forward must be called before backward")
        grad = self.head.backward(grad_logits)
        grad = self.final_norm.backward(grad)
        for block in reversed(self.blocks):
            grad = block.backward(grad)
        grad = self.input_dropout.backward(grad)
        self.position_embedding.grad.fill(0)
        self.position_embedding.grad[: self._time] = grad.sum(axis=0)
        self.token_embedding.backward(grad)

    def next_token_logits(self, token_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(token_ids, dtype=np.int64)
        if ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, time]")
        if ids.shape[1] > self.context_length:
            ids = ids[:, -self.context_length :]
        logits = self.forward(ids)
        return logits[:, -1]
