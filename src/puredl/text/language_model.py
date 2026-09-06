from __future__ import annotations

import numpy as np

from puredl.core import Module
from puredl.layers.embedding import Embedding
from puredl.layers.linear import Linear

from .recurrent import LSTM, VanillaRNN


class CharLanguageModel(Module):
    """Character language model backed by a vanilla RNN or an LSTM."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        *,
        cell: str = "rnn",
        rng: np.random.Generator | None = None,
        dtype=np.float32,
    ) -> None:
        super().__init__()
        if cell not in {"rnn", "lstm"}:
            raise ValueError("cell must be 'rnn' or 'lstm'")
        rng = rng or np.random.default_rng()
        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.hidden_size = int(hidden_size)
        self.cell = cell
        self.embedding = Embedding(vocab_size, embedding_dim, rng=rng, dtype=dtype)
        if cell == "rnn":
            self.recurrent = VanillaRNN(embedding_dim, hidden_size, rng=rng, dtype=dtype)
        else:
            self.recurrent = LSTM(embedding_dim, hidden_size, rng=rng, dtype=dtype)
        self.head = Linear(hidden_size, vocab_size, rng=rng, dtype=dtype)

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        embedded = self.embedding.forward(token_ids)
        hidden = self.recurrent.forward(embedded)
        return self.head.forward(hidden)

    def backward(self, grad_logits: np.ndarray) -> None:
        grad_hidden = self.head.backward(grad_logits)
        recurrent_result = self.recurrent.backward(grad_hidden)
        grad_embedded = recurrent_result[0]
        self.embedding.backward(grad_embedded)

    def initial_state(self, batch_size: int = 1):
        dtype = self.embedding.weight.data.dtype
        h = np.zeros((batch_size, self.hidden_size), dtype=dtype)
        if self.cell == "rnn":
            return h
        c = np.zeros_like(h)
        return h, c

    def step_token(self, token_ids: np.ndarray, state):
        embedded = self.embedding.weight.data[np.asarray(token_ids, dtype=np.int64)]
        if self.cell == "rnn":
            h = self.recurrent.step(embedded, state)
            logits = h @ self.head.weight.data.T
            if self.head.bias is not None:
                logits = logits + self.head.bias.data
            return logits, h
        h_prev, c_prev = state
        h, c = self.recurrent.step(embedded, h_prev, c_prev)
        logits = h @ self.head.weight.data.T
        if self.head.bias is not None:
            logits = logits + self.head.bias.data
        return logits, (h, c)
