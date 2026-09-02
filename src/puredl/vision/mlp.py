from __future__ import annotations

import numpy as np

from puredl.core import Module
from puredl.layers import Linear, ReLU


class MLPClassifier(Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_classes: int,
        *,
        rng: np.random.Generator | None = None,
        dtype=np.float32,
    ) -> None:
        super().__init__()
        rng = rng or np.random.default_rng()
        self.fc1 = Linear(in_features, hidden_features, rng=rng, dtype=dtype)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_features, num_classes, rng=rng, dtype=dtype)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self.fc1.forward(x)
        x = self.relu.forward(x)
        return self.fc2.forward(x)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        grad = self.fc2.backward(grad_out)
        grad = self.relu.backward(grad)
        return self.fc1.backward(grad)
