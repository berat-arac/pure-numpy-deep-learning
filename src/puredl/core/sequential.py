from __future__ import annotations

import numpy as np

from .module import Module


class Sequential(Module):
    """Simple sequential container for layers with explicit backward methods."""

    def __init__(self, *layers: Module) -> None:
        super().__init__()
        self.layers = list(layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        grad = grad_out
        for layer in reversed(self.layers):
            backward = getattr(layer, "backward", None)
            if backward is None:
                raise TypeError(f"{type(layer).__name__} does not implement backward")
            grad = backward(grad)
        return grad
