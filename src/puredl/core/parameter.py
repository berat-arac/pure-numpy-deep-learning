from __future__ import annotations

import numpy as np


class Parameter:
    """Trainable NumPy array with an explicitly managed gradient buffer."""

    def __init__(self, data: np.ndarray):
        array = np.asarray(data)
        if not np.issubdtype(array.dtype, np.floating):
            array = array.astype(np.float32)
        self.data = array
        self.grad = np.zeros_like(self.data)

    def zero_grad(self) -> None:
        self.grad.fill(0)
