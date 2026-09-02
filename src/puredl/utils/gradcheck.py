from __future__ import annotations

from collections.abc import Callable

import numpy as np


def finite_difference_gradient(
    array: np.ndarray,
    scalar_fn: Callable[[], float],
    *,
    eps: float = 1e-5,
) -> np.ndarray:
    grad = np.zeros_like(array, dtype=np.float64)
    for index in np.ndindex(array.shape):
        original = float(array[index])
        array[index] = original + eps
        plus = scalar_fn()
        array[index] = original - eps
        minus = scalar_fn()
        array[index] = original
        grad[index] = (plus - minus) / (2.0 * eps)
    return grad


def relative_error(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    numerator = np.max(np.abs(a - b))
    denominator = np.max(np.maximum(np.abs(a) + np.abs(b), eps))
    return float(numerator / denominator)
