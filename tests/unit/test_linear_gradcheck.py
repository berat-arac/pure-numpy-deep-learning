import numpy as np

from puredl.layers import Linear
from puredl.utils import finite_difference_gradient, relative_error


def test_linear_weight_gradient_passes_finite_difference_check():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(4, 3)).astype(np.float64)
    upstream = rng.normal(size=(4, 2)).astype(np.float64)
    layer = Linear(3, 2, rng=rng, dtype=np.float64)

    layer.forward(x)
    layer.backward(upstream)
    analytical = layer.weight.grad.copy()

    def scalar_fn() -> float:
        return float(np.sum(layer.forward(x) * upstream))

    numerical = finite_difference_gradient(layer.weight.data, scalar_fn, eps=1e-6)
    assert relative_error(analytical, numerical) < 1e-8
