import numpy as np

from puredl.layers import Conv2D
from puredl.utils import finite_difference_gradient, relative_error


def test_conv2d_weight_gradient_passes_finite_difference_check():
    rng = np.random.default_rng(91)
    x = rng.normal(size=(1, 1, 4, 4)).astype(np.float64)
    layer = Conv2D(1, 1, 3, padding=1, rng=rng, dtype=np.float64)
    y = layer.forward(x)
    upstream = rng.normal(size=y.shape).astype(np.float64)
    layer.backward(upstream)
    analytical = layer.weight.grad.copy()

    def scalar_fn() -> float:
        return float(np.sum(layer.forward(x) * upstream))

    numerical = finite_difference_gradient(layer.weight.data, scalar_fn, eps=1e-6)
    assert relative_error(analytical, numerical) < 1e-7
