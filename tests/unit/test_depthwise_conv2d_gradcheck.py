import numpy as np

from puredl.layers import DepthwiseConv2D
from puredl.utils import finite_difference_gradient, relative_error


def test_depthwise_weight_gradient_passes_finite_difference_check():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(1, 2, 4, 4)).astype(np.float64)
    layer = DepthwiseConv2D(2, 3, padding=1, rng=rng, dtype=np.float64)
    out = layer.forward(x)
    upstream = rng.normal(size=out.shape).astype(np.float64)
    layer.backward(upstream)
    analytical = layer.weight.grad.copy()

    def scalar_fn() -> float:
        return float(np.sum(layer.forward(x) * upstream))

    numerical = finite_difference_gradient(layer.weight.data, scalar_fn, eps=1e-6)
    assert relative_error(analytical, numerical) < 1e-7


def test_depthwise_input_gradient_passes_finite_difference_check():
    rng = np.random.default_rng(18)
    x = rng.normal(size=(1, 2, 3, 3)).astype(np.float64)
    layer = DepthwiseConv2D(2, 3, padding=1, rng=rng, dtype=np.float64)
    out = layer.forward(x)
    upstream = rng.normal(size=out.shape).astype(np.float64)
    analytical = layer.backward(upstream)

    def scalar_fn() -> float:
        return float(np.sum(layer.forward(x) * upstream))

    numerical = finite_difference_gradient(x, scalar_fn, eps=1e-6)
    assert relative_error(analytical, numerical) < 1e-7
