import numpy as np

from puredl.layers import LayerNorm
from puredl.utils.gradcheck import finite_difference_gradient, relative_error


def test_layernorm_full_backward_matches_finite_difference():
    rng = np.random.default_rng(11)
    layer = LayerNorm(3, eps=1e-5, dtype=np.float64)
    layer.weight.data[...] = rng.normal(1.0, 0.2, size=3)
    layer.bias.data[...] = rng.normal(0.0, 0.1, size=3)
    x = rng.normal(size=(2, 2, 3))
    upstream = rng.normal(size=x.shape)

    layer.forward(x)
    grad_x = layer.backward(upstream)
    analytic_w = layer.weight.grad.copy()
    analytic_b = layer.bias.grad.copy()

    def scalar_fn():
        return float(np.sum(layer.forward(x) * upstream))

    numeric_x = finite_difference_gradient(x, scalar_fn, eps=1e-6)
    numeric_w = finite_difference_gradient(layer.weight.data, scalar_fn, eps=1e-6)
    numeric_b = finite_difference_gradient(layer.bias.data, scalar_fn, eps=1e-6)

    assert relative_error(grad_x, numeric_x) < 2e-6
    assert relative_error(analytic_w, numeric_w) < 2e-6
    assert relative_error(analytic_b, numeric_b) < 2e-6
