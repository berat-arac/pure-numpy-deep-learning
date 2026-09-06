import numpy as np

from puredl.layers import GELU
from puredl.utils.gradcheck import finite_difference_gradient, relative_error


def test_gelu_backward_matches_finite_difference():
    rng = np.random.default_rng(12)
    layer = GELU()
    x = rng.normal(size=(2, 3)).astype(np.float64)
    upstream = rng.normal(size=x.shape)
    layer.forward(x)
    analytic = layer.backward(upstream)

    def scalar_fn():
        return float(np.sum(layer.forward(x) * upstream))

    numeric = finite_difference_gradient(x, scalar_fn, eps=1e-6)
    assert relative_error(analytic, numeric) < 1e-7
