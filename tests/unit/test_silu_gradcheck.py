import numpy as np

from puredl.layers import SiLU
from puredl.utils import finite_difference_gradient, relative_error


def test_silu_input_gradient_passes_finite_difference_check():
    rng = np.random.default_rng(505)
    x = rng.normal(size=(3, 4)).astype(np.float64)
    upstream = rng.normal(size=x.shape).astype(np.float64)
    layer = SiLU()

    layer.forward(x)
    analytical = layer.backward(upstream)

    def scalar_fn() -> float:
        return float(np.sum(layer.forward(x) * upstream))

    numerical = finite_difference_gradient(x, scalar_fn, eps=1e-6)
    assert relative_error(analytical, numerical) < 1e-8
