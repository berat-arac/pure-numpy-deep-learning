import numpy as np

from puredl.layers import BatchNorm2D
from puredl.utils import finite_difference_gradient, relative_error


def test_batchnorm2d_input_gradient_passes_finite_difference_check():
    rng = np.random.default_rng(404)
    x = rng.normal(size=(2, 2, 2, 2)).astype(np.float64)
    upstream = rng.normal(size=x.shape).astype(np.float64)
    layer = BatchNorm2D(2, eps=1e-5, momentum=0.0, dtype=np.float64)
    layer.gamma.data[...] = rng.normal(size=2)
    layer.beta.data[...] = rng.normal(size=2)

    layer.forward(x)
    analytical = layer.backward(upstream)

    def scalar_fn() -> float:
        return float(np.sum(layer.forward(x) * upstream))

    numerical = finite_difference_gradient(x, scalar_fn, eps=1e-6)
    assert relative_error(analytical, numerical) < 1e-7
