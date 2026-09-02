import numpy as np

from puredl.layers import SqueezeExcitation
from puredl.utils import finite_difference_gradient, relative_error


def test_squeeze_excitation_input_gradient_passes_finite_difference_check():
    rng = np.random.default_rng(222)
    x = rng.normal(size=(1, 2, 2, 2)).astype(np.float64)
    layer = SqueezeExcitation(
        2,
        squeeze_channels=1,
        rng=rng,
        dtype=np.float64,
    )
    out = layer.forward(x)
    upstream = rng.normal(size=out.shape).astype(np.float64)
    analytical = layer.backward(upstream)

    def scalar_fn() -> float:
        return float(np.sum(layer.forward(x) * upstream))

    numerical = finite_difference_gradient(x, scalar_fn, eps=1e-6)
    assert relative_error(analytical, numerical) < 1e-7
