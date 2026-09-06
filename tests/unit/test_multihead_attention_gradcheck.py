import numpy as np

from puredl.text import MultiHeadSelfAttention
from puredl.utils.gradcheck import finite_difference_gradient, relative_error


def test_multihead_self_attention_input_gradient_matches_finite_difference():
    rng = np.random.default_rng(14)
    layer = MultiHeadSelfAttention(4, 2, causal=True, rng=rng, dtype=np.float64)
    x = rng.normal(size=(1, 3, 4)).astype(np.float64) * 0.15
    upstream = rng.normal(size=x.shape).astype(np.float64) * 0.2
    layer.forward(x)
    analytic = layer.backward(upstream)

    def scalar_fn():
        return float(np.sum(layer.forward(x) * upstream))

    numeric = finite_difference_gradient(x, scalar_fn, eps=1e-6)
    assert relative_error(analytic, numeric) < 5e-6
