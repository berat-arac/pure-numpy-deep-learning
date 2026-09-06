import numpy as np

from puredl.layers import Embedding
from puredl.utils.gradcheck import finite_difference_gradient, relative_error


def test_embedding_backward_matches_finite_difference():
    rng = np.random.default_rng(1)
    layer = Embedding(5, 3, rng=rng, dtype=np.float64)
    ids = np.array([[0, 1, 1], [4, 2, 0]], dtype=np.int64)
    upstream = rng.normal(size=(2, 3, 3))
    out = layer.forward(ids)
    layer.backward(upstream)
    analytic = layer.weight.grad.copy()

    def scalar_fn():
        return float(np.sum(layer.forward(ids) * upstream))

    numeric = finite_difference_gradient(layer.weight.data, scalar_fn, eps=1e-6)
    assert relative_error(analytic, numeric) < 1e-7
