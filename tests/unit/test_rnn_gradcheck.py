import numpy as np

from puredl.text import VanillaRNN
from puredl.utils.gradcheck import finite_difference_gradient, relative_error


def test_vanilla_rnn_bptt_matches_finite_difference():
    rng = np.random.default_rng(2)
    layer = VanillaRNN(2, 2, rng=rng, dtype=np.float64)
    x = rng.normal(size=(1, 3, 2)) * 0.2
    upstream = rng.normal(size=(1, 3, 2)) * 0.3
    out = layer.forward(x)
    grad_x, _ = layer.backward(upstream)
    analytic_wih = layer.weight_ih.grad.copy()
    analytic_whh = layer.weight_hh.grad.copy()
    analytic_b = layer.bias.grad.copy()

    def scalar_fn():
        return float(np.sum(layer.forward(x) * upstream))

    numeric_wih = finite_difference_gradient(layer.weight_ih.data, scalar_fn, eps=1e-6)
    numeric_whh = finite_difference_gradient(layer.weight_hh.data, scalar_fn, eps=1e-6)
    numeric_b = finite_difference_gradient(layer.bias.data, scalar_fn, eps=1e-6)
    numeric_x = finite_difference_gradient(x, scalar_fn, eps=1e-6)

    assert relative_error(analytic_wih, numeric_wih) < 1e-6
    assert relative_error(analytic_whh, numeric_whh) < 1e-6
    assert relative_error(analytic_b, numeric_b) < 1e-6
    assert relative_error(grad_x, numeric_x) < 1e-6
