import numpy as np

from puredl.text import ScaledDotProductAttention
from puredl.utils.gradcheck import finite_difference_gradient, relative_error


def test_scaled_dot_product_attention_backward_matches_finite_difference():
    rng = np.random.default_rng(13)
    attention = ScaledDotProductAttention(causal=False)
    q = rng.normal(size=(1, 1, 2, 2)).astype(np.float64) * 0.3
    k = rng.normal(size=q.shape).astype(np.float64) * 0.3
    v = rng.normal(size=q.shape).astype(np.float64) * 0.3
    upstream = rng.normal(size=q.shape).astype(np.float64)

    attention.forward(q, k, v)
    grad_q, grad_k, grad_v = attention.backward(upstream)

    def scalar_fn():
        return float(np.sum(attention.forward(q, k, v) * upstream))

    numeric_q = finite_difference_gradient(q, scalar_fn, eps=1e-6)
    numeric_k = finite_difference_gradient(k, scalar_fn, eps=1e-6)
    numeric_v = finite_difference_gradient(v, scalar_fn, eps=1e-6)

    assert relative_error(grad_q, numeric_q) < 2e-6
    assert relative_error(grad_k, numeric_k) < 2e-6
    assert relative_error(grad_v, numeric_v) < 2e-6


def test_causal_attention_cannot_read_future_values():
    attention = ScaledDotProductAttention(causal=True)
    q = np.ones((1, 1, 3, 2), dtype=np.float64)
    k = np.ones_like(q)
    v1 = np.array([[[[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]]]])
    v2 = v1.copy()
    v2[:, :, 2] = np.array([-1000.0, 5000.0])

    out1 = attention.forward(q, k, v1)
    out2 = attention.forward(q, k, v2)
    np.testing.assert_allclose(out1[:, :, :2], out2[:, :, :2], atol=1e-12)
