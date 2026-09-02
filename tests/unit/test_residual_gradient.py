import numpy as np

from puredl.layers import BasicResidualBlock


def test_identity_skip_gradient_is_not_lost():
    rng = np.random.default_rng(44)
    block = BasicResidualBlock(2, 2, rng=rng, dtype=np.float64)
    block.conv1.weight.data.fill(0.0)
    block.conv2.weight.data.fill(0.0)
    x = np.full((2, 2, 3, 3), 1.5, dtype=np.float64)
    upstream = rng.normal(size=x.shape)

    out = block.forward(x)
    grad_x = block.backward(upstream)

    np.testing.assert_allclose(out, x, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(grad_x, upstream, rtol=1e-10, atol=1e-10)
