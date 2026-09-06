import numpy as np

from puredl.text import DecoderBlock
from puredl.utils.gradcheck import finite_difference_gradient, relative_error


def test_decoder_block_residual_backward_matches_finite_difference():
    rng = np.random.default_rng(15)
    block = DecoderBlock(4, 2, 6, dropout=0.0, rng=rng, dtype=np.float64)
    x = rng.normal(size=(1, 2, 4)).astype(np.float64) * 0.1
    upstream = rng.normal(size=x.shape).astype(np.float64) * 0.1
    block.forward(x)
    analytic = block.backward(upstream)

    def scalar_fn():
        return float(np.sum(block.forward(x) * upstream))

    numeric = finite_difference_gradient(x, scalar_fn, eps=1e-6)
    assert relative_error(analytic, numeric) < 1e-5
