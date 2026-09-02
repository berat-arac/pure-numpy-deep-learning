import numpy as np

from puredl.layers import StochasticDepth


def test_stochastic_depth_eval_is_exact_identity():
    layer = StochasticDepth(0.4, rng=np.random.default_rng(9))
    layer.eval()
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2)
    upstream = np.ones_like(x)
    np.testing.assert_array_equal(layer.forward(x), x)
    np.testing.assert_array_equal(layer.backward(upstream), upstream)


def test_stochastic_depth_training_drops_whole_samples():
    layer = StochasticDepth(0.5, rng=np.random.default_rng(3))
    x = np.ones((8, 3, 2, 2), dtype=np.float32)
    out = layer.forward(x)
    per_sample = out[:, 0, 0, 0]
    assert set(np.unique(per_sample)).issubset({0.0, 2.0})
    for index in range(out.shape[0]):
        np.testing.assert_array_equal(out[index], np.full_like(out[index], per_sample[index]))
