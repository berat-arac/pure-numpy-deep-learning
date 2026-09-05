import numpy as np

from puredl.layers import Dropout


def test_dropout_eval_is_identity():
    layer = Dropout(0.5, rng=np.random.default_rng(1))
    layer.eval()
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = layer.forward(x)
    grad = layer.backward(np.ones_like(x))
    np.testing.assert_array_equal(out, x)
    np.testing.assert_array_equal(grad, np.ones_like(x))


def test_dropout_training_uses_same_mask_in_backward():
    layer = Dropout(0.25, rng=np.random.default_rng(2))
    layer.train()
    x = np.ones((16, 8), dtype=np.float32)
    out = layer.forward(x)
    grad = layer.backward(np.ones_like(x))
    np.testing.assert_array_equal(out, grad)
