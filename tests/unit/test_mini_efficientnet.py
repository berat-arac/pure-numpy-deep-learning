import numpy as np

from puredl.vision import MiniEfficientNet


def test_mini_efficientnet_forward_backward_shapes_are_finite():
    rng = np.random.default_rng(333)
    model = MiniEfficientNet(width=0.25, drop_path_rate=0.0, rng=rng)
    x = rng.normal(size=(2, 3, 32, 32)).astype(np.float32)
    logits = model.forward(x)
    grad_x = model.backward(np.ones_like(logits))
    assert logits.shape == (2, 10)
    assert grad_x.shape == x.shape
    assert np.isfinite(logits).all()
    assert np.isfinite(grad_x).all()
    assert all(np.isfinite(parameter.grad).all() for parameter in model.parameters())
