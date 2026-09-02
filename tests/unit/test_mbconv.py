import numpy as np

from puredl.vision import MBConv


def test_mbconv_preserves_shape_when_residual_is_enabled():
    rng = np.random.default_rng(101)
    block = MBConv(
        4,
        4,
        kernel_size=3,
        stride=1,
        expand_ratio=2,
        drop_prob=0.0,
        rng=rng,
    )
    x = rng.normal(size=(2, 4, 8, 8)).astype(np.float32)
    out = block.forward(x)
    grad = block.backward(np.ones_like(out))
    assert out.shape == x.shape
    assert grad.shape == x.shape
    assert all(np.isfinite(parameter.grad).all() for parameter in block.parameters())


def test_mbconv_stride_two_reduces_spatial_size_and_changes_channels():
    rng = np.random.default_rng(102)
    block = MBConv(
        4,
        8,
        kernel_size=3,
        stride=2,
        expand_ratio=2,
        drop_prob=0.0,
        rng=rng,
    )
    x = rng.normal(size=(2, 4, 8, 8)).astype(np.float32)
    out = block.forward(x)
    grad = block.backward(np.ones_like(out))
    assert out.shape == (2, 8, 4, 4)
    assert grad.shape == x.shape
