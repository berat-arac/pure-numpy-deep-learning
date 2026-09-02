import numpy as np

from puredl.layers import Conv2D, DepthwiseSeparableConv2D


def test_depthwise_separable_uses_fewer_parameters_than_standard_conv():
    rng = np.random.default_rng(77)
    standard = Conv2D(16, 32, 3, padding=1, bias=False, rng=rng)
    separable = DepthwiseSeparableConv2D(
        16, 32, 3, padding=1, bias=False, rng=rng
    )
    standard_params = sum(parameter.data.size for parameter in standard.parameters())
    separable_params = sum(parameter.data.size for parameter in separable.parameters())
    assert separable_params < standard_params
    assert standard_params == 16 * 32 * 3 * 3
    assert separable_params == 16 * 3 * 3 + 16 * 32
