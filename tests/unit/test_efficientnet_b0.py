import numpy as np

from puredl.vision import (
    EFFICIENTNET_B0_STAGES,
    EfficientNet,
    EfficientNetB0,
    round_channels,
    round_repeats,
)


def test_b0_stage_layout_matches_expected_configuration():
    actual = [
        (
            cfg.expand_ratio,
            cfg.kernel_size,
            cfg.stride,
            cfg.in_channels,
            cfg.out_channels,
            cfg.repeats,
        )
        for cfg in EFFICIENTNET_B0_STAGES
    ]
    assert actual == [
        (1, 3, 1, 32, 16, 1),
        (6, 3, 2, 16, 24, 2),
        (6, 5, 2, 24, 40, 2),
        (6, 3, 2, 40, 80, 3),
        (6, 5, 1, 80, 112, 3),
        (6, 5, 2, 112, 192, 4),
        (6, 3, 1, 192, 320, 1),
    ]


def test_b0_has_expected_block_and_parameter_counts():
    model = EfficientNetB0(rng=np.random.default_rng(3))
    assert model.total_blocks == 16
    parameter_count = sum(parameter.data.size for parameter in model.parameters())
    assert parameter_count == 5_288_548


def test_scaling_rounds_channels_and_repeats():
    assert round_channels(32, 1.0) == 32
    assert round_channels(32, 0.5) == 16
    assert round_channels(24, 1.4) == 32
    assert round_repeats(3, 1.0) == 3
    assert round_repeats(3, 1.2) == 4


def test_drop_path_schedule_is_progressive():
    model = EfficientNet(
        num_classes=10,
        width_mult=0.25,
        depth_mult=0.25,
        drop_path_rate=0.2,
        stem_stride=1,
        rng=np.random.default_rng(4),
    )
    probabilities = [block.drop_prob for block in model.blocks]
    assert probabilities[0] == 0.0
    assert all(a <= b for a, b in zip(probabilities, probabilities[1:]))
    assert probabilities[-1] < 0.2


def test_scaled_efficientnet_cifar_forward_backward_shapes():
    rng = np.random.default_rng(5)
    model = EfficientNet(
        num_classes=10,
        width_mult=0.25,
        depth_mult=0.25,
        dropout_rate=0.0,
        drop_path_rate=0.0,
        stem_stride=1,
        rng=rng,
    )
    x = rng.normal(size=(1, 3, 32, 32)).astype(np.float32)
    logits = model.forward(x)
    assert logits.shape == (1, 10)
    grad_x = model.backward(np.ones_like(logits))
    assert grad_x.shape == x.shape
    assert np.isfinite(logits).all()
    assert np.isfinite(grad_x).all()
