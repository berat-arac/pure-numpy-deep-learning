import numpy as np

from puredl.vision import MiniEfficientNet


def test_mini_efficientnet_state_dict_round_trip_includes_batchnorm_buffers():
    x_rng = np.random.default_rng(900)
    x = x_rng.normal(size=(2, 3, 32, 32)).astype(np.float32)

    source = MiniEfficientNet(
        width=0.25,
        drop_path_rate=0.0,
        rng=np.random.default_rng(901),
    )
    source.train()
    source.forward(x)
    state = source.state_dict()

    restored = MiniEfficientNet(
        width=0.25,
        drop_path_rate=0.0,
        rng=np.random.default_rng(902),
    )
    restored.load_state_dict(state)

    source.eval()
    restored.eval()
    np.testing.assert_allclose(
        source.forward(x),
        restored.forward(x),
        rtol=1e-6,
        atol=1e-6,
    )
    assert any(name.endswith("running_mean") for name in state)
    assert any(name.endswith("running_var") for name in state)
