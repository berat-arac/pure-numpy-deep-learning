import numpy as np

from puredl.losses import CrossEntropyLoss
from puredl.optim import Adam
from puredl.text import TinyTransformerLM, language_model_loss


def test_tiny_transformer_shapes_and_state_roundtrip():
    rng = np.random.default_rng(16)
    model = TinyTransformerLM(11, 8, 8, 2, 2, 16, dropout=0.0, rng=rng)
    tokens = rng.integers(0, 11, size=(2, 6))
    logits = model.forward(tokens)
    assert logits.shape == (2, 6, 11)

    state = model.state_dict()
    clone = TinyTransformerLM(11, 8, 8, 2, 2, 16, dropout=0.0, rng=np.random.default_rng(99))
    clone.load_state_dict(state)
    np.testing.assert_allclose(model.forward(tokens), clone.forward(tokens), rtol=1e-6, atol=1e-6)


def test_tiny_transformer_can_fit_a_repeating_next_token_pattern():
    rng = np.random.default_rng(17)
    model = TinyTransformerLM(4, 8, 12, 3, 1, 24, dropout=0.0, rng=rng)
    optimizer = Adam(model.parameters(), lr=0.01)
    loss_fn = CrossEntropyLoss()
    base = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0], dtype=np.int64)
    x = np.stack([base[:8] for _ in range(8)])
    y = np.stack([base[1:] for _ in range(8)])

    initial = None
    final = None
    for _ in range(80):
        optimizer.zero_grad()
        logits = model.forward(x)
        loss, grad = language_model_loss(loss_fn, logits, y)
        if initial is None:
            initial = loss
        model.backward(grad)
        optimizer.step()
        final = loss

    assert initial is not None and final is not None
    assert final < initial * 0.15
    predictions = np.argmax(model.forward(x), axis=-1)
    assert np.mean(predictions == y) > 0.95
