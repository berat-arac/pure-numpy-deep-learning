import numpy as np

from puredl.losses import CrossEntropyLoss
from puredl.text import CharLanguageModel, CharVocabulary, generate_text, language_model_loss


def test_rnn_language_model_forward_backward_and_sampling():
    rng = np.random.default_rng(4)
    vocab = CharVocabulary.from_text("abc \n")
    model = CharLanguageModel(vocab.size, 5, 7, cell="rnn", rng=rng)
    tokens = vocab.encode("ab cab\n").reshape(1, -1)
    logits = model.forward(tokens[:, :-1])
    loss, grad = language_model_loss(CrossEntropyLoss(), logits, tokens[:, 1:])
    assert np.isfinite(loss)
    model.backward(grad)
    assert all(np.all(np.isfinite(p.grad)) for p in model.parameters())
    generated = generate_text(model, vocab, "a", max_new_chars=8, rng=rng)
    assert generated.startswith("a")
    assert len(generated) == 9


def test_lstm_language_model_forward_backward():
    rng = np.random.default_rng(5)
    vocab = CharVocabulary.from_text("abc ")
    model = CharLanguageModel(vocab.size, 4, 6, cell="lstm", rng=rng)
    tokens = vocab.encode("ab cab").reshape(1, -1)
    logits = model.forward(tokens[:, :-1])
    loss, grad = language_model_loss(CrossEntropyLoss(), logits, tokens[:, 1:])
    assert np.isfinite(loss)
    model.backward(grad)
    assert all(np.all(np.isfinite(p.grad)) for p in model.parameters())
