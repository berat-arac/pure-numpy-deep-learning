from __future__ import annotations

import numpy as np

from puredl.losses import CrossEntropyLoss
from puredl.optim import Adam
from puredl.text import TinyTransformerLM, language_model_loss


def main() -> None:
    rng = np.random.default_rng(21)
    model = TinyTransformerLM(
        vocab_size=6,
        context_length=12,
        d_model=24,
        num_heads=4,
        num_layers=2,
        ff_dim=48,
        dropout=0.0,
        rng=rng,
    )
    optimizer = Adam(model.parameters(), lr=0.005)
    loss_fn = CrossEntropyLoss()

    pattern = np.tile(np.arange(6, dtype=np.int64), 3)
    x_base = pattern[:12]
    y_base = pattern[1:13]
    x = np.stack([np.roll(x_base, i % 6) for i in range(24)])
    y = np.stack([np.roll(y_base, i % 6) for i in range(24)])

    params = sum(p.data.size for p in model.parameters())
    print(f"parameters={params:,} layers=2 heads=4 context=12")
    for epoch in range(1, 7):
        optimizer.zero_grad()
        logits = model.forward(x)
        loss, grad = language_model_loss(loss_fn, logits, y)
        model.backward(grad)
        optimizer.step()
        acc = float(np.mean(np.argmax(logits, axis=-1) == y))
        print(f"epoch={epoch} loss={loss:.4f} accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
