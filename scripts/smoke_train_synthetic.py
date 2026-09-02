"""Fast integrated training smoke test using a NumPy-generated dataset."""
from __future__ import annotations

import numpy as np

from puredl.losses import CrossEntropyLoss
from puredl.optim import Adam
from puredl.vision import DeepCNN


def make_dataset(samples: int = 256, seed: int = 7):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 0.15, size=(samples, 3, 32, 32)).astype(np.float32)
    y = rng.integers(0, 10, size=samples, dtype=np.int64)
    for i, label in enumerate(y):
        row = 2 + (label // 5) * 14
        col = 2 + (label % 5) * 6
        x[i, :, row : row + 10, col : col + 5] += 1.5
    return x, y


def main():
    rng = np.random.default_rng(1337)
    x, y = make_dataset()
    model = DeepCNN(width=0.25, rng=rng)
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=2e-3)
    batch_size = 32

    for epoch in range(3):
        order = rng.permutation(len(x))
        loss_sum = 0.0
        correct = 0
        for start in range(0, len(x), batch_size):
            ids = order[start : start + batch_size]
            xb, yb = x[ids], y[ids]
            optimizer.zero_grad()
            logits = model.forward(xb)
            loss = loss_fn.forward(logits, yb)
            model.backward(loss_fn.backward())
            optimizer.step()
            loss_sum += loss * len(xb)
            correct += int((logits.argmax(axis=1) == yb).sum())
        print(
            f"epoch={epoch + 1} loss={loss_sum / len(x):.4f} "
            f"accuracy={correct / len(x):.4f}"
        )


if __name__ == "__main__":
    main()
