"""Integrated NumPy-only learning smoke test for the MBConv stack."""
from __future__ import annotations

import numpy as np

from puredl.losses import CrossEntropyLoss
from puredl.optim import Adam
from puredl.vision import MiniEfficientNet


def make_dataset(samples: int = 128, seed: int = 17):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 0.10, size=(samples, 3, 24, 24)).astype(np.float32)
    y = rng.integers(0, 4, size=samples, dtype=np.int64)
    for index, label in enumerate(y):
        if label == 0:
            x[index, :, 3:8, 3:21] += 1.4
        elif label == 1:
            x[index, :, 16:21, 3:21] += 1.4
        elif label == 2:
            x[index, :, 3:21, 3:8] += 1.4
        else:
            x[index, :, 3:21, 16:21] += 1.4
    return x, y


def main():
    rng = np.random.default_rng(1337)
    x, y = make_dataset()
    model = MiniEfficientNet(
        num_classes=4,
        width=0.25,
        drop_path_rate=0.0,
        rng=rng,
    )
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1.5e-3)
    batch_size = 16
    parameters = sum(parameter.data.size for parameter in model.parameters())
    print(f"parameters={parameters:,}")

    for epoch in range(3):
        model.train()
        order = rng.permutation(len(x))
        loss_sum = 0.0
        correct = 0
        for start in range(0, len(x), batch_size):
            ids = order[start : start + batch_size]
            xb, yb = x[ids], y[ids]
            optimizer.zero_grad()
            logits = model.forward(xb)
            loss = criterion.forward(logits, yb)
            model.backward(criterion.backward())
            optimizer.step()
            loss_sum += loss * len(xb)
            correct += int((logits.argmax(axis=1) == yb).sum())
        print(
            f"epoch={epoch + 1} loss={loss_sum / len(x):.4f} "
            f"accuracy={correct / len(x):.4f}"
        )


if __name__ == "__main__":
    main()
