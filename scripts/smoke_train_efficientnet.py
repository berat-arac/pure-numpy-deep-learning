"""Integrated learning smoke test for the config-driven EfficientNet builder."""
from __future__ import annotations

import numpy as np

from puredl.losses import CrossEntropyLoss
from puredl.optim import Adam
from puredl.vision import EfficientNet


def make_dataset(samples: int = 128, seed: int = 29):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 0.10, size=(samples, 3, 32, 32)).astype(np.float32)
    y = rng.integers(0, 4, size=samples, dtype=np.int64)
    for index, label in enumerate(y):
        if label == 0:
            x[index, :, 4:10, 4:28] += 1.3
        elif label == 1:
            x[index, :, 22:28, 4:28] += 1.3
        elif label == 2:
            x[index, :, 4:28, 4:10] += 1.3
        else:
            x[index, :, 4:28, 22:28] += 1.3
    return x, y


def main():
    rng = np.random.default_rng(1337)
    x, y = make_dataset()
    model = EfficientNet(
        num_classes=4,
        width_mult=0.25,
        depth_mult=0.25,
        dropout_rate=0.0,
        drop_path_rate=0.0,
        stem_stride=1,
        rng=rng,
    )
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1.5e-3)
    batch_size = 16
    parameters = sum(parameter.data.size for parameter in model.parameters())
    print(f"parameters={parameters:,} blocks={model.total_blocks}")

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
