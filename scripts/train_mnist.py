"""Train the NumPy MLP on MNIST.

Dataset download and IDX parsing use Python's standard library.
The model, loss, optimizer, forward pass, and backward pass use NumPy only.
"""
from __future__ import annotations

import argparse

import numpy as np

from puredl.data import load_mnist
from puredl.losses import CrossEntropyLoss
from puredl.optim import Adam
from puredl.vision import MLPClassifier


def iterate_batches(x, y, batch_size, rng):
    order = rng.permutation(len(x))
    for start in range(0, len(x), batch_size):
        idx = order[start : start + batch_size]
        yield x[idx], y[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    args = parser.parse_args()

    x_train, y_train, x_test, y_test = load_mnist(args.data_dir, download=True, flatten=True)
    if args.limit_train:
        x_train, y_train = x_train[: args.limit_train], y_train[: args.limit_train]
    if args.limit_test:
        x_test, y_test = x_test[: args.limit_test], y_test[: args.limit_test]

    rng = np.random.default_rng(1337)
    model = MLPClassifier(784, 256, 10, rng=rng)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        seen = 0
        for xb, yb in iterate_batches(x_train, y_train, args.batch_size, rng):
            optimizer.zero_grad()
            logits = model.forward(xb)
            loss = criterion.forward(logits, yb)
            model.backward(criterion.backward())
            optimizer.step()
            total_loss += loss * len(xb)
            seen += len(xb)

        model.eval()
        test_logits = model.forward(x_test)
        test_acc = float((test_logits.argmax(axis=1) == y_test).mean())
        print(
            f"epoch={epoch + 1:02d} train_loss={total_loss / seen:.4f} "
            f"test_acc={test_acc:.4f}"
        )


if __name__ == "__main__":
    main()
