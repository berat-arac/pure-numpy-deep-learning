from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from puredl.data import batches, load_cifar10
from puredl.losses import CrossEntropyLoss
from puredl.optim import CosineAnnealingLR, SGD
from puredl.training import load_checkpoint, save_checkpoint
from puredl.vision import MiniEfficientNet


def evaluate(model, x, y, batch_size):
    model.eval()
    criterion = CrossEntropyLoss()
    correct = 0
    total = 0
    loss_sum = 0.0
    rng = np.random.default_rng(0)
    for bx, by in batches(x, y, batch_size, rng, shuffle=False):
        logits = model.forward(bx)
        loss_sum += criterion.forward(logits, by) * len(bx)
        correct += int((logits.argmax(1) == by).sum())
        total += len(bx)
    return loss_sum / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.035)
    parser.add_argument("--width", type=float, default=0.5)
    parser.add_argument("--drop-path", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--checkpoint", default="checkpoints/cifar10_mini_efficientnet.npz"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    train_x, train_y, test_x, test_y = load_cifar10(args.data_dir, download=True)
    if args.limit_train:
        train_x, train_y = train_x[: args.limit_train], train_y[: args.limit_train]
    if args.limit_test:
        test_x, test_y = test_x[: args.limit_test], test_y[: args.limit_test]

    model = MiniEfficientNet(
        width=args.width,
        drop_path_rate=args.drop_path,
        rng=rng,
    )
    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.02,
    )
    criterion = CrossEntropyLoss()
    start_epoch = 0

    if args.resume and Path(args.checkpoint).exists():
        state = load_checkpoint(
            args.checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        start_epoch = int(state["epoch"]) + 1
        print(f"resumed from epoch {state['epoch']}")

    parameter_count = sum(parameter.data.size for parameter in model.parameters())
    print(f"parameters={parameter_count:,}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.perf_counter()
        correct = 0
        total = 0
        loss_sum = 0.0
        for bx, by in batches(
            train_x,
            train_y,
            args.batch_size,
            rng,
            shuffle=True,
            augment=True,
        ):
            optimizer.zero_grad()
            logits = model.forward(bx)
            loss = criterion.forward(logits, by)
            model.backward(criterion.backward())
            optimizer.step()
            loss_sum += loss * len(bx)
            correct += int((logits.argmax(1) == by).sum())
            total += len(bx)

        val_loss, val_acc = evaluate(model, test_x, test_y, args.batch_size)
        lr_used = optimizer.lr
        scheduler.step()
        metrics = {
            "train_loss": loss_sum / total,
            "train_acc": correct / total,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": lr_used,
        }
        save_checkpoint(
            args.checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metrics=metrics,
        )
        elapsed = time.perf_counter() - t0
        print(
            f"epoch={epoch + 1:03d} "
            f"train_loss={metrics['train_loss']:.4f} "
            f"train_acc={metrics['train_acc']:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} "
            f"lr={lr_used:.6f} sec={elapsed:.1f}"
        )


if __name__ == "__main__":
    main()
