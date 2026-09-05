from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from puredl.data import batches, load_cifar10
from puredl.losses import CrossEntropyLoss
from puredl.optim import Adam, CosineAnnealingLR, SGD
from puredl.training import load_checkpoint, save_checkpoint
from puredl.vision import EfficientNet


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


def apply_overfit_preset(args):
    """Turn the training script into a small-data memorization diagnostic."""
    args.dropout = 0.0
    args.drop_path = 0.0
    args.weight_decay = 0.0
    args.no_augment = True
    args.fixed_lr = True
    args.optimizer = "adam"
    args.lr = 1e-3
    if args.limit_train == 0:
        args.limit_train = 256
    if args.limit_test == 0:
        args.limit_test = 256
    if args.checkpoint is None:
        args.checkpoint = "checkpoints/cifar10_efficientnet_overfit.npz"
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default="sgd")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--width-mult", type=float, default=1.0)
    parser.add_argument("--depth-mult", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--drop-path", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--fixed-lr", action="store_true")
    parser.add_argument(
        "--overfit-test",
        action="store_true",
        help=(
            "Run a small fixed-data memorization diagnostic. This disables "
            "augmentation and regularization, uses Adam with a fixed 1e-3 "
            "learning rate, and reports final-model accuracy on the train subset."
        ),
    )
    args = parser.parse_args()

    if args.overfit_test:
        args = apply_overfit_preset(args)
    elif args.checkpoint is None:
        args.checkpoint = "checkpoints/cifar10_efficientnet.npz"

    rng = np.random.default_rng(args.seed)
    train_x, train_y, test_x, test_y = load_cifar10(args.data_dir, download=True)
    if args.limit_train:
        train_x, train_y = train_x[: args.limit_train], train_y[: args.limit_train]
    if args.limit_test:
        test_x, test_y = test_x[: args.limit_test], test_y[: args.limit_test]

    model = EfficientNet(
        num_classes=10,
        width_mult=args.width_mult,
        depth_mult=args.depth_mult,
        dropout_rate=args.dropout,
        drop_path_rate=args.drop_path,
        stem_stride=1,
        rng=rng,
    )

    if args.optimizer == "adam":
        optimizer = Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

    scheduler = None
    if not args.fixed_lr:
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
    mode = "overfit_test" if args.overfit_test else "train"
    schedule_name = "fixed" if scheduler is None else "cosine"
    print(
        f"parameters={parameter_count:,} blocks={model.total_blocks} "
        f"width_mult={args.width_mult:g} depth_mult={args.depth_mult:g}"
    )
    print(
        f"mode={mode} optimizer={args.optimizer} lr={optimizer.lr:.6f} "
        f"weight_decay={args.weight_decay:g} augmentation={not args.no_augment} "
        f"scheduler={schedule_name}"
    )

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
            augment=not args.no_augment,
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
        train_eval_loss = None
        train_eval_acc = None
        if args.overfit_test:
            train_eval_loss, train_eval_acc = evaluate(
                model, train_x, train_y, args.batch_size
            )

        lr_used = optimizer.lr
        if scheduler is not None:
            scheduler.step()

        metrics = {
            "train_loss": loss_sum / total,
            "train_acc": correct / total,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": lr_used,
        }
        if train_eval_loss is not None and train_eval_acc is not None:
            metrics["fit_loss"] = train_eval_loss
            metrics["fit_acc"] = train_eval_acc

        save_checkpoint(
            args.checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metrics=metrics,
        )
        elapsed = time.perf_counter() - t0

        extra = ""
        if train_eval_loss is not None and train_eval_acc is not None:
            extra = f" fit_loss={train_eval_loss:.4f} fit_acc={train_eval_acc:.4f}"
        print(
            f"epoch={epoch + 1:03d} "
            f"train_loss={metrics['train_loss']:.4f} "
            f"train_acc={metrics['train_acc']:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} "
            f"lr={lr_used:.6f}{extra} sec={elapsed:.1f}"
        )


if __name__ == "__main__":
    main()
