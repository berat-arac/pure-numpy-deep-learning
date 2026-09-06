from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from puredl.losses import CrossEntropyLoss
from puredl.optim import Adam
from puredl.text import (
    CharVocabulary,
    TinyTransformerLM,
    clip_grad_norm,
    generate_transformer_text,
    language_model_loss,
    load_text_corpus,
    sequence_batches,
)
from puredl.training.checkpoint import load_checkpoint, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a pure NumPy decoder-only character Transformer")
    parser.add_argument("--dataset", choices=("natural", "local-prose", "repo"), default="natural")
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--max-chars", type=int, default=200000)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=20)
    parser.add_argument("--sample-len", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--prompt", type=str, default="The ")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/text"))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_text(args: argparse.Namespace, root: Path) -> str:
    text = load_text_corpus(
        root,
        corpus_path=args.corpus,
        dataset=args.dataset,
        auto_download=True,
    )
    if args.max_chars > 0:
        text = text[: args.max_chars]
    minimum = args.context_length * args.batch_size * 2
    if len(text) < minimum:
        raise ValueError("corpus is too small for the requested training configuration")
    return text


def evaluate(model, token_ids, *, context_length, batch_size, max_batches):
    loss_fn = CrossEntropyLoss()
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    batches = 0
    try:
        for x, y in sequence_batches(
            token_ids,
            seq_len=context_length,
            batch_size=batch_size,
            shuffle=False,
        ):
            logits = model.forward(x)
            b, t, vocab = logits.shape
            loss = loss_fn.forward(logits.reshape(b * t, vocab), y.reshape(-1))
            total_loss += loss
            total_correct += int(np.sum(np.argmax(logits, axis=-1) == y))
            total_tokens += y.size
            batches += 1
            if max_batches > 0 and batches >= max_batches:
                break
    finally:
        if was_training:
            model.train()
    return total_loss / max(batches, 1), total_correct / max(total_tokens, 1)


def main() -> None:
    args = parse_args()
    if args.d_model % args.heads != 0:
        raise ValueError("d-model must be divisible by heads")
    root = Path(__file__).resolve().parents[1]
    rng = np.random.default_rng(args.seed)
    text = load_text(args, root)
    vocab = CharVocabulary.from_text(text)
    ids = vocab.encode(text)
    split = max(args.context_length + 1, int(0.9 * len(ids)))
    split = min(split, len(ids) - args.context_length - 1)
    train_ids = ids[:split]
    val_ids = ids[split:]

    model = TinyTransformerLM(
        vocab.size,
        args.context_length,
        args.d_model,
        args.heads,
        args.layers,
        args.ff_dim,
        dropout=args.dropout,
        rng=rng,
    )
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = CrossEntropyLoss()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "char_transformer.npz"
    metadata_path = args.output_dir / "char_transformer.json"

    start_epoch = 1
    if args.resume:
        if not checkpoint.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        state = load_checkpoint(checkpoint, model=model, optimizer=optimizer)
        start_epoch = int(state.get("epoch", 0)) + 1
        print(f"resumed_from_epoch={start_epoch - 1}")

    params = sum(p.data.size for p in model.parameters())
    corpus_name = str(args.corpus) if args.corpus is not None else args.dataset
    print(
        f"model=transformer parameters={params:,} vocab={vocab.size} chars={len(text):,} "
        f"context={args.context_length} d_model={args.d_model} heads={args.heads} layers={args.layers} "
        f"corpus={corpus_name}"
    )

    if start_epoch > args.epochs:
        print("checkpoint already reached requested epoch count")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        start = time.perf_counter()
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        grad_norm_sum = 0.0
        batches = 0
        for x, y in sequence_batches(
            train_ids,
            seq_len=args.context_length,
            batch_size=args.batch_size,
            shuffle=True,
            rng=rng,
        ):
            optimizer.zero_grad()
            logits = model.forward(x)
            loss, grad = language_model_loss(loss_fn, logits, y)
            model.backward(grad)
            grad_norm = clip_grad_norm(model.parameters(), args.clip_norm)
            optimizer.step()
            total_loss += loss
            total_correct += int(np.sum(np.argmax(logits, axis=-1) == y))
            total_tokens += y.size
            grad_norm_sum += grad_norm
            batches += 1
            if args.max_train_batches > 0 and batches >= args.max_train_batches:
                break

        val_loss, val_acc = evaluate(
            model,
            val_ids,
            context_length=args.context_length,
            batch_size=args.batch_size,
            max_batches=args.max_val_batches,
        )
        elapsed = time.perf_counter() - start
        print(
            f"epoch={epoch:03d} train_loss={total_loss/max(batches,1):.4f} "
            f"train_acc={total_correct/max(total_tokens,1):.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"grad_norm={grad_norm_sum/max(batches,1):.3f} sec={elapsed:.1f}"
        )
        if args.sample_len > 0:
            try:
                sample = generate_transformer_text(
                    model,
                    vocab,
                    args.prompt,
                    max_new_chars=args.sample_len,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    rng=rng,
                )
                print("sample:")
                print(sample.replace("\r", ""))
            except ValueError as exc:
                print(f"sample skipped: {exc}")

        save_checkpoint(checkpoint, model=model, optimizer=optimizer, epoch=epoch)
        metadata = {
            "model": "transformer",
            "context_length": args.context_length,
            "d_model": args.d_model,
            "heads": args.heads,
            "layers": args.layers,
            "ff_dim": args.ff_dim,
            "dropout": args.dropout,
            "dataset": args.dataset,
            "vocab": vocab.to_dict(),
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"checkpoint={checkpoint}")
    print(f"metadata={metadata_path}")


if __name__ == "__main__":
    main()
