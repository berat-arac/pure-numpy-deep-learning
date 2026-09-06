from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from puredl.losses import CrossEntropyLoss
from puredl.optim import Adam
from puredl.text import (
    CharLanguageModel,
    CharVocabulary,
    clip_grad_norm,
    generate_text,
    language_model_loss,
    load_text_corpus,
    sequence_batches,
)
from puredl.training.checkpoint import save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a pure NumPy character language model")
    parser.add_argument("--model", choices=("rnn", "lstm"), required=True)
    parser.add_argument("--dataset", choices=("natural", "local-prose", "repo"), default="natural")
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=48)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--max-chars", type=int, default=120000)
    parser.add_argument("--sample-len", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--prompt", type=str, default="The ")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/text"))
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
    if len(text) < args.seq_len * args.batch_size * 2:
        raise ValueError("corpus is too small for the requested training configuration")
    return text


def evaluate(model, token_ids, *, seq_len, batch_size, max_batches=20):
    loss_fn = CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    batch_count = 0
    for x, y in sequence_batches(
        token_ids,
        seq_len=seq_len,
        batch_size=batch_size,
        shuffle=False,
    ):
        logits = model.forward(x)
        batch, time, vocab = logits.shape
        loss = loss_fn.forward(logits.reshape(batch * time, vocab), y.reshape(-1))
        total_loss += loss
        total_correct += int(np.sum(np.argmax(logits, axis=-1) == y))
        total_tokens += y.size
        batch_count += 1
        if batch_count >= max_batches:
            break
    return total_loss / max(batch_count, 1), total_correct / max(total_tokens, 1)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    rng = np.random.default_rng(args.seed)
    text = load_text(args, root)
    vocab = CharVocabulary.from_text(text)
    ids = vocab.encode(text)

    split = max(args.seq_len + 1, int(0.9 * len(ids)))
    split = min(split, len(ids) - args.seq_len - 1)
    train_ids = ids[:split]
    val_ids = ids[split:]

    model = CharLanguageModel(
        vocab.size,
        args.embedding_dim,
        args.hidden_size,
        cell=args.model,
        rng=rng,
    )
    optimizer = Adam(model.parameters(), lr=args.lr)
    loss_fn = CrossEntropyLoss()

    parameter_count = sum(parameter.data.size for parameter in model.parameters())
    corpus_name = str(args.corpus) if args.corpus is not None else args.dataset
    print(
        f"model={args.model} parameters={parameter_count:,} vocab={vocab.size} "
        f"chars={len(text):,} train_chars={len(train_ids):,} val_chars={len(val_ids):,} "
        f"corpus={corpus_name}"
    )

    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        batches = 0
        grad_norm_sum = 0.0

        for x, y in sequence_batches(
            train_ids,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            shuffle=True,
            rng=rng,
        ):
            optimizer.zero_grad()
            logits = model.forward(x)
            loss, grad_logits = language_model_loss(loss_fn, logits, y)
            model.backward(grad_logits)
            grad_norm = clip_grad_norm(model.parameters(), args.clip_norm)
            optimizer.step()

            total_loss += loss
            total_correct += int(np.sum(np.argmax(logits, axis=-1) == y))
            total_tokens += y.size
            grad_norm_sum += grad_norm
            batches += 1

        val_loss, val_acc = evaluate(
            model,
            val_ids,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
        )
        elapsed = time.perf_counter() - start
        train_loss = total_loss / max(batches, 1)
        train_acc = total_correct / max(total_tokens, 1)
        mean_grad_norm = grad_norm_sum / max(batches, 1)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"grad_norm={mean_grad_norm:.3f} sec={elapsed:.1f}"
        )

        if args.sample_len > 0:
            try:
                sample = generate_text(
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"char_{args.model}"
    checkpoint_path = args.output_dir / f"{stem}.npz"
    metadata_path = args.output_dir / f"{stem}.json"
    save_checkpoint(checkpoint_path, model=model, optimizer=optimizer, epoch=args.epochs)
    metadata = {
        "cell": args.model,
        "embedding_dim": args.embedding_dim,
        "hidden_size": args.hidden_size,
        "dataset": args.dataset,
        "vocab": vocab.to_dict(),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"checkpoint={checkpoint_path}")
    print(f"metadata={metadata_path}")


if __name__ == "__main__":
    main()
