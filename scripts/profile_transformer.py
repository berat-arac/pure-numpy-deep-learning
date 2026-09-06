from __future__ import annotations

import argparse
import time

import numpy as np

from puredl.losses import CrossEntropyLoss
from puredl.text import TinyTransformerLM, language_model_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile a pure NumPy decoder-only Transformer")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=96)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=31)
    return parser.parse_args()


def timed(label, fn, timings):
    start = time.perf_counter()
    result = fn()
    timings[label] = time.perf_counter() - start
    return result


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    model = TinyTransformerLM(
        args.vocab_size,
        args.context_length,
        args.d_model,
        args.heads,
        args.layers,
        args.ff_dim,
        dropout=0.0,
        rng=rng,
    )
    tokens = rng.integers(0, args.vocab_size, size=(args.batch_size, args.context_length))
    targets = rng.integers(0, args.vocab_size, size=tokens.shape)
    loss_fn = CrossEntropyLoss()
    timings = {}

    parameters = model.parameters()
    param_count = sum(p.data.size for p in parameters)
    param_grad_bytes = sum(p.data.nbytes + p.grad.nbytes for p in parameters)
    print(f"parameters={param_count:,}")
    print(f"layers={args.layers} heads={args.heads} context={args.context_length} d_model={args.d_model}")
    print(f"parameter_and_grad_memory_mb={param_grad_bytes / 1024**2:.2f}")

    embedded = timed("forward_token_embedding", lambda: model.token_embedding.forward(tokens), timings)
    model._time = args.context_length
    x = embedded + model.position_embedding.data[: args.context_length][None, :, :]
    x = timed("forward_input_dropout", lambda: model.input_dropout.forward(x), timings)
    for index, block in enumerate(model.blocks, start=1):
        x = timed(f"forward_block_{index}", lambda block=block, x=x: block.forward(x), timings)
    x = timed("forward_final_norm", lambda: model.final_norm.forward(x), timings)
    logits = timed("forward_head", lambda: model.head.forward(x), timings)
    loss, grad = language_model_loss(loss_fn, logits, targets)
    print(f"loss={loss:.4f}")

    grad = timed("backward_head", lambda: model.head.backward(grad), timings)
    grad = timed("backward_final_norm", lambda: model.final_norm.backward(grad), timings)
    for index, block in reversed(list(enumerate(model.blocks, start=1))):
        grad = timed(f"backward_block_{index}", lambda block=block, grad=grad: block.backward(grad), timings)
    grad = timed("backward_input_dropout", lambda: model.input_dropout.backward(grad), timings)
    start = time.perf_counter()
    model.position_embedding.grad.fill(0)
    model.position_embedding.grad[: args.context_length] = grad.sum(axis=0)
    model.token_embedding.backward(grad)
    timings["backward_embeddings"] = time.perf_counter() - start

    for name, seconds in timings.items():
        print(f"{name}={seconds:.6f}s")
    print(f"total_profiled_time={sum(timings.values()):.6f}s")


if __name__ == "__main__":
    main()
