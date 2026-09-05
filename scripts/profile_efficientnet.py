from __future__ import annotations

import argparse
import time

import numpy as np

from puredl.losses import CrossEntropyLoss
from puredl.vision import EfficientNet


def timed(fn):
    start = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--width-mult", type=float, default=1.0)
    parser.add_argument("--depth-mult", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    model = EfficientNet(
        num_classes=10,
        width_mult=args.width_mult,
        depth_mult=args.depth_mult,
        dropout_rate=0.2,
        drop_path_rate=0.2,
        stem_stride=1,
        rng=rng,
    )
    model.train()
    x = rng.normal(size=(args.batch_size, 3, 32, 32)).astype(np.float32)
    labels = rng.integers(0, 10, size=args.batch_size, dtype=np.int64)
    criterion = CrossEntropyLoss()

    parameter_count = sum(parameter.data.size for parameter in model.parameters())
    parameter_bytes = sum(parameter.data.nbytes + parameter.grad.nbytes for parameter in model.parameters())
    buffer_bytes = sum(buffer.nbytes for _, buffer in model.named_buffers())
    print(f"parameters={parameter_count:,}")
    print(f"blocks={model.total_blocks}")
    print(f"parameter_and_grad_memory_mb={parameter_bytes / 1024**2:.2f}")
    print(f"persistent_buffer_memory_mb={buffer_bytes / 1024**2:.2f}")

    times: list[tuple[str, float]] = []
    out, dt = timed(lambda: model.stem.forward(x))
    times.append(("forward_stem", dt))
    for index, stage in enumerate(model.stages, start=1):
        out, dt = timed(lambda stage=stage, out=out: stage.forward(out))
        times.append((f"forward_stage_{index}", dt))
    out, dt = timed(lambda: model.head.forward(out))
    times.append(("forward_head", dt))
    out, dt = timed(lambda: model.pool.forward(out))
    times.append(("forward_pool", dt))
    out, dt = timed(lambda: model.dropout.forward(out))
    times.append(("forward_dropout", dt))
    logits, dt = timed(lambda: model.classifier.forward(out))
    times.append(("forward_classifier", dt))

    loss = criterion.forward(logits, labels)
    grad = criterion.backward()
    grad, dt = timed(lambda: model.classifier.backward(grad))
    times.append(("backward_classifier", dt))
    grad, dt = timed(lambda: model.dropout.backward(grad))
    times.append(("backward_dropout", dt))
    grad, dt = timed(lambda: model.pool.backward(grad))
    times.append(("backward_pool", dt))
    grad, dt = timed(lambda: model.head.backward(grad))
    times.append(("backward_head", dt))
    for index, stage in reversed(list(enumerate(model.stages, start=1))):
        grad, dt = timed(lambda stage=stage, grad=grad: stage.backward(grad))
        times.append((f"backward_stage_{index}", dt))
    _, dt = timed(lambda: model.stem.backward(grad))
    times.append(("backward_stem", dt))

    print(f"loss={loss:.4f}")
    for name, seconds in times:
        print(f"{name}={seconds:.6f}s")
    print(f"total_profiled_time={sum(seconds for _, seconds in times):.6f}s")


if __name__ == "__main__":
    main()
