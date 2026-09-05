# Pure NumPy Deep Learning

[![CI](https://github.com/berat-arac/pure-numpy-deep-learning/actions/workflows/ci.yml/badge.svg)](https://github.com/berat-arac/pure-numpy-deep-learning/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A from-scratch deep learning project implemented with NumPy only.

No PyTorch, TensorFlow, JAX, CuPy, or automatic differentiation is used. Forward passes, backward passes, optimizer updates, normalization state, checkpointing, and training loops are implemented directly in NumPy and run on CPU.

The Vision track now progresses from a basic MLP to a full EfficientNet-B0 implementation.

## Vision track status

Completed:

- MLP training on MNIST
- Conv2D and MaxPool2D with manual backward propagation
- BatchNorm2D with training and evaluation state
- deeper CNN training on CIFAR-10
- residual blocks
- depthwise and pointwise convolution
- depthwise separable convolution
- squeeze-and-excitation
- stochastic depth
- MBConv blocks
- MiniEfficientNet
- config-driven EfficientNet builder
- canonical EfficientNet-B0 stage layout
- width and depth scaling
- channel and repeat rounding
- classifier dropout
- progressive drop-path scheduling
- full B0 forward and backward propagation
- CPU stage profiler
- full-model trainability diagnostics

The next project track is text modeling with RNN, LSTM, attention, and Transformer implementations.

## Why this project exists

High-level frameworks make deep learning practical, but they hide much of the execution path. This project removes that abstraction so the individual operations, state transitions, gradients, and model-building decisions remain visible.

The goal is not to compete with optimized frameworks in speed or benchmark accuracy. The goal is to implement the important mechanisms directly, verify that they work, and train real models without relying on a neural network framework.

## What was built and why

| Component | What was implemented | Why it is here |
| --- | --- | --- |
| Parameter and Module | Parameters, gradients, nested modules, persistent buffers, state loading | Provides a reusable foundation without building a full framework clone |
| Linear | Dense forward and backward passes | Establishes the first complete trainable network path |
| Conv2D | NumPy convolution with manual input, weight, and bias gradients | Provides the core operation for the vision track |
| MaxPool2D | Forward and backward pooling | Adds standard CNN downsampling while keeping gradient flow explicit |
| BatchNorm2D | Training state, evaluation state, running statistics, backward pass | Supports deeper CNN and MBConv training |
| GlobalAveragePool2D | Spatial reduction before classification | Avoids large dense classifier heads |
| SGD and Adam | Optimizer state and parameter updates | Keeps training independent from external ML frameworks |
| LR schedulers | Step and cosine schedules | Supports controlled longer CPU experiments |
| Checkpoints | Model, optimizer, scheduler, epoch, and metric state | Allows experiments to resume safely |
| Residual block | Main branch and explicit skip-gradient path | Validates residual gradient flow before MBConv |
| DepthwiseConv2D | One spatial convolution per input channel | Adds the efficient spatial operation used by MBConv |
| PointwiseConv2D | Explicit 1x1 channel mixing | Supports MBConv expansion and projection |
| SqueezeExcitation | Learned channel recalibration | Implements the channel gating used inside EfficientNet |
| StochasticDepth | Per-sample residual branch dropping during training | Adds the regularization behavior used by deeper EfficientNet stacks |
| MBConv | Expansion, depthwise convolution, SE, projection, normalization, residual path | Implements the core EfficientNet block |
| MiniEfficientNet | Small MBConv network for CIFAR-scale experiments | Verifies the complete EfficientNet-style stack before full B0 |
| EfficientNet builder | Stage configuration, repeated MBConv construction, scaling, dropout, progressive drop path | Builds EfficientNet from reusable configuration instead of hand-written blocks |
| EfficientNet-B0 | Canonical seven-stage B0 layout with 16 MBConv blocks | Completes the Vision track with the target architecture |

## Development results

These are engineering validation runs, not leaderboard claims.

### MNIST MLP

Configuration:

- 5,000 training examples
- 1,000 test examples
- 2 epochs

```text
epoch=01 train_loss=0.9249 test_acc=0.8560
epoch=02 train_loss=0.3502 test_acc=0.8870
```

### Deeper CNN on a CIFAR-10 subset

Configuration:

- 10,000 training images
- 2,000 validation images
- width multiplier 0.5
- 10 epochs

```text
train_loss: 1.7621 -> 0.9210
train_acc:  34.67% -> 67.72%
val_loss:   1.6529 -> 0.9269
val_acc:    39.15% -> 66.20%
```

### MiniEfficientNet on a CIFAR-10 subset

Configuration:

- 44,670 parameters
- 10,000 training images
- 2,000 validation images
- width multiplier 0.5
- stochastic depth enabled
- 10 epochs

```text
train_loss: 1.8950 -> 0.9943
train_acc:  27.52% -> 63.93%
val_loss:   1.6692 -> 0.9062
val_acc:    37.95% -> 66.05%
```

### Depthwise separable convolution benchmark

```text
standard params:  4,608
separable params:   656
parameter reduction: 85.76%
runtime ratio, separable / standard: 0.937
```

The parameter reduction is large, but runtime is measured separately because fewer parameters do not automatically guarantee faster NumPy execution.

## EfficientNet-B0 validation

The canonical B0 configuration contains seven stages and 16 MBConv blocks.

With a 1,000-class classifier:

```text
parameters=5,288,548
blocks=16
```

For CIFAR-10, the classifier is changed to 10 classes and the stem uses stride 1 so the 32x32 input is not downsampled immediately:

```text
parameters=4,020,358
blocks=16
```

A full forward and backward CPU profile completed successfully on the 10-class B0 configuration:

```text
parameter_and_grad_memory_mb=30.67
persistent_buffer_memory_mb=0.16
total_profiled_time=0.084933s
```

A controlled 256-image memorization diagnostic was used to verify end-to-end trainability. Augmentation and regularization were disabled only for this diagnostic.

```text
fit_acc epoch 1:  8.20%
fit_acc epoch 5: 92.19%
fit_acc epoch 10: 96.09%
best observed fit_acc: 99.61%
```

A separate normal training sanity run kept augmentation and regularization enabled and showed learning on a 2,000-image CIFAR-10 training subset:

```text
train_loss: 2.4674 -> 2.0482
train_acc:  16.75% -> 24.50%
val_loss:   3.2895 -> 2.2256
val_acc:    18.30% -> 23.40%
```

These runs are used to validate architecture construction, backward propagation, optimization, and general training behavior. ImageNet-scale training is intentionally outside the project scope.

## Validation approach

The repository does not use another neural network framework as a correctness dependency.

Validation includes:

- finite difference gradient checks
- deterministic small tensor tests
- shape and state invariants
- explicit residual branch tests
- checkpoint round-trip tests
- integrated loss reduction tests
- real MNIST and CIFAR-10 training runs
- full EfficientNet-B0 memorization diagnostics

Current gradient-check coverage includes Linear, Conv2D, DepthwiseConv2D, BatchNorm2D, SiLU, SqueezeExcitation, and CrossEntropyLoss.

## Repository structure

```text
.
├── .github/workflows/
├── docs/
│   ├── EFFICIENTNET_B0.md
│   ├── EFFICIENT_BLOCKS.md
│   ├── PROJECT_SCOPE.md
│   └── VALIDATION_STRATEGY.md
├── scripts/
│   ├── benchmark_separable_conv.py
│   ├── profile_efficientnet.py
│   ├── smoke_train_efficientnet.py
│   ├── smoke_train_mini_efficientnet.py
│   ├── smoke_train_synthetic.py
│   ├── train_cifar10.py
│   ├── train_efficientnet.py
│   ├── train_mini_efficientnet.py
│   └── train_mnist.py
├── src/puredl/
│   ├── core/
│   ├── data/
│   ├── layers/
│   ├── losses/
│   ├── optim/
│   ├── training/
│   ├── utils/
│   └── vision/
├── tests/unit/
├── LICENSE
├── README.md
├── VISION_ROADMAP.md
├── pyproject.toml
└── requirements.txt
```

## Installation

Python 3.11 or newer is recommended.

### Windows with Git Bash

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
```

### Run the tests

```bash
pytest -v
```

Current checkpoint:

```text
30 passed
```

## Useful commands

### MNIST

```bash
python scripts/train_mnist.py --epochs 2 --limit-train 5000 --limit-test 1000
```

### Deeper CNN on a CIFAR-10 subset

```bash
python scripts/train_cifar10.py \
  --epochs 10 \
  --batch-size 64 \
  --width 0.5 \
  --limit-train 10000 \
  --limit-test 2000
```

### MiniEfficientNet

```bash
python scripts/train_mini_efficientnet.py \
  --epochs 10 \
  --batch-size 32 \
  --width 0.5 \
  --drop-path 0.10 \
  --limit-train 10000 \
  --limit-test 2000
```

### Full EfficientNet-B0 CPU profile

```bash
python scripts/profile_efficientnet.py \
  --batch-size 1 \
  --width-mult 1.0 \
  --depth-mult 1.0
```

### EfficientNet-B0 trainability diagnostic

```bash
python scripts/train_efficientnet.py \
  --overfit-test \
  --epochs 30 \
  --batch-size 16 \
  --width-mult 1.0 \
  --depth-mult 1.0 \
  --limit-train 256 \
  --limit-test 256
```

## Project constraints

Intentionally not used:

- PyTorch
- TensorFlow
- JAX
- CuPy
- automatic differentiation
- CUDA-specific kernels
- distributed training

NumPy is not treated as a temporary placeholder for another backend. The CPU-only constraint is part of the project.

## Roadmap

The Vision track is complete through EfficientNet-B0.

See [VISION_ROADMAP.md](VISION_ROADMAP.md) for the completed progression and the planned Text track.

## License

MIT. See [LICENSE](LICENSE).
