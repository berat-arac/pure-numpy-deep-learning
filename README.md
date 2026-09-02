# Pure NumPy Deep Learning

[![CI](https://github.com/berat-arac/pure-numpy-deep-learning/actions/workflows/ci.yml/badge.svg)](https://github.com/berat-arac/pure-numpy-deep-learning/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A from-scratch deep learning project built to understand how neural networks work below high-level frameworks.

The current repository contains the **Vision track**. Every model operation, backward pass, optimizer update, normalization step, checkpoint state, and training loop is implemented with NumPy.

No PyTorch, TensorFlow, JAX, CuPy, or automatic differentiation is used.

## Current status

The project has progressed from a basic MLP to a trainable MiniEfficientNet-style model.

Completed:

- MLP training on MNIST
- Conv2D with manual backward propagation
- MaxPool2D
- BatchNorm2D with training and evaluation state
- Global average pooling
- deeper CNN training on CIFAR-10
- SGD, Adam, learning rate schedulers, and checkpoints
- residual blocks
- depthwise and pointwise convolutions
- depthwise separable convolution
- squeeze-and-excitation
- stochastic depth
- MBConv blocks
- MiniEfficientNet-style CIFAR model

Current work:

- full EfficientNet-B0 stage builder
- width and depth scaling
- drop-path scheduling across the network
- CPU profiling and optimization

## Why this project exists

Modern frameworks make neural network training convenient, but they also hide a large amount of the execution path. This project deliberately removes that abstraction.

The goal is not to outperform optimized frameworks. The goal is to build the important pieces directly, train them on real datasets, test the backward passes, and observe the engineering tradeoffs that appear when the framework is removed.

## What was built and why

| Component | What was implemented | Why it is here |
| --- | --- | --- |
| Parameter and Module | Parameters, gradients, nested modules, persistent buffers, state loading | Provides a small reusable foundation without building a full framework clone |
| Linear | Dense forward and backward passes | Establishes the basic training pipeline before convolutional models |
| Conv2D | NumPy convolution with manual input, weight, and bias gradients | Core operation for the vision track |
| MaxPool2D | Forward and backward pooling | Supports standard CNN downsampling while keeping gradient flow explicit |
| BatchNorm2D | Training state, evaluation state, running statistics, full backward pass | Required for stable deeper CNN and MBConv training |
| GlobalAveragePool2D | Spatial reduction before classification | Removes the need for large dense classifier heads |
| SGD and Adam | Optimizer state and parameter updates | Keeps training independent from external ML libraries |
| LR schedulers | Step and cosine schedules | Makes longer CIFAR experiments practical and reproducible |
| Checkpoints | Model, optimizer, scheduler, epoch, and metrics | Allows long CPU experiments to resume safely |
| Residual block | Main branch plus explicit skip-gradient handling | Verifies gradient flow through skip connections before MBConv |
| DepthwiseConv2D | One spatial convolution per input channel | Introduces the efficient convolution pattern used by EfficientNet |
| PointwiseConv2D | Explicit 1x1 channel mixing | Complements depthwise convolution and supports MBConv expansion and projection |
| SqueezeExcitation | Channel recalibration block | Adds the channel attention mechanism used inside EfficientNet blocks |
| StochasticDepth | Per-sample residual branch dropping during training | Adds the regularization behavior needed for deeper EfficientNet-style stacks |
| MBConv | Expansion, depthwise convolution, SE, projection, normalization, residual path | Combines the main EfficientNet building blocks into one trainable unit |
| MiniEfficientNet | Small CIFAR-scale MBConv network | Validates the complete block stack before building full EfficientNet-B0 |

## Real training results

These results are development gates, not leaderboard claims.

### MNIST MLP

Configuration:

- 5,000 training examples
- 1,000 test examples
- 2 epochs

Observed result:

```text
epoch=01 train_loss=0.9249 test_acc=0.8560
epoch=02 train_loss=0.3502 test_acc=0.8870
```

### Deeper CNN on CIFAR-10 subset

Configuration:

- 10,000 training images
- 2,000 validation images
- width multiplier 0.5
- 10 epochs

Observed progression:

```text
train_loss: 1.7621 -> 0.9210
train_acc:  34.67% -> 67.72%
val_loss:   1.6529 -> 0.9269
val_acc:    39.15% -> 66.20%
```

### MiniEfficientNet on CIFAR-10 subset

Configuration:

- 44,670 parameters
- 10,000 training images
- 2,000 validation images
- width multiplier 0.5
- stochastic depth enabled
- 10 epochs

Observed progression:

```text
train_loss: 1.8950 -> 0.9943
train_acc:  27.52% -> 63.93%
val_loss:   1.6692 -> 0.9062
val_acc:    37.95% -> 66.05%
```

### Depthwise separable convolution benchmark

A local CPU benchmark compared a standard convolution with the depthwise separable implementation used in this project.

```text
standard params:  4,608
separable params:   656
parameter reduction: 85.76%
runtime ratio, separable / standard: 0.937
```

The parameter reduction is large, but runtime is measured separately because fewer parameters do not automatically guarantee faster NumPy execution.

## Validation approach

The project does not depend on another neural network framework for correctness testing.

Backward implementations are checked with:

- finite difference gradient checks
- deterministic small tensor cases
- shape and state invariants
- explicit residual branch tests
- checkpoint round-trip tests
- integrated loss reduction tests
- real MNIST and CIFAR-10 training runs

Current gradient-check coverage includes Linear, Conv2D, DepthwiseConv2D, BatchNorm2D, SiLU, SqueezeExcitation, and CrossEntropyLoss.

## Repository structure

```text
.
├── docs/
│   ├── EFFICIENT_BLOCKS.md
│   └── VALIDATION_STRATEGY.md
├── scripts/
│   ├── benchmark_separable_conv.py
│   ├── smoke_train_mini_efficientnet.py
│   ├── smoke_train_synthetic.py
│   ├── train_cifar10.py
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
21 passed
```

## Training commands

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

### MiniEfficientNet on a CIFAR-10 subset

```bash
python scripts/train_mini_efficientnet.py \
  --epochs 10 \
  --batch-size 32 \
  --width 0.5 \
  --drop-path 0.10 \
  --limit-train 10000 \
  --limit-test 2000
```

### Resume MiniEfficientNet training

```bash
python scripts/train_mini_efficientnet.py \
  --epochs 10 \
  --batch-size 32 \
  --width 0.5 \
  --drop-path 0.10 \
  --limit-train 10000 \
  --limit-test 2000 \
  --resume
```

### Run the convolution benchmark

```bash
python scripts/benchmark_separable_conv.py
```

## Data and generated files

Datasets, virtual environments, caches, checkpoints, and local experiment outputs are intentionally excluded from Git.

CIFAR-10 and MNIST can be downloaded by the included dataset utilities when needed. Large local data files should not be committed to the repository.

## Next milestone

The next milestone is a full EfficientNet-B0 implementation built from the already tested MBConv components. The focus will be stage construction, scaling, trainability at CIFAR resolution, and CPU profiling while keeping the project strictly NumPy-only.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
