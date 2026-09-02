# Vision Roadmap

This file tracks the vision side of the Pure NumPy Deep Learning project.

## Project rule

All neural network execution and training code is implemented with NumPy. No PyTorch, TensorFlow, JAX, CuPy, or automatic differentiation library is used.

## Phase 1: training foundations

Status: complete

Built:

- parameter and gradient storage
- reusable modules and nested modules
- Linear layer
- activation functions
- cross entropy loss
- SGD and Adam
- finite difference gradient checking
- MNIST data loading and MLP training

Why this phase existed:

Before implementing convolutional networks, the project needed a small but complete training path with manual backward propagation and optimizer updates.

## Phase 2: convolutional network stack

Status: complete

Built:

- Conv2D
- MaxPool2D
- BatchNorm2D
- GlobalAveragePool2D
- deeper CNN
- CIFAR-10 loading and augmentation
- learning rate schedulers
- checkpoint save and resume

Why this phase existed:

This created the reusable vision stack needed for deeper architectures and verified that a fully manual NumPy CNN could learn on a real image dataset.

Development gate on a 10,000 image training subset and 2,000 image validation subset:

```text
train_loss: 1.7621 -> 0.9210
train_acc:  34.67% -> 67.72%
val_loss:   1.6529 -> 0.9269
val_acc:    39.15% -> 66.20%
```

## Phase 3: efficient convolution and residual paths

Status: complete

Built:

- BasicResidualBlock
- explicit skip-gradient validation
- DepthwiseConv2D
- PointwiseConv2D
- DepthwiseSeparableConv2D
- depthwise input and weight gradient checks
- standard versus separable convolution benchmark

Why this phase existed:

EfficientNet depends on residual paths and efficient channel-separated convolution. These operations were implemented and tested independently before combining them into larger blocks.

Local benchmark result:

```text
standard params:  4,608
separable params:   656
parameter reduction: 85.76%
runtime ratio, separable / standard: 0.937
```

## Phase 4: EfficientNet building blocks

Status: complete

Built:

- SqueezeExcitation
- StochasticDepth
- MBConv
- residual and non-residual MBConv paths
- MiniEfficientNet
- MiniEfficientNet checkpoint support
- integrated MiniEfficientNet learning tests

Why this phase existed:

A full EfficientNet contains several interacting mechanisms. Building a smaller model first makes it possible to verify that the complete MBConv training path works before increasing depth and channel count.

Development gate on a 10,000 image training subset and 2,000 image validation subset:

```text
parameters: 44,670
train_loss: 1.8950 -> 0.9943
train_acc:  27.52% -> 63.93%
val_loss:   1.6692 -> 0.9062
val_acc:    37.95% -> 66.05%
```

## Phase 5: EfficientNet-B0

Status: in progress

Planned:

- B0 stage configuration
- repeated MBConv stage builder
- width scaling
- depth scaling
- channel rounding
- classifier dropout
- progressive drop-path scheduling
- full topology tests
- CIFAR-resolution training run
- CPU profiling
- memory and kernel optimization where measurements justify it

Why this phase exists:

This is the final vision milestone. It turns the independently tested components into a complete EfficientNet-B0-style training system while preserving the NumPy-only constraint.

ImageNet-scale training is not a requirement. The goal is a complete and trainable implementation with transparent CPU behavior.
