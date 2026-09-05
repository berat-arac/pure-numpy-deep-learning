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

Development gate on a 10,000-image training subset and 2,000-image validation subset:

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

Development benchmark:

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

Development gate on a 10,000-image training subset and 2,000-image validation subset:

```text
parameters: 44,670
train_loss: 1.8950 -> 0.9943
train_acc:  27.52% -> 63.93%
val_loss:   1.6692 -> 0.9062
val_acc:    37.95% -> 66.05%
```

## Phase 5: EfficientNet-B0

Status: complete

Built:

- canonical seven-stage B0 configuration
- 16-block MBConv depth configuration
- repeated stage builder
- width scaling
- depth scaling
- channel rounding
- repeat scaling
- classifier dropout
- progressive drop-path scheduling
- standard B0 stem mode
- CIFAR-friendly stem mode
- full forward propagation
- full backward propagation
- stage-level CPU profiler
- training and checkpoint support
- controlled small-data memorization diagnostic

Architecture gate:

```text
B0 MBConv blocks: 16
B0 parameters with 1000 classes: 5,288,548
CIFAR-10 configuration parameters: 4,020,358
unit tests: 30 passed
```

Full B0 CPU profile gate:

```text
parameter_and_grad_memory_mb=30.67
persistent_buffer_memory_mb=0.16
total_profiled_time=0.084933s
```

Trainability gate on a fixed 256-image CIFAR-10 subset:

```text
fit_acc epoch 1:  8.20%
fit_acc epoch 5: 92.19%
fit_acc epoch 10: 96.09%
best observed fit_acc: 99.61%
```

A separate normal training sanity run kept augmentation and regularization enabled:

```text
train_loss: 2.4674 -> 2.0482
train_acc:  16.75% -> 24.50%
val_loss:   3.2895 -> 2.2256
val_acc:    18.30% -> 23.40%
```

Why this phase existed:

The final vision goal was not an ImageNet benchmark. It was to construct the full B0 architecture from configuration, run its complete forward and backward path with NumPy, verify that the model can be optimized end to end, and measure its CPU behavior.

Those gates are now complete.

## Vision completion

Status: complete through EfficientNet-B0

The Vision track now demonstrates a progression from a basic dense network to a full EfficientNet-B0 implementation while keeping the training path NumPy-only.

## Next track: text

Planned progression:

- vanilla RNN
- LSTM
- attention
- multi-head attention
- decoder-only Transformer

The same project rules remain in place: NumPy-only model execution, manual backward propagation, CPU training, and explicit validation of the implemented operations.
