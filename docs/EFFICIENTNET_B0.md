# EfficientNet-B0

This document records the final Vision milestone of the project.

## What is implemented

The repository contains a config-driven EfficientNet builder rather than a manually listed sequence of MBConv blocks.

The B0 configuration contains seven MBConv stages with the expected channel counts, kernel sizes, expansion ratios, strides, and repeat counts.

The builder supports:

- width scaling
- depth scaling
- channel rounding
- repeat rounding
- classifier dropout
- progressive stochastic depth across blocks
- standard B0 stem behavior for larger images
- a CIFAR-friendly stem with stride 1

The NumPy-only rule is unchanged. PyTorch, TensorFlow, JAX, CuPy, and automatic differentiation are not runtime or test dependencies.

## Why the builder matters

MiniEfficientNet verified that MBConv, squeeze-and-excitation, depthwise convolution, normalization, residual paths, and stochastic depth could train together.

The full builder adds architecture-level construction. EfficientNet is a repeated stage model, so the network is generated from configuration instead of hand-writing every block. This keeps width scaling, depth scaling, channel rounding, repeat counts, and drop-path progression explicit and testable.

## B0 stage layout

| Stage | Expansion | Kernel | Stride | Input channels | Output channels | Repeats |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 3 | 1 | 32 | 16 | 1 |
| 2 | 6 | 3 | 2 | 16 | 24 | 2 |
| 3 | 6 | 5 | 2 | 24 | 40 | 2 |
| 4 | 6 | 3 | 2 | 40 | 80 | 3 |
| 5 | 6 | 5 | 1 | 80 | 112 | 3 |
| 6 | 6 | 5 | 2 | 112 | 192 | 4 |
| 7 | 6 | 3 | 1 | 192 | 320 | 1 |

The canonical B0 model has 16 MBConv blocks.

With a 1,000-class classifier:

```text
parameters=5,288,548
```

With the CIFAR-10 classifier:

```text
parameters=4,020,358
```

## CIFAR adaptation

The original B0 stem downsamples immediately because it targets much larger images. CIFAR-10 images are only 32 by 32 pixels.

For CIFAR experiments, the model keeps the B0 stages and channel layout but changes only the stem stride from 2 to 1. This preserves more spatial information at the start of the network while keeping the EfficientNet block structure intact.

This is an explicit input-resolution adaptation. It is not presented as an ImageNet benchmark configuration.

## Full-model CPU profile

A complete forward and backward profile was run with batch size 1 on the CIFAR-10 B0 configuration:

```text
parameters=4,020,358
blocks=16
parameter_and_grad_memory_mb=30.67
persistent_buffer_memory_mb=0.16
total_profiled_time=0.084933s
```

The profiler reports the stem, every stage, head, pooling, dropout, classifier, and their backward paths separately so later optimization work can be driven by measurements rather than assumptions.

## Trainability diagnostic

A small-data memorization diagnostic was used to answer one narrow question: can the complete NumPy B0 forward and backward path optimize a fixed dataset end to end?

For this diagnostic only:

- data augmentation was disabled
- classifier dropout was disabled
- stochastic depth was disabled
- weight decay was disabled
- the learning rate was fixed
- Adam was used
- the same 256 CIFAR-10 images were reused every epoch

Observed fit accuracy:

```text
epoch 1:   8.20%
epoch 5:  92.19%
epoch 10: 96.09%
best:     99.61%
```

This is not a generalization result. It is a controlled end-to-end optimization test.

## Normal training sanity gate

A separate run kept augmentation, dropout, stochastic depth, and weight decay enabled on a 2,000-image training subset and 1,000-image validation subset.

```text
train_loss: 2.4674 -> 2.0482
train_acc:  16.75% -> 24.50%
val_loss:   3.2895 -> 2.2256
val_acc:    18.30% -> 23.40%
```

The purpose of this run was to verify normal training behavior after the diagnostic mode, not to establish a CIFAR-10 benchmark.

## Completion status

EfficientNet-B0 is complete as the final Vision architecture milestone.

The project does not require ImageNet-scale training. The completed target is a config-driven B0 architecture with full NumPy forward and backward propagation, CPU profiling, checkpointable training, and demonstrated end-to-end trainability.
