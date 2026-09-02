# Efficient Vision Blocks

This document explains the components added on the path from a standard CNN to EfficientNet.

## DepthwiseConv2D

What it does:

Each input channel receives its own spatial convolution instead of mixing every input channel with every output channel in one operation.

Why it was added:

Depthwise convolution is one of the main efficiency mechanisms used inside MBConv blocks. It also creates an interesting NumPy engineering problem because fewer parameters do not automatically mean lower CPU runtime.

## PointwiseConv2D

What it does:

A 1x1 convolution mixes information across channels without a larger spatial kernel.

Why it was added:

Depthwise convolution handles spatial filtering but does not perform general channel mixing. Pointwise convolution provides that channel mixing and is also used for MBConv expansion and projection.

## DepthwiseSeparableConv2D

What it does:

Runs depthwise convolution followed by pointwise convolution.

Why it was added:

It separates spatial filtering from channel mixing and greatly reduces parameter count compared with a standard convolution of the same input and output channel sizes.

Current benchmark example:

```text
standard params:  4,608
separable params:   656
parameter reduction: 85.76%
runtime ratio, separable / standard: 0.937
```

The runtime is benchmarked independently because implementation details and memory access matter on CPU.

## SqueezeExcitation

What it does:

Summarizes each channel globally, passes those channel descriptors through a small learned gating path, and rescales the original feature channels.

Why it was added:

EfficientNet uses squeeze-and-excitation inside MBConv blocks to let the network adapt channel importance from the current input.

## StochasticDepth

What it does:

During training, some residual branches are skipped for individual samples. During evaluation, the block behaves deterministically.

Why it was added:

This regularization strategy is used in deeper EfficientNet-style networks and needs correct training and evaluation behavior.

## Residual path

What it does:

Adds the block input back to the transformed output when shape and stride allow it.

Why it was added:

Residual paths improve gradient flow and are part of the MBConv design. The project explicitly tests the skip gradient so the branch cannot be accidentally lost during backward propagation.

## MBConv

What it does:

Combines optional channel expansion, normalization, SiLU activation, depthwise convolution, squeeze-and-excitation, pointwise projection, optional stochastic depth, and an optional residual path.

Why it was added:

MBConv is the core building block needed before constructing a complete EfficientNet.

## MiniEfficientNet

What it does:

Uses real MBConv components in a smaller CIFAR-scale model.

Why it was added:

A compact model is much faster to debug and train on CPU. It verifies the combined EfficientNet-style training path before a full B0 stage configuration is introduced.
