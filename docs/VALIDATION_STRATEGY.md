# Validation Strategy

This repository validates manually implemented neural network operations without using another machine learning framework.

## Finite difference gradient checks

Small tensors are used to compare numerical gradient estimates with each layer's manually implemented backward result.

Current coverage includes:

- Linear
- Conv2D
- DepthwiseConv2D weight gradient
- DepthwiseConv2D input gradient
- BatchNorm2D
- SiLU
- SqueezeExcitation input gradient
- CrossEntropyLoss

Why this is used:

A model can sometimes reduce loss even when an individual backward implementation is slightly wrong. Gradient checks help catch those local errors before several layers are combined.

## Deterministic operation tests

Small fixed arrays are used for operations such as pooling and loss functions.

Why this is used:

Simple examples make incorrect indexing, reduction, or shape behavior easier to isolate.

## Residual branch tests

Residual tests can disable the main branch and verify that the skip path still forwards the input and returns the upstream gradient correctly.

Why this is used:

Forgetting the skip contribution in backward propagation can silently damage gradient flow in deeper networks.

## Shape and state tests

Tests cover:

- output shapes
- parameter gradient shapes
- training and evaluation behavior
- BatchNorm running state
- optimizer state
- scheduler state
- checkpoint save and restore

Why this is used:

Many neural network bugs are state-management bugs rather than arithmetic bugs. These tests verify the behavior around the numerical operations.

## Integrated learning tests

Small NumPy-generated image tasks are used to verify that complete model stacks can reduce loss through real optimizer updates.

Why this is used:

Passing isolated tests does not guarantee that several manually implemented layers work together during training.

## Real dataset gates

MNIST and CIFAR-10 are used as development gates before more expensive architectures are added.

Why this is used:

Real data exposes optimization, normalization, batching, and generalization behavior that synthetic tests cannot fully reproduce.
