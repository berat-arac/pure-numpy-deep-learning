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
- Embedding
- VanillaRNN input and parameter gradients
- LSTM input and parameter gradients
- GELU
- LayerNorm input, scale, and bias gradients
- scaled dot-product attention query, key, and value gradients
- multi-head self-attention input gradients
- complete decoder-block input gradients

Why this is used:

A model can sometimes reduce loss even when an individual backward implementation is slightly wrong. Gradient checks help catch local errors before several layers are combined.

## Deterministic operation tests

Small fixed arrays are used for operations such as pooling, loss functions, masking, and state handling.

Why this is used:

Simple examples make incorrect indexing, reduction, or shape behavior easier to isolate.

## Residual branch tests

Residual tests can disable the main branch and verify that the skip path still forwards the input and returns the upstream gradient correctly.

Why this is used:

Forgetting the skip contribution in backward propagation can silently damage gradient flow in deeper networks.

## Causal attention tests

Future value vectors are changed while earlier query positions are held fixed. Earlier outputs must remain unchanged.

Why this is used:

A decoder-only language model must never read future tokens during causal self-attention.

## Shape and state tests

Tests cover:

- output shapes
- parameter gradient shapes
- training and evaluation behavior
- BatchNorm running state
- optimizer state
- scheduler state
- checkpoint save and restore
- EfficientNet stage construction
- width and depth scaling behavior
- Transformer context and head shapes

Why this is used:

Many neural network bugs are state-management or shape bugs rather than arithmetic bugs.

## Integrated learning tests

Small generated tasks verify that complete model stacks can reduce loss through real optimizer updates.

Why this is used:

Passing isolated tests does not guarantee that several manually implemented layers work together during training.

## Real dataset gates

MNIST and CIFAR-10 are used for the Vision track. Natural English prose is used for the final Text track.

Why this is used:

Real data exposes optimization, normalization, batching, regularization, and generalization behavior that synthetic tests cannot fully reproduce.

## Full EfficientNet-B0 memorization diagnostic

The complete EfficientNet-B0 implementation is tested on a small fixed CIFAR-10 subset with augmentation and regularization disabled.

The diagnostic reached 99.61% fit accuracy on 256 fixed training images.

This is not reported as model quality. It is an end-to-end optimization check showing that the complete forward path, backward path, optimizer updates, and model state work together.

## Sequence-model training gates

RNN, LSTM, and decoder-only Transformer models are trained on next-character prediction.

The final Transformer run used 200,000 characters of public-domain English prose for 20 CPU epochs:

```text
train_loss: 2.7814 -> 1.4415
val_loss:   2.5805 -> 1.7730
val_acc:    26.11% -> 48.23%
```

The reported validation accuracy is next-character accuracy. It is used as a training-behavior signal, not as a language benchmark.

Autoregressive prompt completion is also inspected to verify that the complete generation path works after training.
