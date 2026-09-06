# Decoder-only Transformer

## Goal

Build a trainable decoder-only Transformer from NumPy primitives without automatic differentiation or an external neural network framework.

The implementation is developed in stages so attention, normalization, residual, and shape errors are isolated before several decoder blocks are stacked together.

## Attention

Scaled dot-product attention receives query, key, and value tensors split across attention heads.

Causal masking prevents a position from reading later positions. The test suite changes future values and verifies that earlier outputs remain unchanged.

The backward pass computes the complete softmax interaction rather than using a partial shortcut.

## Multi-head self-attention

The multi-head layer implements:

- query, key, and value projections
- head splitting
- causal attention
- head merging
- output projection

The complete input gradient is checked against finite differences on small causal examples.

## LayerNorm

LayerNorm normalizes the final feature dimension and learns scale and bias parameters.

Its backward pass includes the dependencies created by the feature mean and variance. Input, scale, and bias gradients are checked with finite differences.

## Decoder block

The decoder block uses a pre-normalized structure:

- LayerNorm
- causal multi-head self-attention
- residual connection
- LayerNorm
- GELU feed-forward network
- residual connection

Dropout can be enabled during training. Both residual contributions are explicit in backward propagation.

## Language model

The model combines:

- character token embeddings
- trainable positional embeddings
- stacked decoder blocks
- final LayerNorm
- vocabulary projection

Training predicts the next character at every context position. Completion is autoregressive and reuses the latest context window for each new character.

The current implementation deliberately favors clarity over generation speed. It does not use a key-value cache.

## Scaling gates

Before the final natural-language run, the same implementation was validated at several sizes:

- 54,296 parameters, 2 layers, 4 heads, context 32
- 372,443 parameters, 4 layers, 4 heads, context 64
- approximately 1.03M parameters, 6 layers, 4 heads, context 128

Each configuration completed forward, backward, and optimization tests.

## Final natural-language run

Configuration:

- 1,029,199 parameters
- 6 decoder layers
- 4 attention heads
- model width 128
- feed-forward width 384
- context length 128
- 200,000 characters of public-domain English prose
- 20 CPU epochs

```text
train_loss: 2.7814 -> 1.4415
train_acc:  24.16% -> 55.64%
val_loss:   2.5805 -> 1.7730
val_acc:    26.11% -> 48.23%
```

The model produces imperfect but recognizable prose-like continuations with learned spacing, punctuation, dialogue patterns, and common local word structure.

This result demonstrates end-to-end natural-language completion from the manually implemented decoder stack. It is not intended to compete with pretrained language models.
