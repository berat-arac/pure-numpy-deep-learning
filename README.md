# Pure NumPy Deep Learning

[![CI](https://github.com/berat-arac/pure-numpy-deep-learning/actions/workflows/ci.yml/badge.svg)](https://github.com/berat-arac/pure-numpy-deep-learning/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A from-scratch deep learning project implemented with NumPy only.

No PyTorch, TensorFlow, JAX, CuPy, or automatic differentiation is used. Forward passes, backward passes, optimizer updates, normalization state, checkpointing, and training loops are implemented directly in NumPy and run on CPU.

The project has two completed tracks:

- Vision: MLP to CNNs, residual and efficient blocks, then EfficientNet-B0
- Text: RNN to LSTM, causal attention, then a decoder-only Transformer for natural-language completion

## Project status

### Vision track

Completed:

- MLP training on MNIST
- Conv2D, pooling, BatchNorm2D, and deeper CNN training
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
- full B0 forward and backward propagation
- CPU profiling and trainability diagnostics

The canonical 1000-class EfficientNet-B0 configuration contains **5,288,548 parameters**, closely matching the approximately 5.3M parameter count associated with EfficientNet-B0.

### Text track

Completed:

- character vocabulary and sequence batching
- trainable embedding lookup
- vanilla RNN with explicit backpropagation through time
- LSTM with explicit hidden-state and cell-state backward propagation
- gradient clipping and autoregressive recurrent completion
- scaled dot-product attention
- stable softmax attention backward propagation
- causal masking
- multi-head self-attention
- LayerNorm with complete backward propagation
- GELU feed-forward layers
- pre-normalized decoder blocks with explicit residual gradients
- trainable positional embeddings
- stacked decoder-only Transformer
- autoregressive natural-language completion
- checkpoint save, load, and resume
- CPU profiling
- public-domain English prose corpus preparation

The final Transformer training run used a 1,029,199 parameter character-level model with 6 decoder blocks, 4 attention heads, and context length 128.

## Why this project exists

High-level frameworks make deep learning practical, but they hide much of the execution path. This project removes that abstraction so the individual operations, state transitions, gradients, and model-building decisions remain visible.

The goal is not to compete with optimized frameworks in speed or benchmark accuracy. The goal is to implement the important mechanisms directly, verify that they work, and train real models without relying on a neural network framework.

## What was built and why

| Component | What was implemented | Why it is here |
| --- | --- | --- |
| Parameter and Module | Parameters, gradients, nested modules, persistent buffers, state loading | Provides a reusable foundation without building a framework clone |
| Linear | Dense forward and backward passes | Establishes the first complete trainable path |
| Conv2D | NumPy convolution with manual input, weight, and bias gradients | Provides the core operation for the Vision track |
| BatchNorm2D | Training and evaluation state, running statistics, backward pass | Supports deeper CNN and MBConv training |
| Residual block | Main branch and explicit skip-gradient path | Validates residual gradient flow before MBConv |
| Depthwise and pointwise convolution | Efficient spatial filtering and explicit 1x1 channel mixing | Builds the convolution structure used by MBConv |
| SqueezeExcitation | Learned channel recalibration | Implements the channel gating used by EfficientNet |
| StochasticDepth | Residual branch dropping during training | Adds EfficientNet-style regularization |
| MBConv | Expansion, depthwise convolution, SE, projection, normalization, residual path | Implements the core EfficientNet block |
| EfficientNet-B0 | Config-driven seven-stage layout with 16 MBConv blocks | Completes the Vision track with the target architecture |
| Embedding | Character ID lookup with repeated-index gradient accumulation | Provides a compact trainable sequence representation |
| Vanilla RNN | Recurrent hidden state and explicit BPTT | Establishes the first sequence-model baseline |
| LSTM | Gated hidden and cell state with explicit BPTT | Adds controlled recurrent memory before attention |
| Scaled attention | Causal scaled dot-product attention with complete softmax backward propagation | Establishes the attention primitive independently |
| Multi-head attention | Q, K, V projections, head splitting, causal attention, merging, output projection | Builds decoder self-attention |
| LayerNorm | Feature normalization with trainable scale, bias, and full backward propagation | Avoids partial-gradient shortcuts in Transformer blocks |
| Decoder block | Pre-normalized attention and GELU feed-forward branches with explicit residual gradients | Combines verified Transformer primitives |
| Decoder-only Transformer | Token and positional embeddings, stacked decoder blocks, final normalization, LM head | Completes the Text track with autoregressive natural-language completion |

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

### EfficientNet-B0 validation

The full CIFAR-10 configuration contains 4,020,358 parameters and 16 MBConv blocks. Full forward and backward propagation were profiled on CPU.

A controlled memorization diagnostic with augmentation and regularization disabled reached 99.61% fit accuracy on 256 fixed training images. This is not a model-quality result. It is an end-to-end optimization check showing that the complete model can fit a small fixed dataset.

### RNN and LSTM gates

RNN and LSTM were trained as real next-character language models with explicit BPTT. After the Text track moved to ordinary prose, both models were also rerun on a prose-only pipeline to verify that training was no longer tied to source-code text.

```text
RNN  prose gate val_loss: 3.1990 -> 2.3494
RNN  prose gate val_acc:  15.36% -> 34.57%

LSTM prose gate val_loss: 3.2993 -> 2.5126
LSTM prose gate val_acc:  14.23% -> 30.37%
```

These values are training-pipeline checks, not language-model benchmarks.

### Decoder-only Transformer on natural English prose

Final configuration:

- 1,029,199 parameters
- 6 decoder blocks
- 4 attention heads
- model width 128
- feed-forward width 384
- context length 128
- 200,000 characters of public-domain English prose
- character-level next-character prediction
- 20 CPU training epochs

```text
train_loss: 2.7814 -> 1.4415
train_acc:  24.16% -> 55.64%
val_loss:   2.5805 -> 1.7730
val_acc:    26.11% -> 48.23%
```

The validation accuracy is next-character accuracy, not word-level or language-understanding accuracy.

The model also produces autoregressive continuations from ordinary prompts. The output remains imperfect because this is a small character-level model trained from scratch, but it learns word boundaries, punctuation, dialogue-like structure, and recurring prose patterns.

Example prompt:

```text
The 
```

Example generated fragment during training:

```text
The said Alice.
`How won't I not be sput to be begin what a could not ot down the dood
```

This sample is included as a behavior demonstration, not as a quality benchmark.

## Validation strategy

The project does not use another ML framework as a numerical oracle. Correctness is checked with:

- finite difference gradient checks
- deterministic small-array tests
- residual branch tests
- shape and state tests
- checkpoint round-trip tests
- causal future-visibility tests
- integrated learning tests
- real dataset training gates
- full-model trainability diagnostics

Important manually implemented backward paths covered by finite difference tests include:

- Linear
- Conv2D
- DepthwiseConv2D
- BatchNorm2D
- SqueezeExcitation
- Embedding
- RNN
- LSTM
- GELU
- LayerNorm
- scaled dot-product attention
- multi-head self-attention
- complete decoder block

See [docs/VALIDATION_STRATEGY.md](docs/VALIDATION_STRATEGY.md) for details.

## Natural-language corpus

The default Text training pipeline combines three small public-domain English prose sources:

- Alice's Adventures in Wonderland
- Pride and Prejudice
- The Adventures of Sherlock Holmes

The corpus is stored under `data/text/` and is not committed to Git. A custom UTF-8 text file can be supplied instead.

Prepare the default corpus once:

```bash
python scripts/prepare_text_corpus.py
```

## Installation

Python 3.11 or newer is required.

```bash
python -m venv .venv
```

Git Bash on Windows:

```bash
source .venv/Scripts/activate
```

Install the project and development dependencies:

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Run the complete test suite:

```bash
pytest -v
```

## Vision examples

Train the MNIST MLP:

```bash
python scripts/train_mnist.py --epochs 2 --limit-train 5000 --limit-test 1000
```

Train the deeper CNN on a CIFAR-10 subset:

```bash
python scripts/train_cifar10.py \
  --epochs 10 \
  --batch-size 64 \
  --width 0.5 \
  --limit-train 10000 \
  --limit-test 2000
```

Train MiniEfficientNet:

```bash
python scripts/train_mini_efficientnet.py \
  --epochs 10 \
  --batch-size 32 \
  --width 0.5 \
  --drop-path 0.10 \
  --limit-train 10000 \
  --limit-test 2000
```

Profile the full EfficientNet-B0 forward and backward path:

```bash
python scripts/profile_efficientnet.py \
  --batch-size 1 \
  --width-mult 1.0 \
  --depth-mult 1.0
```

Run the controlled EfficientNet-B0 trainability diagnostic:

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

## Text examples

Train the RNN:

```bash
python scripts/train_char_lm.py --model rnn --dataset natural
```

Train the LSTM:

```bash
python scripts/train_char_lm.py --model lstm --dataset natural
```

Train the Transformer:

```bash
python scripts/train_transformer.py \
  --dataset natural \
  --epochs 20 \
  --batch-size 8 \
  --context-length 128 \
  --d-model 128 \
  --heads 4 \
  --layers 6 \
  --ff-dim 384 \
  --dropout 0.05 \
  --lr 0.001 \
  --max-chars 200000
```

Resume the same Transformer run:

```bash
python scripts/train_transformer.py \
  --dataset natural \
  --epochs 20 \
  --batch-size 8 \
  --context-length 128 \
  --d-model 128 \
  --heads 4 \
  --layers 6 \
  --ff-dim 384 \
  --dropout 0.05 \
  --lr 0.001 \
  --max-chars 200000 \
  --resume
```

Generate a continuation:

```bash
python scripts/complete_transformer.py \
  --checkpoint outputs/text/char_transformer.npz \
  --metadata outputs/text/char_transformer.json \
  --prompt "For a moment " \
  --length 250 \
  --temperature 0.55 \
  --top-k 10
```

Profile the Transformer forward and backward path:

```bash
python scripts/profile_transformer.py \
  --batch-size 2 \
  --context-length 128 \
  --vocab-size 96 \
  --d-model 128 \
  --heads 4 \
  --layers 6 \
  --ff-dim 384
```

## Repository structure

```text
pure-numpy-deep-learning/
|-- docs/
|-- scripts/
|-- src/
|   `-- puredl/
|       |-- core/
|       |-- data/
|       |-- layers/
|       |-- losses/
|       |-- optim/
|       |-- text/
|       |-- training/
|       |-- utils/
|       `-- vision/
|-- tests/
|-- README.md
|-- TEXT_ROADMAP.md
|-- VISION_ROADMAP.md
`-- pyproject.toml
```

## Scope

Intentionally outside the current project:

- GPU backends
- automatic differentiation
- distributed training
- ImageNet-scale EfficientNet training
- large-language-model scale training
- production framework compatibility
- Transformer key-value caching

The constraint is deliberate: the learning and model implementation remains NumPy-only.

## Roadmaps

- [Vision roadmap](VISION_ROADMAP.md)
- [Text roadmap](TEXT_ROADMAP.md)

## License

MIT
