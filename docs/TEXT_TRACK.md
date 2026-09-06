# Text Track

## Goal

The Text track builds sequence models that predict and complete ordinary text using NumPy only.

The track stays character-level so the implementation can focus on recurrent state, attention, gradient flow, causal masking, and autoregressive generation without making tokenizer design a separate project.

## Shared corpus

The final completion experiments use natural English prose rather than repository source code.

The default preparation script downloads and combines three small public-domain works:

- Alice's Adventures in Wonderland
- Pride and Prejudice
- The Adventures of Sherlock Holmes

The combined text is normalized to a compact character vocabulary and stored under `data/text/`. Dataset files are local artifacts and are not committed to Git.

A user-supplied UTF-8 file can be passed with `--corpus`. An offline `local-prose` mode also exists for smoke testing without a network connection.

## RNN

The vanilla RNN provides the first recurrent baseline. It maintains a hidden state across the sequence and is trained with explicit backpropagation through time.

It is included to make recurrent state and temporal gradient flow visible before gated memory is introduced.

## LSTM

The LSTM adds a separate cell state and learned gates controlling what is retained, written, and exposed.

Its backward pass propagates gradients through both the hidden state and the cell state.

## Transformer

The Transformer path is built from independently validated primitives:

- scaled dot-product attention
- causal masking
- multi-head self-attention
- LayerNorm
- GELU feed-forward layers
- residual decoder blocks
- token embeddings
- positional embeddings
- stacked decoder blocks

The model uses a pre-normalized decoder structure and predicts the next character at every position.

## Completion behavior

RNN and LSTM models reuse recurrent state during completion. The Transformer uses a sliding causal context window and recomputes the context for every generated character.

This is intentionally simple and transparent. Key-value caching is outside the current milestone.

Completion quality is treated as a model-behavior demonstration, not a benchmark claim.

## Final Transformer run

Configuration:

- 1,029,199 parameters
- 6 decoder blocks
- 4 attention heads
- model width 128
- feed-forward width 384
- context length 128
- 200,000 natural-language characters
- 20 CPU epochs

```text
train_loss: 2.7814 -> 1.4415
train_acc:  24.16% -> 55.64%
val_loss:   2.5805 -> 1.7730
val_acc:    26.11% -> 48.23%
```

The validation accuracy measures next-character prediction. It is not word-level accuracy or a language-understanding score.

## Training workflow

Prepare the natural-language corpus once:

```bash
python scripts/prepare_text_corpus.py
```

Train recurrent models:

```bash
python scripts/train_char_lm.py --model rnn --dataset natural
python scripts/train_char_lm.py --model lstm --dataset natural
```

Train the Transformer:

```bash
python scripts/train_transformer.py --dataset natural
```

Transformer checkpoints are saved after each epoch and can be resumed with `--resume` when the same architecture and corpus settings are used.

## Validation

Text validation includes:

- embedding gradient checks including repeated token IDs
- RNN and LSTM BPTT finite difference checks
- LayerNorm full backward finite difference checks
- GELU finite difference checks
- attention query, key, and value finite difference checks
- causal future-visibility tests
- multi-head self-attention input-gradient checks
- complete decoder-block residual-gradient checks
- integrated Transformer pattern fitting
- real next-character training
- autoregressive prompt completion
- corpus normalization and prose-pipeline tests
