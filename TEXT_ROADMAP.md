# Text Roadmap

The Text track is complete for the current project scope.

The goal was to move from recurrent sequence models to a decoder-only Transformer while keeping every trainable operation and backward pass inside NumPy.

## Phase 1: Shared character pipeline

Status: complete

Implemented:

- deterministic character vocabulary
- encode and decode helpers
- fixed-length next-character sequence batching
- train and validation splitting
- custom UTF-8 corpus loading
- public-domain natural-language corpus preparation
- offline prose-only smoke corpus

Purpose:

Create one transparent character-level pipeline that can be reused by RNN, LSTM, and Transformer models.

## Phase 2: Vanilla RNN

Status: complete

Implemented:

- trainable character embeddings
- recurrent hidden state
- explicit backpropagation through time
- next-character language-model loss
- gradient clipping
- autoregressive completion
- finite difference gradient checks

Purpose:

Establish the first complete recurrent training path and make temporal gradient flow visible.

## Phase 3: LSTM

Status: complete

Implemented:

- input, forget, output, and candidate gates
- hidden state and cell state
- explicit hidden-state and cell-state backward propagation
- next-character training
- autoregressive completion
- finite difference gradient checks

Purpose:

Add gated recurrent memory before moving to attention-based sequence modeling.

## Phase 4: Attention primitives

Status: complete

Implemented:

- scaled dot-product attention
- stable softmax backward propagation
- causal masking
- query, key, and value gradients
- future-visibility tests
- finite difference validation

Purpose:

Validate attention independently before composing a full decoder block.

## Phase 5: Multi-head attention and normalization

Status: complete

Implemented:

- query, key, and value projections
- head splitting and merging
- multi-head causal self-attention
- output projection
- LayerNorm with complete backward propagation
- GELU with manual backward propagation
- feed-forward sublayer
- finite difference validation

Purpose:

Verify the main Transformer primitives before stacking residual decoder blocks.

## Phase 6: Decoder-only Transformer

Status: complete

Implemented:

- pre-normalized decoder blocks
- two explicit residual gradient paths per block
- token embeddings
- trainable positional embeddings
- stacked causal decoder blocks
- final LayerNorm
- language-model head
- dropout support
- next-character training
- autoregressive completion with a sliding context window
- checkpoint save and load
- resumable training
- CPU profiling

Development scaling gates:

- 54,296 parameters, 2 layers, 4 heads, context 32
- 372,443 parameters, 4 layers, 4 heads, context 64
- approximately 1.03M parameters, 6 layers, 4 heads, context 128

All configurations completed forward, backward, and optimization gates.

## Phase 7: Natural-language completion

Status: complete

The final run used ordinary English prose instead of repository source code.

Configuration:

- 1,029,199 parameters
- 6 decoder blocks
- 4 attention heads
- model width 128
- feed-forward width 384
- context length 128
- 200,000 characters
- 20 CPU epochs

Training result:

```text
train_loss: 2.7814 -> 1.4415
train_acc:  24.16% -> 55.64%
val_loss:   2.5805 -> 1.7730
val_acc:    26.11% -> 48.23%
```

The accuracy is next-character accuracy. It is not a language-understanding benchmark.

Completion tests from prompts such as `She was `, `The man `, and `For a moment ` show that the model learned common word boundaries, punctuation, dialogue-like structure, and local prose patterns. The generated English remains imperfect, which is expected from a small character-level model trained from scratch.

## Final status

Completed:

```text
Character pipeline
    -> RNN
    -> LSTM
    -> causal attention
    -> multi-head self-attention
    -> decoder block
    -> decoder-only Transformer
    -> natural-language completion
```

The current Text milestone ends here. Larger pretrained language models, token-level subword modeling, key-value caching, GPU acceleration, and distributed training are outside the project scope.
