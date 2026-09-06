from __future__ import annotations

import numpy as np

from .language_model import CharLanguageModel
from .vocabulary import CharVocabulary


def _sample_logits(
    logits: np.ndarray,
    *,
    temperature: float,
    top_k: int | None,
    rng: np.random.Generator,
) -> int:
    if temperature <= 0:
        return int(np.argmax(logits))
    scores = logits.astype(np.float64) / float(temperature)
    if top_k is not None and 0 < top_k < scores.size:
        keep = np.argpartition(scores, -top_k)[-top_k:]
        mask = np.full(scores.shape, -np.inf, dtype=np.float64)
        mask[keep] = scores[keep]
        scores = mask
    scores -= np.max(scores)
    probs = np.exp(scores)
    probs /= probs.sum()
    return int(rng.choice(scores.size, p=probs))


def generate_text(
    model: CharLanguageModel,
    vocab: CharVocabulary,
    prompt: str,
    *,
    max_new_chars: int = 200,
    temperature: float = 0.8,
    top_k: int | None = 20,
    rng: np.random.Generator | None = None,
) -> str:
    if not prompt:
        raise ValueError("prompt must not be empty")
    prompt_ids = vocab.encode(prompt)
    rng = rng or np.random.default_rng()
    state = model.initial_state(1)
    logits = None
    for token in prompt_ids:
        logits, state = model.step_token(np.array([token], dtype=np.int64), state)
    assert logits is not None
    output = list(prompt)
    current_logits = logits[0]
    for _ in range(max_new_chars):
        next_id = _sample_logits(
            current_logits,
            temperature=temperature,
            top_k=top_k,
            rng=rng,
        )
        output.append(vocab.chars[next_id])
        logits, state = model.step_token(np.array([next_id], dtype=np.int64), state)
        current_logits = logits[0]
    return "".join(output)


def generate_transformer_text(
    model,
    vocab: CharVocabulary,
    prompt: str,
    *,
    max_new_chars: int = 200,
    temperature: float = 0.8,
    top_k: int | None = 20,
    rng: np.random.Generator | None = None,
) -> str:
    """Autoregressive character completion using a decoder-only Transformer."""
    if not prompt:
        raise ValueError("prompt must not be empty")
    ids = list(vocab.encode(prompt).astype(int))
    rng = rng or np.random.default_rng()
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_chars):
            context = np.asarray(ids[-model.context_length :], dtype=np.int64)[None, :]
            logits = model.next_token_logits(context)[0]
            next_id = _sample_logits(
                logits,
                temperature=temperature,
                top_k=top_k,
                rng=rng,
            )
            ids.append(next_id)
    finally:
        if was_training:
            model.train()
    return vocab.decode(ids)
