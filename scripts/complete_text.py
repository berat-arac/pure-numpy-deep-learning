from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from puredl.text import CharLanguageModel, CharVocabulary, generate_text
from puredl.training.checkpoint import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with a trained pure NumPy RNN or LSTM")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--length", type=int, default=240)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    vocab = CharVocabulary.from_dict(metadata["vocab"])
    model = CharLanguageModel(
        vocab.size,
        int(metadata["embedding_dim"]),
        int(metadata["hidden_size"]),
        cell=str(metadata["cell"]),
        rng=np.random.default_rng(0),
    )
    load_checkpoint(args.checkpoint, model=model)
    result = generate_text(
        model,
        vocab,
        args.prompt,
        max_new_chars=args.length,
        temperature=args.temperature,
        top_k=args.top_k,
        rng=np.random.default_rng(args.seed),
    )
    print(result)


if __name__ == "__main__":
    main()
