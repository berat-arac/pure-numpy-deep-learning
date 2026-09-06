from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CharVocabulary:
    chars: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.chars:
            raise ValueError("vocabulary must contain at least one character")
        if len(set(self.chars)) != len(self.chars):
            raise ValueError("vocabulary characters must be unique")

    @classmethod
    def from_text(cls, text: str) -> "CharVocabulary":
        if not text:
            raise ValueError("text must not be empty")
        return cls(tuple(sorted(set(text))))

    @property
    def size(self) -> int:
        return len(self.chars)

    @property
    def stoi(self) -> dict[str, int]:
        return {char: index for index, char in enumerate(self.chars)}

    def encode(self, text: str) -> np.ndarray:
        table = self.stoi
        missing = sorted(set(text) - set(table))
        if missing:
            preview = "".join(missing[:12])
            raise ValueError(f"text contains characters outside vocabulary: {preview!r}")
        return np.fromiter((table[ch] for ch in text), dtype=np.int64, count=len(text))

    def decode(self, ids: np.ndarray | list[int]) -> str:
        array = np.asarray(ids, dtype=np.int64).reshape(-1)
        if array.size and (array.min() < 0 or array.max() >= self.size):
            raise ValueError("token id out of range")
        return "".join(self.chars[int(index)] for index in array)

    def to_dict(self) -> dict[str, list[str]]:
        return {"chars": list(self.chars)}

    @classmethod
    def from_dict(cls, state: dict[str, list[str]]) -> "CharVocabulary":
        return cls(tuple(state["chars"]))
