from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import re
import urllib.request

import numpy as np


NATURAL_TEXT_SOURCES = (
    (
        "Alice's Adventures in Wonderland",
        "https://raw.githubusercontent.com/google/snappy/main/testdata/alice29.txt",
    ),
    (
        "Pride and Prejudice",
        "https://raw.githubusercontent.com/crista/exercises-in-programming-style/master/pride-and-prejudice.txt",
    ),
    (
        "The Adventures of Sherlock Holmes",
        "https://raw.githubusercontent.com/GITenberg/The-Adventures-of-Sherlock-Holmes_1661/master/1661.txt",
    ),
)

DEFAULT_NATURAL_CORPUS = Path("data/text/public_domain_english.txt")


def build_repo_corpus(root: str | Path) -> str:
    """Build a deterministic character corpus from repository source and docs."""
    root = Path(root)
    candidates: list[Path] = []
    for base in (root / "src", root / "docs"):
        if base.exists():
            candidates.extend(base.rglob("*.py"))
            candidates.extend(base.rglob("*.md"))
    for name in ("README.md", "VISION_ROADMAP.md", "TEXT_ROADMAP.md"):
        path = root / name
        if path.exists():
            candidates.append(path)

    unique = sorted({path.resolve() for path in candidates})
    chunks: list[str] = []
    for path in unique:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        relative = path.relative_to(root.resolve())
        chunks.append(f"\n# FILE: {relative.as_posix()}\n{text}\n")
    corpus = "".join(chunks)
    if not corpus:
        raise ValueError("no text files found for repository corpus")
    return corpus


def build_local_prose_corpus(root: str | Path) -> str:
    """Build an offline English prose corpus from Markdown documentation only."""
    root = Path(root)
    candidates: list[Path] = []
    for base in (root, root / "docs"):
        if not base.exists():
            continue
        candidates.extend(base.glob("*.md"))

    chunks: list[str] = []
    for path in sorted({item.resolve() for item in candidates}):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        text = _strip_markdown_code(text)
        text = _strip_markdown_markup(text)
        text = normalize_natural_text(text)
        if text.strip():
            chunks.append(text.strip())
    corpus = "\n\n".join(chunks)
    if not corpus:
        raise ValueError("no Markdown prose found for local corpus")
    return corpus


def _strip_markdown_code(text: str) -> str:
    return re.sub(r"```.*?```", " ", text, flags=re.DOTALL)


def _strip_markdown_markup(text: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+[.)]\s+", "", text, flags=re.MULTILINE)
    text = text.replace("|", " ")
    return text


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove common Project Gutenberg header and footer blocks when present."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    start_patterns = (
        r"\*\*\*\s*START OF (?:THIS|THE) PROJECT GUTENBERG EBOOK[^\n]*\*\*\*",
        r"\*END\*THE SMALL PRINT![^\n]*",
    )
    end_patterns = (
        r"\*\*\*\s*END OF (?:THIS|THE) PROJECT GUTENBERG EBOOK[^\n]*\*\*\*",
        r"End of (?:the )?Project Gutenberg[^\n]*",
    )

    start_index = None
    for pattern in start_patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            start_index = match.end()
            break
    if start_index is not None:
        normalized = normalized[start_index:]

    end_index = None
    for pattern in end_patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            end_index = match.start()
            break
    if end_index is not None:
        normalized = normalized[:end_index]

    return normalized.strip()


def normalize_natural_text(text: str) -> str:
    """Normalize prose to a compact ASCII-friendly character corpus."""
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\u00a0": " ",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(char for char in text if char == "\n" or char == "\t" or ord(char) >= 32)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def _download_text(url: str, *, label: str) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "pure-numpy-deep-learning/0.6.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        chunks: list[bytes] = []
        while True:
            chunk = response.read(1024 * 128)
            if not chunk:
                break
            chunks.append(chunk)
            downloaded += len(chunk)
            if total:
                percent = 100.0 * downloaded / total
                print(
                    f"Downloading {label}: {percent:6.2f}% "
                    f"({downloaded / 1024**2:.2f}/{total / 1024**2:.2f} MiB)",
                    end="\r",
                    flush=True,
                )
        if total:
            print()
    raw = b"".join(chunks)
    return raw.decode("utf-8", errors="replace")


def prepare_natural_corpus(
    root: str | Path,
    *,
    output_path: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Download and combine small public-domain English prose sources."""
    root = Path(root)
    path = Path(output_path) if output_path is not None else root / DEFAULT_NATURAL_CORPUS
    if not path.is_absolute():
        path = root / path
    if path.exists() and not force:
        return path

    parts: list[str] = []
    for label, url in NATURAL_TEXT_SOURCES:
        print(f"Source: {label}")
        text = _download_text(url, label=label)
        text = strip_gutenberg_boilerplate(text)
        text = normalize_natural_text(text)
        parts.append(text)

    corpus = "\n\n".join(part.strip() for part in parts if part.strip()) + "\n"
    if not corpus.strip():
        raise RuntimeError("downloaded natural text corpus is empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(corpus, encoding="utf-8")
    print(f"Natural text corpus: {path} ({len(corpus):,} characters)")
    return path


def load_text_corpus(
    root: str | Path,
    *,
    corpus_path: str | Path | None = None,
    dataset: str = "natural",
    auto_download: bool = True,
) -> str:
    """Load a custom, natural-language, local-prose, or repository corpus."""
    root = Path(root)
    if corpus_path is not None:
        path = Path(corpus_path)
        if not path.is_absolute():
            path = root / path
        return normalize_natural_text(path.read_text(encoding="utf-8"))

    if dataset == "natural":
        path = root / DEFAULT_NATURAL_CORPUS
        if not path.exists():
            if not auto_download:
                raise FileNotFoundError(
                    f"natural corpus not found at {path}; run scripts/prepare_text_corpus.py"
                )
            path = prepare_natural_corpus(root)
        return normalize_natural_text(path.read_text(encoding="utf-8"))
    if dataset == "local-prose":
        return build_local_prose_corpus(root)
    if dataset == "repo":
        return build_repo_corpus(root)
    raise ValueError("dataset must be one of: natural, local-prose, repo")


def sequence_batches(
    token_ids: np.ndarray,
    *,
    seq_len: int,
    batch_size: int,
    shuffle: bool = True,
    rng: np.random.Generator | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
    if seq_len <= 0 or batch_size <= 0:
        raise ValueError("seq_len and batch_size must be positive")
    if ids.size <= seq_len:
        raise ValueError("corpus is too short for requested sequence length")

    starts = np.arange(0, ids.size - seq_len, seq_len, dtype=np.int64)
    if shuffle:
        rng = rng or np.random.default_rng()
        rng.shuffle(starts)

    usable = (starts.size // batch_size) * batch_size
    starts = starts[:usable]
    for offset in range(0, starts.size, batch_size):
        batch_starts = starts[offset : offset + batch_size]
        x = np.stack([ids[s : s + seq_len] for s in batch_starts])
        y = np.stack([ids[s + 1 : s + seq_len + 1] for s in batch_starts])
        yield x, y
