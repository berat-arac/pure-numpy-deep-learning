from __future__ import annotations

import argparse
from pathlib import Path

from puredl.text import DEFAULT_NATURAL_CORPUS, prepare_natural_corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the public-domain English text corpus")
    parser.add_argument("--output", type=Path, default=DEFAULT_NATURAL_CORPUS)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    path = prepare_natural_corpus(root, output_path=args.output, force=args.force)
    print(f"ready={path}")


if __name__ == "__main__":
    main()
