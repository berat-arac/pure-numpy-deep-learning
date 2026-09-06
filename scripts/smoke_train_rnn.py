from __future__ import annotations

import subprocess
import sys


if __name__ == "__main__":
    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                "scripts/train_char_lm.py",
                "--model",
                "rnn",
                "--epochs",
                "3",
                "--batch-size",
                "16",
                "--seq-len",
                "48",
                "--embedding-dim",
                "32",
                "--hidden-size",
                "64",
                "--max-chars",
                "40000",
                "--sample-len",
                "100",
            ]
        )
    )
