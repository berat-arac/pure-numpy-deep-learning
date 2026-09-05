from pathlib import Path


def test_source_and_scripts_do_not_import_ml_frameworks():
    root = Path(__file__).resolve().parents[2]
    forbidden = (
        "import torch",
        "from torch",
        "import tensorflow",
        "from tensorflow",
        "import jax",
        "from jax",
        "import sklearn",
        "from sklearn",
        "import cupy",
        "from cupy",
    )
    offenders = []
    for base in (root / "src", root / "scripts"):
        for path in base.rglob("*.py"):
            text = path.read_text(encoding="utf-8").lower()
            for token in forbidden:
                if token in text:
                    offenders.append(f"{path.relative_to(root)}: {token}")
    assert not offenders, "Forbidden ML framework imports found: " + ", ".join(offenders)
