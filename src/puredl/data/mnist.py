from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    tmp = destination.with_suffix(destination.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(destination)


def download_mnist(root: str | Path = "data") -> Path:
    root = Path(root) / "mnist"
    root.mkdir(parents=True, exist_ok=True)
    for filename in FILES.values():
        _download(BASE_URL + filename, root / filename)
    return root


def _read_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid MNIST image magic number in {path}: {magic}")
        raw = f.read()
    expected = count * rows * cols
    if len(raw) != expected:
        raise ValueError(f"Unexpected MNIST image payload size in {path}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(count, rows, cols)


def _read_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid MNIST label magic number in {path}: {magic}")
        raw = f.read()
    if len(raw) != count:
        raise ValueError(f"Unexpected MNIST label payload size in {path}")
    return np.frombuffer(raw, dtype=np.uint8).astype(np.int64)


def load_mnist(
    root: str | Path = "data",
    *,
    download: bool = True,
    flatten: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    folder = download_mnist(root) if download else Path(root) / "mnist"
    train_x = _read_images(folder / FILES["train_images"]).astype(np.float32) / 255.0
    train_y = _read_labels(folder / FILES["train_labels"])
    test_x = _read_images(folder / FILES["test_images"]).astype(np.float32) / 255.0
    test_y = _read_labels(folder / FILES["test_labels"])
    if flatten:
        train_x = train_x.reshape(len(train_x), -1)
        test_x = test_x.reshape(len(test_x), -1)
    else:
        train_x = train_x[:, None, :, :]
        test_x = test_x[:, None, :, :]
    return train_x, train_y, test_x, test_y
