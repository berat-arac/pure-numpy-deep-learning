from __future__ import annotations

import pickle
import sys
import tarfile
import urllib.request
import warnings
from pathlib import Path

import numpy as np

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)[:, None, None]
STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)[:, None, None]


def _safe_extract(tf: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in tf.getmembers():
        target = (destination / member.name).resolve()
        if destination != target and destination not in target.parents:
            raise ValueError(f"Unsafe path in CIFAR archive: {member.name}")
    tf.extractall(destination)


def _download_progress(block_count: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    downloaded = min(block_count * block_size, total_size)
    percent = downloaded * 100.0 / total_size
    mib = downloaded / (1024 * 1024)
    total_mib = total_size / (1024 * 1024)
    sys.stdout.write(f"\rDownloading CIFAR-10: {percent:6.2f}% ({mib:.1f}/{total_mib:.1f} MiB)")
    sys.stdout.flush()
    if downloaded >= total_size:
        sys.stdout.write("\n")


def download_cifar10(root: str | Path) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "cifar-10-python.tar.gz"
    folder = root / "cifar-10-batches-py"
    if folder.exists():
        return folder
    if not archive.exists():
        tmp = archive.with_suffix(archive.suffix + ".part")
        print(f"Downloading CIFAR-10 to {archive}")
        try:
            urllib.request.urlretrieve(URL, tmp, reporthook=_download_progress)
            tmp.replace(archive)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
    print(f"Extracting CIFAR-10 into {root}")
    with tarfile.open(archive, "r:gz") as tf:
        _safe_extract(tf, root)
    return folder


def _read(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        # The official CIFAR-10 Python archive contains an old NumPy dtype
        # encoding that emits a NumPy 2.4 deprecation warning while unpickling.
        # The warning is limited to archive metadata and does not affect the data.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"dtype\(\): align should be passed.*",
            )
            data = pickle.load(f, encoding="bytes")
    x = data[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    y = np.asarray(data[b"labels"], dtype=np.int64)
    return x, y


def load_cifar10(root="data", download=True, normalize=True):
    folder = download_cifar10(root) if download else Path(root) / "cifar-10-batches-py"
    xs, ys = [], []
    for i in range(1, 6):
        x, y = _read(folder / f"data_batch_{i}")
        xs.append(x)
        ys.append(y)
    train_x = np.concatenate(xs)
    train_y = np.concatenate(ys)
    test_x, test_y = _read(folder / "test_batch")
    if normalize:
        train_x = (train_x - MEAN) / STD
        test_x = (test_x - MEAN) / STD
    return train_x, train_y, test_x, test_y


def augment_batch(x, rng, padding=4):
    out = np.empty_like(x)
    padded = np.pad(
        x,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="reflect",
    )
    for i in range(len(x)):
        y = int(rng.integers(0, 2 * padding + 1))
        xx = int(rng.integers(0, 2 * padding + 1))
        sample = padded[i, :, y : y + 32, xx : xx + 32]
        if rng.random() < 0.5:
            sample = sample[:, :, ::-1]
        out[i] = sample
    return out


def batches(x, y, batch_size, rng, shuffle=True, augment=False):
    idx = np.arange(len(x))
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, len(x), batch_size):
        ids = idx[start : start + batch_size]
        bx = x[ids]
        if augment:
            bx = augment_batch(bx, rng)
        yield bx, y[ids]
