from __future__ import annotations

import pickle

import numpy as np

from puredl.data import batches, load_cifar10


def _write_batch(path, count, label_offset=0):
    x = (np.arange(count * 3 * 32 * 32, dtype=np.uint32) % 256).astype(np.uint8)
    x = x.reshape(count, 3 * 32 * 32)
    labels = [(label_offset + i) % 10 for i in range(count)]
    with open(path, "wb") as f:
        pickle.dump({b"data": x, b"labels": labels}, f)


def test_data_package_import_and_local_cifar_loading(tmp_path):
    folder = tmp_path / "cifar-10-batches-py"
    folder.mkdir()
    for i in range(1, 6):
        _write_batch(folder / f"data_batch_{i}", 2, i)
    _write_batch(folder / "test_batch", 3)

    train_x, train_y, test_x, test_y = load_cifar10(
        tmp_path, download=False, normalize=False
    )

    assert train_x.shape == (10, 3, 32, 32)
    assert train_y.shape == (10,)
    assert test_x.shape == (3, 3, 32, 32)
    assert test_y.shape == (3,)
    assert train_x.dtype == np.float32

    rng = np.random.default_rng(0)
    bx, by = next(batches(train_x, train_y, 4, rng, shuffle=False))
    assert bx.shape == (4, 3, 32, 32)
    assert by.shape == (4,)
