import numpy as np

from puredl.core import Parameter
from puredl.text import sequence_batches, clip_grad_norm


def test_sequence_batches_shift_targets_by_one():
    ids = np.arange(41, dtype=np.int64)
    batches = list(sequence_batches(ids, seq_len=5, batch_size=2, shuffle=False))
    assert batches
    x, y = batches[0]
    np.testing.assert_array_equal(y[:, :-1], x[:, 1:])
    np.testing.assert_array_equal(y[:, 0], x[:, 0] + 1)


def test_clip_grad_norm_scales_global_norm():
    p1 = Parameter(np.ones(2, dtype=np.float32))
    p2 = Parameter(np.ones(1, dtype=np.float32))
    p1.grad[:] = [3.0, 4.0]
    p2.grad[:] = [12.0]
    norm = clip_grad_norm([p1, p2], 6.5)
    assert abs(norm - 13.0) < 1e-6
    new_norm = np.sqrt(np.sum(p1.grad**2) + np.sum(p2.grad**2))
    assert abs(float(new_norm) - 6.5) < 1e-5
