import numpy as np

from puredl.layers import MaxPool2D


def test_maxpool_known_forward_and_backward():
    x = np.array(
        [[[[1.0, 3.0, 2.0, 0.0],
           [4.0, 6.0, 5.0, 1.0],
           [0.0, 2.0, 8.0, 7.0],
           [1.0, 3.0, 4.0, 9.0]]]],
        dtype=np.float64,
    )
    pool = MaxPool2D(2, stride=2)
    out = pool.forward(x)
    expected = np.array([[[[6.0, 5.0], [3.0, 9.0]]]])
    np.testing.assert_array_equal(out, expected)

    grad = pool.backward(np.ones_like(out))
    expected_grad = np.zeros_like(x)
    expected_grad[0, 0, 1, 1] = 1.0
    expected_grad[0, 0, 1, 2] = 1.0
    expected_grad[0, 0, 3, 1] = 1.0
    expected_grad[0, 0, 3, 3] = 1.0
    np.testing.assert_array_equal(grad, expected_grad)
