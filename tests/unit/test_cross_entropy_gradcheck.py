import numpy as np

from puredl.losses import CrossEntropyLoss
from puredl.utils import finite_difference_gradient, relative_error


def test_cross_entropy_gradient_matches_finite_difference():
    logits = np.array(
        [[0.2, -0.3, 1.1], [1.2, 0.4, -0.7]],
        dtype=np.float64,
    )
    targets = np.array([2, 0], dtype=np.int64)
    loss = CrossEntropyLoss()
    loss.forward(logits, targets)
    analytic = loss.backward().copy()

    numerical_logits = logits.copy()

    def objective():
        return CrossEntropyLoss().forward(numerical_logits, targets)

    numerical = finite_difference_gradient(numerical_logits, objective, eps=1e-6)
    assert relative_error(analytic, numerical) < 1e-7
