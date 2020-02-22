
from collections import namedtuple

import numpy as np
import pytest


# TODO : We can probably do better with those fixtures. This is a work in progress.

@pytest.fixture
def valid_features_a():
    return np.asarray([1, 2, 3, 4, 5]).reshape(1, -1)


@pytest.fixture
def invalid_features_a():
    return [1, 2, 3, 4, 5]


@pytest.fixture
def sum_kernel():
    Kernel = namedtuple('Kernel', 'fit_transform')

    def fit_transform(X: np.ndarray, y: np.ndarray):
        return X + y

    _kernel = Kernel(fit_transform=fit_transform)
    return _kernel
