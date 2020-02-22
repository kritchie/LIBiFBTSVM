
import pytest

import numpy as np


@pytest.fixture
def valid_features_a():
    yield np.asarray([1, 2, 3, 4, 5]).reshape(1, -1)


@pytest.fixture
def invalid_features_a():
    yield [1, 2, 3, 4, 5]
