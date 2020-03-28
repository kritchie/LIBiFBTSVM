from collections import namedtuple

import numpy as np
import pytest


Dataset = namedtuple('Dataset', ['X', 'y'])


@pytest.fixture
def dataset_3_classes():

    # Tuple of (feature,label)
    dataset = [
        (0.9, '1'),
        (1.0, '1'),
        (1.1, '1'),
        (10.9, '2'),
        (11.0, '2'),
        (11.1, '2'),
        (110.9, '3'),
        (111.0, '3'),
        (111.1, '3'),
    ]

    return Dataset(X=np.array([x[0] for x in dataset]),
                   y=np.array([y[1] for y in dataset]))
