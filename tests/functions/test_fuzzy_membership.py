
import pytest

import numpy as np

from libifbtsvm.functions.fuzzy_membership import (
    _get_membership,
    fuzzy_membership
)


def test_fuzzy_membership():

    class_p = np.asarray([
        [1, 1],
        [0.5, 1],
        [1, 0.5]
    ])

    class_n = np.asarray([
        [4, 4],
        [4.5, 4],
        [4, 4.5]
    ])

    params = {
        'u': 0.5,
        'epsilon': 0.5,
    }

    _fuzzy = fuzzy_membership(params, class_p, class_n)

    _truth = np.asarray([[0.49621928], [0.47637051], [0.47637051]])

    assert np.isclose(_fuzzy.sp, _truth).all()
    assert np.isclose(_fuzzy.sn, _truth).all()
    assert np.invert(np.nonzero(_fuzzy.noise_p)).all()
    assert np.invert(np.nonzero(_fuzzy.noise_n)).all()
