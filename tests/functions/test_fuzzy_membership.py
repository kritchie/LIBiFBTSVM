
import pytest

import numpy as np

from libifbtsvm.functions.fuzzy_membership import (
    fuzzy_membership
)


@pytest.mark.parametrize('u, epsilon', [
    (-0.1, 0.5),
    (None, 0.5),
    ('test', 0.5),
    (1.1, 0.5),
])
def test_fuzzy_membership_u_error(valid_ensemble_a, valid_ensemble_b, u, epsilon):
    params = {
        'u': u,
        'epsilon': epsilon,
    }

    with pytest.raises(ValueError):
        _ = fuzzy_membership(params, valid_ensemble_a, valid_ensemble_b)


@pytest.mark.parametrize('epsilon, u', [
    (-0.1, 0.5),
    (None, 0.5),
    ('test', 0.5),
])
def test_fuzzy_membership_epsilon_error(valid_ensemble_a, valid_ensemble_b, u, epsilon):
    params = {
        'u': u,
        'epsilon': epsilon,
    }

    with pytest.raises(ValueError):
        _ = fuzzy_membership(params, valid_ensemble_a, valid_ensemble_b)


def test_fuzzy_membership_valid(valid_ensemble_a, valid_ensemble_b):

    params = {
        'u': 0.5,
        'epsilon': 0.5,
    }

    _fuzzy = fuzzy_membership(params, valid_ensemble_a, valid_ensemble_b)

    _truth = np.asarray([[0.49621928], [0.47637051], [0.47637051]])

    assert np.isclose(_fuzzy.sp, _truth).all()
    assert np.isclose(_fuzzy.sn, _truth).all()
    assert np.invert(np.nonzero(_fuzzy.noise_p)).all()
    assert np.invert(np.nonzero(_fuzzy.noise_n)).all()
