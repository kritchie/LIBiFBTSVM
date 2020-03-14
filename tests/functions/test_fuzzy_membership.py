
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


def test_fuzzy_membership_no_noise(valid_ensemble_a, valid_ensemble_b):

    params = {
        'u': 0.5,
        'epsilon': 0.5,
    }

    _fuzzy = fuzzy_membership(params=params, class_p=valid_ensemble_a, class_n=valid_ensemble_b)

    _truth = np.asarray([[0.49621928], [0.47637051], [0.47637051]])

    assert np.isclose(_fuzzy.sp, _truth).all()
    assert np.isclose(_fuzzy.sn, _truth).all()

    assert not np.nonzero(_fuzzy.noise_p)[0].size > 0
    assert not np.nonzero(_fuzzy.noise_n)[0].size > 0


def test_fuzzy_membership_noise(valid_ensemble_a, valid_ensemble_b):

    params = {
        'u': 0.5,
        'epsilon': 0.5,
    }

    # Update ensemble "b" to have a point closer to center of "a"
    valid_ensemble_b[0][0] = 0.8
    valid_ensemble_b[0][1] = 0.8

    _fuzzy = fuzzy_membership(params=params, class_p=valid_ensemble_a, class_n=valid_ensemble_b)

    _truth_p = np.asarray([[0.49621928], [0.47637051], [0.47637051]])
    _truth_n = np.asarray([[0.04410816], [0.46875], [0.46875]])

    assert np.isclose(_fuzzy.sp, _truth_p).all()
    assert np.isclose(_fuzzy.sn, _truth_n).all()

    assert not np.nonzero(_fuzzy.noise_p)[0].size > 0
    assert np.nonzero(_fuzzy.noise_n)[0] == [0]
