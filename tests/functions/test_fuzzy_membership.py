
import pytest

import numpy as np

from sklearn.datasets import load_iris

from libifbtsvm import Hyperparameters
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
    params = Hyperparameters(fuzzy=u, epsilon=epsilon)

    with pytest.raises(ValueError):
        _ = fuzzy_membership(params, valid_ensemble_a, valid_ensemble_b)


@pytest.mark.parametrize('epsilon, u', [
    (-0.1, 0.5),
    (None, 0.5),
    ('test', 0.5),
])
def test_fuzzy_membership_epsilon_error(valid_ensemble_a, valid_ensemble_b, u, epsilon):
    params = Hyperparameters(fuzzy=u, epsilon=epsilon)

    with pytest.raises(ValueError):
        _ = fuzzy_membership(params, valid_ensemble_a, valid_ensemble_b)


def test_fuzzy_membership_no_noise(valid_ensemble_a, valid_ensemble_b):
    params = Hyperparameters(fuzzy=0.5, epsilon=0.5)

    _fuzzy = fuzzy_membership(params=params, class_p=valid_ensemble_a, class_n=valid_ensemble_b)

    _truth = np.asarray([1.0, 0.5, 0.5])

    assert np.isclose(_fuzzy.sp, _truth).all()
    assert np.isclose(_fuzzy.sn, _truth).all()

    assert not np.nonzero(_fuzzy.noise_p)[0].size > 0
    assert not np.nonzero(_fuzzy.noise_n)[0].size > 0


def test_fuzzy_membership_noise(valid_ensemble_a, valid_ensemble_b):
    params = Hyperparameters(fuzzy=0.5, epsilon=0.5)

    # Update ensemble "b" to have a point closer to center of "a"
    valid_ensemble_b[0][0] = 0.8
    valid_ensemble_b[0][1] = 0.8

    _fuzzy = fuzzy_membership(params=params, class_p=valid_ensemble_a, class_n=valid_ensemble_b)

    _truth_p = np.asarray([1.0, 0.5, 0.5])
    _truth_n = np.asarray([0.5, 1.0, 1.0])

    assert np.isclose(_fuzzy.sp, _truth_p).all()
    assert np.isclose(_fuzzy.sn, _truth_n).all()

    assert not np.nonzero(_fuzzy.noise_p)[0].size > 0
    assert np.nonzero(_fuzzy.noise_n)[0] == [0]


def test_fuzzy_membership_iris():
    params = Hyperparameters(fuzzy=0.5, epsilon=0.5)
    dataset = load_iris()

    x_p = dataset.data[np.where(dataset.target == 0)]
    x_n = dataset.data[np.where(dataset.target == 1)]

    membership = fuzzy_membership(params=params, class_p=x_p, class_n=x_n)

    _truth = np.asarray([[0.80251715], [0.97998912], [0.812318], [0.9807224], [0.98189354], [0.99937236],
                         [0.96043452], [0.56903383], [0.97212066], [0.96048151], [0.60125974], [0.99984087],
                         [0.97872578], [0.99542787], [0.97290636], [0.95614206], [0.99496593], [0.99843223],
                         [0.9804573], [0.9889402], [0.95800151], [0.99971397], [0.96464217], [0.99565949],
                         [0.99577327], [0.97731756], [0.90653641], [0.84949026], [0.999488], [0.94963879],
                         [0.97055683], [0.95245829], [0.9980525], [0.94679333], [0.98472415], [0.9759757],
                         [0.92815776], [0.98848116], [0.99733116], [0.99105587], [0.99493091], [0.9970333],
                         [0.99903366], [0.61157738], [0.9992254], [0.99909612], [1.], [0.99982867], [0.5],
                         [0.99991744]])
    np.testing.assert_allclose(membership.sn, _truth)

    _truth = np.asarray([[0.99992168], [0.9917287], [0.99376558], [0.98430655], [0.99974305], [0.95670035],
                        [0.99387979], [1.], [0.91239646], [0.99587275], [0.98883625], [0.99914972], [0.98704236],
                        [0.85665848], [0.78204237], [0.56573912], [0.96225372], [0.99991496], [0.90482216],
                        [0.99526856], [0.99049658], [0.99760092], [0.96536234], [0.99558788], [0.98841007],
                        [0.99139502], [0.99961253], [0.99956062], [0.99959803], [0.99427123], [0.99396476],
                        [0.99323856], [0.94598095], [0.85250365], [0.99691726], [0.99689705], [0.98412401],
                        [0.99910674], [0.93096412], [0.99996815], [0.99976035], [0.5], [0.95871688], [0.99539269],
                        [0.9728796], [0.98887459], [0.99416071], [0.98977514], [0.99446211], [0.99990073]])

    np.testing.assert_allclose(membership.sp, _truth)
