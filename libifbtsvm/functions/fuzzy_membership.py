
from typing import Tuple, Union

import numpy as np

from libifbtsvm.models.ifbtsvm import (
    FuzzyMembership,
    Hyperparameters,
)


Radius = Union[np.ndarray, int, float, complex]


def _get_membership(max_radius_a, radis_a, radius_b, len_a, u, epsilon) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the membership and noise values based on pre-computed radius values

    :return:
    -[0] membership values for class "a"
    -[1] noise values for class "a"
    """
    membership = np.zeros(len_a)
    noise = np.greater_equal(radis_a, radius_b)

    _noise_index = np.nonzero(noise)[0]
    _normal_index = np.nonzero(np.invert(noise))[0]

    membership[_normal_index] = u * (1 - np.square(np.absolute(radis_a[_normal_index]) / (max_radius_a + epsilon)))
    membership[_noise_index] = (1 - u) * (1 - np.square(np.absolute(radis_a[_noise_index]) / (max_radius_a + epsilon)))
    membership = np.expand_dims(membership, axis=1)

    return membership, noise


def fuzzy_membership(params: Hyperparameters, class_p: np.ndarray, class_n: np.ndarray) -> FuzzyMembership:
    """
    This method computes the fuzzy membership of feature vectors for positive and negative class.

    :param params: Control parameters for calculating the membership functions. These parameters are
                   defined by (de Mello et al. 2019)'s fuzzy membership function.

                   Parameters are:
                   - "epsilon" (float): Must be > 0 to avoid fuzzy membership to be equal to 0.
                   - "u" (float): Must be between 0.0 and 1.0, to balance effect of normal/noisy data points.

    :param class_p: Numpy arrays of features. Holds vectors for the positive class.
    :param class_n: Numpy arrays of features. Holds vectors for the negative class.
    :return: A FuzzyMembership object describing the fuzzy membership for vectors of both classes.
    """

    epsilon = params.epsilon
    if not epsilon or not isinstance(epsilon, float) or not epsilon > 0:
        raise ValueError('Parameter "epsilon" cannot be None and must be a floating value greater than 0')

    u = params.u
    if not u or not isinstance(u, float) or not (0.0 <= u <= 1.0):
        raise ValueError('Parameter "u" cannot be None and must be a floating value between 0.0 and 1.0')

    _mean_p = np.mean(class_p, axis=0)
    _mean_n = np.mean(class_n, axis=0)

    _size_p = len(class_p)
    _size_n = len(class_n)

    radius_p_p = np.sum(np.square(np.tile(_mean_p, (_size_p, 1)) - class_p), axis=1)
    radius_p_n = np.sum(np.square(np.tile(_mean_n, (_size_p, 1)) - class_p), axis=1)
    max_radius_p = np.amax(radius_p_p)

    radius_n_n = np.sum(np.square(np.tile(_mean_n, (_size_n, 1)) - class_n), axis=1)
    radius_n_p = np.sum(np.square(np.tile(_mean_p, (_size_n, 1)) - class_n), axis=1)
    max_radius_n = np.amax(radius_n_n)

    sp, noise_p = _get_membership(max_radius_p, radius_p_p, radius_p_n, len(class_p), u, epsilon)
    sn, noise_n = _get_membership(max_radius_n, radius_n_n, radius_n_p, len(class_n), u, epsilon)

    return FuzzyMembership(sp=sp, sn=sn, noise_p=noise_p, noise_n=noise_n)
