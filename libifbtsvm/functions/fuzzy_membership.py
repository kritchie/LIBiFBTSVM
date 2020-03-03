
from typing import Dict, Tuple, Union

import numpy as np


class FuzzyMembership(object):
    """
    A FuzzyMembership object.
    """

    def __init__(self, sp: np.ndarray, sn: np.ndarray, noise_p: np.ndarray, noise_n: np.ndarray):
        self.sp = sp
        self.sn = sn
        self.noise_p = noise_p
        self.noise_n = noise_n


Radius = Union[np.ndarray, int, float, complex]


def _get_max_radius(a: np.ndarray, b: np.ndarray) -> Tuple[Radius, np.ndarray, np.ndarray]:
    """
    Radius function

    Based on (de Mello et al. 2019)'s radius computation for fuzzy membership.

    :return: Returns the radius of:
    -[0] maximum value for entries of class "a"
    -[1] entries of class "a" related to class "a"
    -[2] entries of class "b" related to class "a"
    """
    _mean_a = np.mean(a, axis=0)

    radius_a = np.max(np.abs(a - _mean_a))

    radius_b = np.sum(np.square(np.tile(_mean_a, (len(b), 1)) - a), axis=1)  # ||xi--Xcen+||^2
    radius_max_a = np.amax(radius_a)  # max value

    return radius_max_a, radius_a, radius_b


def _get_membership(max_radius_a, radis_a, radius_b, len_a, u, epsilon) -> Tuple[np.ndarray, np.ndarray]:

    membership = np.zeros(len_a)
    noise = np.greater_equal(radis_a, radius_b)

    _noise_index = np.nonzero(noise)[0]
    _normal_index = np.nonzero(np.invert(noise))[0]

    membership[_normal_index] = (1 - u) * (1 - np.square(np.absolute(radis_a[_normal_index]) / (max_radius_a + epsilon)))
    membership[_noise_index] = (1 - u) * (1 - np.square(np.absolute(radis_a[_noise_index]) / (max_radius_a + epsilon)))
    membership = np.expand_dims(membership, axis=1)
    return membership, noise


def fuzzy_membership(params: Dict, class_p: np.ndarray, class_n: np.ndarray) -> FuzzyMembership:
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

    epsilon = params.get('epsilon')
    if not epsilon:
        raise ValueError('Parameter "epsilon" cannot be None and must be a value greater than 0')

    u = params.get('u')
    if not u or not (0.0 <= u <= 1.0):
        raise ValueError('Parameter "u" cannot be None and must between 0.0 and 1.0')

    max_radius_p, radius_p_p, radius_p_n = _get_max_radius(class_p, class_n)
    max_radius_n, radius_n_n, radius_n_p = _get_max_radius(class_n, class_p)

    sp, noise_p = _get_membership(max_radius_p, radius_p_p, radius_p_n, len(class_p), u, epsilon)
    sn, noise_n = _get_membership(max_radius_n, radius_n_n, radius_n_p, len(class_n), u, epsilon)

    return FuzzyMembership(sp=sp, sn=sn, noise_p=noise_p, noise_n=noise_n)
