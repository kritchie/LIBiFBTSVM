
from typing import Dict, Tuple, Union

import numpy as np


class FuzzyMembership(object):
    """
    A FuzzyMembership object.
    """

    def __init__(self, sp: float, sn: float, noise_p: float, noise_n: float):
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
    - entries of "a" related to "a" (radius_a)
    - entries of "b" related to "a" (radius_b)
    - maximum value of "a" (radius_a)
    """
    mean_a = np.mean(a, axis=0)

    radius_a = np.max(np.abs(a - mean_a))

    radius_b = np.sum(np.square(np.tile(mean_a, (len(b), 1)) - a), axis=1)  # ||xi--Xcen+||^2
    radius_max_a = np.amax(radius_a)  # max value

    return radius_max_a, radius_a, radius_b


def fuzzy_membership(params: Dict, class_p: np.ndarray, class_n: np.ndarray) -> FuzzyMembership:
    """
    This method computes the fuzzy membership of feature vectors for positive and negative class.

    :param params: Control parameters for calculating the membership functions. These parameters are
                   defined by (de Mello et al. 2019)'s fuzzy membership function.

                   Parameters are:
                   - "epsilon" (float): Must be > 0 to avoid fuzzy membership to be equal to 0.
                   - "u" (float): Must be between 0.0 and 1.0, to balance effect of normal/noisy data points.

    :param class_p: Numpy arrays of features. Holds vectors of the positive class.
    :param class_n: Numpy arrays of features. Holds vectors of the negative class.
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

    noise_p = np.where(radius_p_p >= radius_p_n)
    normal_p = np.where(radius_p_n >= radius_p_p)

    # TODO Add S_p

    noise_n = np.where(radius_n_n >= radius_n_p)
    normal_n = np.where(radius_n_p >= radius_n_n)

    # TODO Add S_n

    return FuzzyMembership(sp=0.0, sn=0.0, noise_p=0.0, noise_n=0.0)  # FIXME : Implement me !
