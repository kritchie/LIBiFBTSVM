
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


class Hyperparameters(object):
    """
    Object representing the hyperparameters of a model.

    Parameter.C1 = int value (eg. 8) %C1
    Parameter.C3= int value (eg. 8)  %C3
    Parameter.C2 = int value (eg. 2) %C2
    Parameter.C4 = int value (eg. 2)

    """
    epsilon: float
    fuzzy_parameter: float
    C1: float
    C2: float
    C3: float
    C4: float
    max_evaluations: int
    phi: float


class Hyperplane(object):
    """
    Object representing the values for describing a classification hyperplane.
    """
    alpha: np.ndarray
    weights: np.ndarray
    iterations: int
    projected_gradients: np.ndarray
