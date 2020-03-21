
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
    """
    epsilon: float
    u: float
    CC: float
    CC2: float
    CR: float
    CR2: float
    max_evaluations: int


class Hyperplane(object):
    """
    Object representing the values for describing a classification hyperplane.
    """
    alpha: np.ndarray
    weights: np.ndarray
    iterations: int
    projected_gradients: np.ndarray
