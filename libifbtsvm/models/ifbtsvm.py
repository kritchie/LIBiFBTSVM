
from typing import Union

import numpy as np

from sklearn.kernel_approximation import TransformerMixin, BaseEstimator

Kernel = (TransformerMixin, BaseEstimator)
Label = Union[int, str]


class FuzzyMembership(object):

    def __init__(self, sp: np.ndarray, sn: np.ndarray, noise_p: np.ndarray, noise_n: np.ndarray):
        """
        Creates an object representing fuzzy membership values for a data set

        # TODO : Complete docs
        :param sp:
        :param sn:
        :param noise_p:
        :param noise_n:
        """
        self.sp = sp
        self.sn = sn
        self.noise_p = noise_p
        self.noise_n = noise_n


class Hyperparameters(object):

    def __init__(self, epsilon=None, fuzzy=None, C1=None, C2=None, C3=None, C4=None,
                 max_evals=None, phi=None, kernel=None, repetition=None):
        """
        Creates an object representing the hyperparameters of a classification model

        # TODO : Complete docs
        :param epsilon:
        :param fuzzy:
        :param C1:
        :param C2:
        :param C3:
        :param C4:
        :param max_iter:
        :param phi:
        :param repition:
        """
        self.epsilon: float = epsilon
        self.fuzzy_parameter: float = fuzzy
        self.C1: float = C1
        self.C2: float = C2
        self.C3: float = C3
        self.C4: float = C4
        self.max_iter: int = max_iter
        self.phi: float = phi
        self.kernel: Kernel = kernel
        self.repetition = repetition


class Hyperplane(object):

    def __init__(self, alpha=None, weights=None, iterations=None, proj_gradients=None):
        """
        Creates an object representing the values for describing a classification hyperplane.

        # TODO : Complete docs
        :param alpha:
        :param weights:
        :param iterations:
        :param proj_gradients:
        """
        self.alpha: np.ndarray = alpha
        self.weights: np.ndarray = weights
        self.iterations: int = iterations
        self.projected_gradients: np.ndarray = proj_gradients


class ClassificationModel(object):

    def __init__(self, fuzzy: FuzzyMembership, weights_p: Hyperplane, weights_n: Hyperplane,
                 class_p: Label, class_n: Label, data_p: np.ndarray, data_n: np.ndarray):
        """
        Creates an object representing a classification model

        # TODO : Complete docs
        :param fuzzy:
        :param weights_p:
        :param weights_n:
        :param class_p:
        :param class_n:
        :param data_p:
        :param data_n:
        """
        self.fuzzy_membership: FuzzyMembership = fuzzy
        self.p: Hyperplane = weights_p
        self.n: Hyperplane = weights_n
        self.class_p: Label = class_p
        self.class_n: Label = class_n
        self.data_p: np.ndarray = data_p
        self.data_n: np.ndarray = data_n
