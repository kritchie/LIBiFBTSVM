
import itertools

from multiprocessing import pool
from typing import Generator, Tuple

import numpy as np

from sklearn.svm import SVC


DAGSubProbem = Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]


class ifbtsvm(SVC):

    def __init__(self, parameters, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parameters = parameters

    def decision_function(self, X):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None):
        """
        Trains a iFBTSVM model

        :param X: The training samples
        :param y: The class labels for each training sample

        :param sample_weight: (Not supported)
        """
        # Create the DAG Model here
        # with pool() as _pool:
        #     _pool.map(_, args)
        #
        # for clazz in classes:
        #
        #     index_samples = np.where(y == clazz)
        #     samples = X[index_samples]

    def increment(self, X: np.ndarray, y: np.ndarray):
        pass

    def decrement(self, X: np.ndarray, y: np.ndarray):
        pass

    @classmethod
    def generate_sub_sets(cls, X: np.ndarray, y: np.ndarray) -> DAGSubProbem:
        """
        Generates sub-data sets based on the DAG classification principle.

        Example, for 4 classes, the function will return the following:
        [0]: Values and labels of Class 1 and 2
        [1]: Values and labels of Class 1 and 3
        [2]: Values and labels of Class 1 and 4
        [3]: Values and labels of Class 2 and 3
        [4]: Values and labels of Class 2 and 4
        [5]: Values and labels of Class 3 and 4

        :param X: The full training set
        :param y: The full training labels set
        :return: Generator of tuple containing values and labels for positive and negative class
                 based on the current iteration in the classification DAG.

        - [0] Values for current X positive
        - [1] Labels for current X positive
        - [2] Values for current X negative
        - [3] Labels for current X negative
        """
        classes = np.unique(y)
        for _p in range(classes.size):

            for _n in range(_p + 1, classes.size):

                _index_p = np.where(y == classes[_p])
                _index_n = np.where(y == classes[_n])

                yield X[_index_p], y[_index_p], X[_index_n], y[_index_n]

    def get_params(self, deep=True):
        pass

    def predict(self, X):
        pass

    def score(self, X, y, sample_weight=None):
        pass

    def set_params(self, **params):
        pass
