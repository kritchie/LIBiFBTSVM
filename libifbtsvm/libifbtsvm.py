
from multiprocessing import pool
from typing import Generator, Tuple

import numpy as np

from sklearn.svm import SVC


DAGSubProbem = Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]


class ifbtsvm(SVC):

    def __init__(self, parameters, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parameters = parameters
        self.classifiers = {}

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

        # TODO : Get the DAGs generator
        # TODO : multiprocessing, call classify for each dag
        # TODO : Set self.classifiers[p][n] = classifier

    def increment(self, X: np.ndarray, y: np.ndarray):
        """

        :param X:
        :param y:
        :return:
        """
        # TODO : Implement classifier increment here
        pass

    def decrement(self, X: np.ndarray, y: np.ndarray):
        """

        :param X:
        :param y:
        :return:
        """
        # TODO : Implement classifier decrement here
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
        """
        Returns parameters of the classifier.

        :param deep: Not-implemented, as no effect and is kept to comply with sklearn.svm.SVC interface
        :return: A dictionary of parameters
        """
        return self.parameters

    def predict(self, X):
        """

        :param X:
        :return:
        """
        # TODO : implement DAG prediction here
        pass

    def score(self, X, y, sample_weight=None):
        """

        :param X:
        :param y:
        :param sample_weight:
        :return:
        """
        # TODO : implement DAG scoring here
        pass

    def set_params(self, **params):
        """
        Sets parameters of the classifier

        :param params: Keyword arguments and their values
        :return: None
        """
        for key, val in params.items():
            self.parameters[key] = val
