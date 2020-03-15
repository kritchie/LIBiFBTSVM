
from typing import Dict, Generator, Tuple

import numpy as np

from joblib import (
    delayed,
    Parallel,
    parallel_backend,
)

from sklearn.svm import SVC

from libifbtsvm.functions import (
    fuzzy_membership,
    FuzzyMembership
)


TrainingSet = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
DAGSubSet = Generator[TrainingSet, None, None]


class ifbtsvm(SVC):

    # TODO Define a descriptive class for parameters
    def __init__(self, parameters, *args, n_jobs=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.parameters = parameters
        self.classifiers = None
        self.n_jobs = n_jobs

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
        # TODO : Possible improvement here would be to use shared memory
        #      : instead of copying the data each time. This could save a lot
        #      : of memory if the number of classes is high
        with parallel_backend(backend='loky', n_jobs=self.n_jobs):
            self.classifiers = Parallel()(delayed(self._fit_dag_step)
                                          (subset, self.parameters) for subset in self._generate_sub_sets(X, y))

        # TODO Implement building of DAG classifier logic

    @classmethod
    def _fit_dag_step(cls, subset: TrainingSet, parameters: Dict = None):
        """
        Trains a classifier based on a sub-set of data, as a step in the DAG classifier algorithm.

        :param subset: Sub-set of data containing the training data for this DAG step
        """
        x_p = subset[0]
        y_p = subset[1]

        x_n = subset[2]
        y_n = subset[3]

        # Calculate fuzzy membership
        membership = fuzzy_membership(params=parameters, class_p=x_p, class_n=x_n)

        # FIXME Get correct notation
        # Build H matrix which is [X_p/n, e] where "e" is a column of ones ("1")
        H_p = np.append(x_p, np.ones(x_p.shape[1], 1), axis=1)
        H_n = np.append(x_n, np.ones(x_n.shape[1], 1), axis=1)

        # FIXME find explanation in the paper
        CCp = parameters.get('CC') * membership.sn
        CCn = parameters.get('CC2') * membership.sp

        CRp = parameters.get('CR2')
        CRn = parameters.get('CR')

        return {'class_p': y_p[0], 'class_n': y_n[0], 'classifier': {}}

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
    def _generate_sub_sets(cls, X: np.ndarray, y: np.ndarray) -> DAGSubSet:
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
