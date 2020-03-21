
from typing import Generator, Tuple

import numpy as np

from joblib import (  # type: ignore
    delayed,
    Parallel,
    parallel_backend,
)

from sklearn.svm import SVC

from libifbtsvm.functions import (
    fuzzy_membership,
    train_model,
)
from libifbtsvm.models.ifbtsvm import (
    FuzzyMembership,
    Hyperparameters,
    Hyperplane,
)

TrainingSet = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
DAGSubSet = Generator[TrainingSet, None, None]


class iFBTSVM(SVC):

    def __init__(self, parameters: Hyperparameters, *args, n_jobs=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.parameters = parameters
        self._classifiers = {}
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
        #      : instead of copying the data each time. This could save
        #      : a non-negligible amount of memory if the number of classes is high.
        with parallel_backend(backend='loky', n_jobs=self.n_jobs):
            trained_hyperplanes = Parallel()(delayed(self._fit_dag_step)
                                             (subset, self.parameters) for subset in self._generate_sub_sets(X, y))

        for hypp in trained_hyperplanes:
            _clsf = self._classifiers.get(hypp['classP'], {})
            _clsf[hypp['classN']] = hypp
            self._classifiers[hypp['classP']] = _clsf

        # TODO Implement building of DAG classifier logic

    @classmethod
    def _fit_dag_step(cls, subset: TrainingSet, parameters: Hyperparameters):
        """
        Trains a classifier based on a sub-set of data, as a step in the DAG classifier algorithm.

        :param subset: Sub-set of data containing the training data for this DAG step
        """
        # Features (x_p) of the current "positive" class
        x_p = subset[0]
        y_p = subset[1]

        # Features (x_n) of the current "negative" class
        x_n = subset[2]
        y_n = subset[3]

        # Calculate fuzzy membership for points
        membership: FuzzyMembership = fuzzy_membership(params=parameters, class_p=x_p, class_n=x_n)

        # Build H matrix which is [X_p/n, e] where "e" is an extra column of ones ("1") appended at the end of the
        # matrix
        # i.e.
        #
        #   if  X_p = | 1  2  3 |  and   e = | 1 |  then  H_p = | 1 2 3 1 |
        #             | 4  5  6 |            | 1 |              | 4 5 6 1 |
        #             | 7  8  9 |            | 1 |              | 7 8 9 1 |
        #
        H_p = np.append(x_p, np.ones((x_p.shape[0], 1)), axis=1)
        H_n = np.append(x_n, np.ones((x_n.shape[0], 1)), axis=1)

        _C1 = parameters.C1 * membership.sn
        _C3 = parameters.C3 * membership.sp

        _C2 = parameters.C2
        _C4 = parameters.C4

        # Train the model using the algorithm described by (de Mello et al. 2019)
        hyperplane_p: Hyperplane = train_model(parameters=parameters, H=H_n, G=H_p, C=_C4, CCx=_C3)
        hyperplane_n: Hyperplane = train_model(parameters=parameters, H=H_p, G=H_n, C=_C2, CCx=_C1)
        hyperplane_n.weights = -hyperplane_n.weights

        return {'hyperplaneP': hyperplane_p, 'hyperplaneN': hyperplane_n, 'fuzzyMembership': membership,
                'classP': y_p[0], 'classN': y_n[0]}

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
