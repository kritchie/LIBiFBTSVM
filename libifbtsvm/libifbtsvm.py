
from typing import Dict, Generator, Tuple

import numpy as np

from joblib import (  # type: ignore
    delayed,
    Parallel,
)
from numpy import linalg
from sklearn.svm import SVC

from libifbtsvm.functions import (
    fuzzy_membership,
    train_model,
)
from libifbtsvm.models.ifbtsvm import (
    ClassificationModel,
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
        self._classifiers: Dict = {}
        self.n_jobs = n_jobs
        self.kernel = parameters.kernel

    def decision_function(self, X):
        """
        Evalutes the decision function over X.

        :param X: Array of features to evaluate the decision on.
        :return: Array of decision evaluation.
        """
        pass

    def decrement(self, X: np.ndarray, y: np.ndarray):
        """

        :param X:
        :param y:
        :return:
        """
        # TODO : Implement classifier decrement here
        pass

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None):
        """
        Trains a iFBTSVM model

        :param X: The training samples
        :param y: The class labels for each training sample

        :param sample_weight: (Not supported)
        """
        X = self.kernel.fit_transform(X=X, y=y) if self.kernel else X  # type: ignore

        # Train the DAG models in parallel
        trained_hyperplanes = Parallel(n_jobs=self.n_jobs, prefer='threads')(
            delayed(self._fit_dag_step)(subset, self.parameters) for subset in self._generate_sub_sets(X, y)
        )

        # Create the DAG Model here
        for hypp in trained_hyperplanes:
            _clsf = self._classifiers.get(hypp.class_p, {})
            _clsf[hypp.class_n] = hypp
            self._classifiers[hypp.class_p] = _clsf

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

        return ClassificationModel(class_p=y_p[0],
                                   class_n=y_n[0],
                                   fuzzy=membership,
                                   weights_p=hyperplane_p,
                                   weights_n=hyperplane_n)

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

    def increment(self, X: np.ndarray, y: np.ndarray):
        """

        :param X:
        :param y:
        :return:
        """
        # TODO : Implement classifier increment here
        pass

    def predict(self, X):
        """
        Performs classification X.

        :param X: Array of features to classify
        :return: Array of classification result
        """
        X = self.kernel.transform(X=X) if self.kernel else X

        lh_keys = list(set(self._classifiers.keys()))

        rh_keys = set()
        for value in self._classifiers.values():
            for key, _ in value.items():
                rh_keys.add(key)
        rh_keys = list(rh_keys)

        classes = []

        for row in X:

            _dag_index_lh = 0
            _dag_index_rh = 0

            f_pos = 0
            f_neg = 0

            class_1 = None
            class_2 = None

            while True:
                try:
                    class_1 = lh_keys[_dag_index_lh]
                    class_2 = rh_keys[_dag_index_rh]

                    model: ClassificationModel = self._classifiers[class_1][class_2]

                    f_pos = np.divide(np.matmul(row, model.p.weights[:-1]) + model.p.weights[-1],
                                      linalg.norm(model.p.weights[:-1]))
                    f_neg = np.divide(np.matmul(row, model.n.weights[:-1]) + model.n.weights[-1],
                                      linalg.norm(model.n.weights[:-1]))

                    if abs(f_pos) < abs(f_neg):
                        _dag_index_lh += 1
                        _dag_index_rh += 1

                    else:
                        _dag_index_rh += 1

                except (StopIteration, IndexError):

                    if abs(f_pos) < abs(f_neg):
                        classes.append(class_2)

                    else:
                        classes.append(class_1)

                    break

        return classes

    def set_params(self, **params):
        """
        Sets parameters of the classifier

        :param params: Keyword arguments and their values
        :return: None
        """
        for key, val in params.items():
            setattr(self.parameters, key, val)

    def score(self, X, y, sample_weight=None):
        """
        Returns the accuracy of a classification.

        :param X: Array of features to classify
        :param y: Array of truth values for the features
        :param sample_weight: Not supported
        :return: Accuracy score of the classification
        """
        predictions = self.predict(X=X)
        accuracy = 0
        for i in range(len(y)):
            if predictions[i] == y[i]:
                accuracy += 1

        return accuracy / len(y)
