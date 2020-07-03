
import pytest

import numpy as np

from libifbtsvm import iFBTSVM, Hyperparameters


def test_compute_score_none():
    params = Hyperparameters(phi=0.5)
    svm = iFBTSVM(parameters=params)

    score = None
    c = np.zeros((5,))
    for i in range(5):
        c[i] = i + 1

    _score = svm._compute_score(score, c)

    assert np.array_equal(_score, np.asarray([[1, 2, 3, 4, 5], [1, 1, 1, 1, 1]]))


def test_compute_score():
    params = Hyperparameters(phi=0.5)
    svm = iFBTSVM(parameters=params)

    score = np.asarray([[1, 2, 3, 4, 5], [1, 1, 1, 1, 1]])
    c = np.zeros((5,))
    for i in range(5):
        c[i] = i + 1

    _score = svm._compute_score(score, c)

    assert np.array_equal(_score, np.asarray([[1, 2, 3, 4, 5], [2, 2, 2, 2, 2]]))

    score = np.asarray([[1, 2, 3, 4, 5], [1, 1, 1, 1, 1]])
    c = np.zeros((10,))
    for i in range(10):
        c[i] = i + 1

    _score = svm._compute_score(score, c)

    assert np.array_equal(_score, np.asarray([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                              [2, 2, 2, 2, 2, 1, 1, 1, 1, 1]]))


def test_decrement():

    score = np.asarray([[0, 1, 2], [1, 2, 3]])
    alphas = np.asarray([0, 1, 2])
    fuzzy = np.asarray([0, 1, 2])
    data = np.asarray([0, 1, 2])

    score, alphas, fuzzy, data = iFBTSVM._decrement([0], score, alphas, fuzzy, data)

    assert np.array_equal(score, np.asarray([[1, 2], [2, 3]]))
    assert np.array_equal(alphas, np.asarray([1, 2]))
    assert np.array_equal(fuzzy, np.asarray([1, 2]))
    assert np.array_equal(data, np.asarray([1, 2]))


def test_filter_gradients():

    weights = np.asarray([-0.07136844, -0.21096315, 0.31555559, 0.14247409, -0.5475241])
    gradients = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5])
    data = np.asarray([[5.4, 3.4, 1.7, 0.2],
                       [5.1, 3.4, 1.5, 0.2],
                       [4.6, 3.1, 1.5, 0.2],
                       [4.7, 3.2, 1.6, 0.2],
                       [5.3, 3.7, 1.5, 0.2]])
    label = np.ones((5, 1))

    filtered = iFBTSVM._filter_gradients(weights=weights, gradients=gradients, data=data, label=label)

    assert np.array_equal(filtered[0], np.asarray([[5.1, 3.4, 1.5, 0.2], [5.3, 3.7, 1.5, 0.2]]))
    assert np.array_equal(filtered[1], np.asarray([1, 1]))
