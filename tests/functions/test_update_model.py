
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
    # TODO Implement Me
    pass


def test_decrement():

    repetition = 5
    score = np.zeros((2, 2))

    score[1][1] = 6

    c = np.zeros((2, 1))
    alphas = np.zeros((2, 1))
    fuzzy = np.zeros((2, 1))
    data = np.zeros((2, 1))

    score, alphas, fuzzy, data = iFBTSVM._decrement(repetition, score, c, alphas, fuzzy, data)

    assert np.array_equal(score, np.zeros((2, 1)))
    assert np.array_equal(alphas, np.zeros((2, 1)))
    assert np.array_equal(fuzzy, np.zeros((2, 1)))
    assert np.array_equal(data, np.zeros((2, 1)))


def test_filter_gradients():
    # TODO : Implement Me
    pass


def test_increment_dag_step():
    # TODO : Implement Me
    pass


