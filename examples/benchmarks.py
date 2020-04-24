
import time
import random

import numpy as np

from sklearn.datasets import load_iris as load_dataset

from libifbtsvm import iFBTSVM
from libifbtsvm.models.ifbtsvm import Hyperparameters


if __name__ == '__main__':

    dataset = load_dataset()

    # Samples
    # _len_d = len(dataset.data)
    # _rand_indices = [random.randint(0, _len_d) for _ in range(1000)]
    #
    # dataset.data = dataset.data[_rand_indices]
    # dataset.target = dataset.target[_rand_indices]

    params = Hyperparameters(
        epsilon=0.0000001,
        fuzzy=0.01,
        C1=8,
        C2=2,
        C3=8,
        C4=2,
        max_iter=1,
        phi=0.00001,
        kernel=None,
    )

    # Initialisation iFBTSVM with 3 jobs for parallel training
    ifbtsvm = iFBTSVM(parameters=params, n_jobs=1)

    # Training
    before = time.monotonic()
    for i in range(100):
        ifbtsvm.fit(X=dataset.data, y=dataset.target)
    after = time.monotonic()
    elapsed = (after - before)
    print(f'Train time: {elapsed}s')
    # #
    # # Prediction
    # accuracy = ifbtsvm.score(X=dataset.data, y=dataset.target)
    # print(f'Accuracy iFBTSVM: {accuracy * 100.0}%')
    #
    # # Training
    # from sklearn.svm import LinearSVC
    # svm = LinearSVC(max_iter=100)
    # before = time.monotonic()
    # svm.fit(X=dataset.data, y=dataset.target)
    # after = time.monotonic()
    # elapsed = (after - before)
    # print(f'Train time: {elapsed}s')
    #
    # # Prediction
    # accuracy = svm.score(X=dataset.data, y=dataset.target)
    # print(f'Accuracy LinearSVC: {accuracy * 100.0}%')
