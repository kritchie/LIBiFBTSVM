
import time

from sklearn.datasets import load_iris

from libifbtsvm import iFBTSVM
from libifbtsvm.models.ifbtsvm import Hyperparameters


if __name__ == '__main__':

    dataset = load_iris()
    params = Hyperparameters(
        epsilon=0.0000001,
        fuzzy=0.01,
        C1=8,
        C2=2,
        C3=8,
        C4=2,
        max_evals=500,
        phi=0.00001,
        kernel=None,
    )

    # Initialisation iFBTSVM
    ifbtsvm = iFBTSVM(parameters=params, n_jobs=1)

    _temp = dataset.data[:int(len(dataset.data)/2),:]

    # Training
    before = time.monotonic()
    ifbtsvm.fit(X=dataset.data[:int(len(dataset.data)/2)], y=dataset.target[:int(len(dataset.data)/2)])
    after = time.monotonic()
    elapsed = (after - before)

    # Prediction
    accuracy = ifbtsvm.score(X=dataset.data, y=dataset.target)
    print(f'Accuracy iFBTSVM: {accuracy * 100.0}% Train time: {elapsed}s')

    dataset2 = load_iris()

    before = time.monotonic()
    ifbtsvm.update(X=dataset2.data[int(len(dataset.data)/2):], y=dataset.target[int(len(dataset.data)/2):],
                   batch_size=20)
    after = time.monotonic()
    elapsed = (after - before)

    # Prediction
    accuracy = ifbtsvm.score(X=dataset.data, y=dataset.target)
    print(f'Accuracy Updated iFBTSVM: {accuracy * 100.0}% Train time: {elapsed}s')
