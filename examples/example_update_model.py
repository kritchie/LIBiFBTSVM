
import time

from sklearn.datasets import load_iris as load_data
from sklearn.kernel_approximation import RBFSampler
from sklearn.utils import shuffle

from libifbtsvm import iFBTSVM
from libifbtsvm.models.ifbtsvm import Hyperparameters


if __name__ == '__main__':

    dataset = load_data()
    dataset.data, dataset.target = shuffle(dataset.data, dataset.target)

    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.01,
        C1=8,
        C2=2,
        C3=8,
        C4=2,
        max_evals=500,
        phi=1e-9,
        kernel=RBFSampler(gamma=1, n_components=20),
        repetition=5,
    )

    # Initialisation iFBTSVM
    ifbtsvm = iFBTSVM(parameters=params, n_jobs=1)

    # Training
    before = time.monotonic()
    ifbtsvm.fit(X=dataset.data[:int(len(dataset.data) / 2)], y=dataset.target[:int(len(dataset.data) / 2)])
    after = time.monotonic()
    elapsed = (after - before)

    # Prediction
    accuracy = ifbtsvm.score(X=dataset.data, y=dataset.target)
    print(f'Accuracy iFBTSVM: {accuracy * 100.0}% Train time: {elapsed}s')

    # Update Model
    before = time.monotonic()
    ifbtsvm.update(X=dataset.data[int(len(dataset.data) / 2):], y=dataset.target[int(len(dataset.data) / 2):],
                   batch_size=20)
    after = time.monotonic()
    elapsed = (after - before)

    # Prediction
    accuracy = ifbtsvm.score(X=dataset.data, y=dataset.target)
    print(f'Accuracy Updated iFBTSVM: {accuracy * 100.0}% Train time: {elapsed}s')
