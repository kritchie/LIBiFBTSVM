import os
import numpy as np
import pandas as pd
import time

from sklearn.kernel_approximation import RBFSampler

from libifbtsvm import iFBTSVM, Hyperparameters


DATA_DIR = os.getenv('DATA_DIR', './data')


def rbf():
    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.1,
        C1=8,
        C2=2,
        C3=8,
        C4=2,
        max_iter=250,
        phi=0,
        kernel=RBFSampler(gamma=0.45, n_components=300),
        forget_score=10,
    )

    _data = pd.read_csv(f'{DATA_DIR}/RBF/10K/RBF10K.csv')
    train_data = _data.values[:10000, 0:10]
    train_label = _data.values[:10000, 10:]
    test_data = _data.values[10000:, 0:10]
    test_label = _data.values[10000:, 10:]

    ifbtsvm = iFBTSVM(parameters=params, n_jobs=4)

    # Training
    num_points = 5000
    before = time.monotonic()
    ifbtsvm.fit(X=train_data[:num_points],
                y=train_label[:num_points].reshape(train_label[:num_points].shape[0]))
    after = time.monotonic()
    elapsed = (after - before)
    accuracy_1 = ifbtsvm.score(X=test_data, y=test_label)

    # Update
    batch_size = int(len(train_data) / 100 * 5 + 0.5)  # 5% of original dataset
    before = time.monotonic()
    ifbtsvm.update(X=train_data[num_points:],
                   y=train_label[num_points:].reshape(train_label[num_points:].shape[0]),
                   batch_size=batch_size)
    after = time.monotonic()
    u_elapsed = after - before

    # Prediction
    accuracy_2 = ifbtsvm.score(X=test_data, y=test_label)
    print(f'RBF\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t'
          f'Update (BatchSize|Accuracy|Time): {batch_size}|{np.around(accuracy_2 * 100.0, 3)}%|{np.around(u_elapsed, 3)}s')


if __name__ == '__main__':
    rbf()
