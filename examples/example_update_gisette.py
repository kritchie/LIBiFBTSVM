import os
import numpy as np
import pandas as pd
import time

from sklearn.kernel_approximation import RBFSampler

from libifbtsvm import iFBTSVM, Hyperparameters


DATA_DIR = os.getenv('DATA_DIR', './data')


def gisette():

    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.1,
        C1=8,
        C2=2,
        C3=8,
        C4=2,
        max_iter=500,
        phi=0,
        kernel=None,  # RBFSampler(gamma=0.4, n_components=150),
        forget_score=10,
    )

    train_data = pd.read_csv(f'{DATA_DIR}/gisette_train.data', delim_whitespace=True)
    train_label = pd.read_csv(f'{DATA_DIR}/gisette_train.labels', delim_whitespace=True)
    test_data = pd.read_csv(f'{DATA_DIR}/gisette_valid.data', delim_whitespace=True)
    test_label = pd.read_csv(f'{DATA_DIR}/gisette_valid.labels', delim_whitespace=True)

    ifbtsvm = iFBTSVM(parameters=params, n_jobs=1)

    # Training
    num_points = 500
    before = time.monotonic()
    ifbtsvm.fit(X=train_data[:num_points].values,
                y=train_label[:num_points].values.reshape(train_label[:num_points].values.shape[0]))
    after = time.monotonic()
    elapsed = (after - before)
    accuracy_1 = ifbtsvm.score(X=test_data.values, y=test_label.values)

    print(f'Gisette\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t')

    # Update
    batch_size = 500
    before = time.monotonic()
    ifbtsvm.update(X=train_data[num_points:].values,
                   y=train_label[num_points:].values.reshape(train_label[num_points:].values.shape[0]),
                   batch_size=batch_size)
    after = time.monotonic()
    u_elapsed = after - before

    # Prediction
    accuracy_2 = ifbtsvm.score(X=test_data.values, y=test_label.values)
    print(f'Gisette\t'
          f'Update (BatchSize|Accuracy|Time): {batch_size}|{np.around(accuracy_2 * 100.0, 3)}%|{np.around(u_elapsed, 3)}s')


if __name__ == '__main__':
    gisette()
