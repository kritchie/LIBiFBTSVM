import os
import numpy as np
import pandas as pd
import time

from sklearn.kernel_approximation import RBFSampler

from libifbtsvm import iFBTSVM, Hyperparameters


DATA_DIR = os.getenv('DATA_DIR', './data')


def border():

    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.01,
        C1=8,
        C2=2,
        C3=8,
        C4=2,
        max_evals=500,
        phi=0.00001,
        kernel=RBFSampler(gamma=0.4, n_components=150),
        repetition=5,
    )

    train_data = pd.read_csv(f'{DATA_DIR}/Border_train_data.csv')
    train_label = pd.read_csv(f'{DATA_DIR}/Border_train_label.csv')
    test_data = pd.read_csv(f'{DATA_DIR}/Border_test_data.csv')
    test_label = pd.read_csv(f'{DATA_DIR}/Border_test_label.csv')

    ifbtsvm = iFBTSVM(parameters=params, n_jobs=4)

    # Training
    num_points = 60
    before = time.monotonic()
    ifbtsvm.fit(X=train_data[:num_points].values,
                y=train_label[:num_points].values.reshape(train_label[:num_points].values.shape[0]))
    after = time.monotonic()
    elapsed = (after - before)
    accuracy_1 = ifbtsvm.score(X=test_data.values, y=test_label.values)

    # Update
    batch_size = 100
    before = time.monotonic()
    ifbtsvm.update(X=train_data[num_points:].values,
                   y=train_label[num_points:].values.reshape(train_label[num_points:].values.shape[0]),
                   batch_size=batch_size)
    after = time.monotonic()
    u_elapsed = after - before

    # Prediction
    accuracy_2 = ifbtsvm.score(X=test_data.values, y=test_label.values)
    print(f'Border\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t'
          f'Update (BatchSize|Accuracy|Time): {batch_size}|{np.around(accuracy_2 * 100.0, 3)}%|{np.around(u_elapsed, 3)}s')


if __name__ == '__main__':
    border()
