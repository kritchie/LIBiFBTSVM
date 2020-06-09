"""
This file contains code to run benchmarks against the LIBiFBTSVM.

Those benchmarks and evaluation are based on (de Mello, A. R., Stemmer, M. R., & Koerich, A. L., 2019)
"""
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
        fuzzy=0.1,
        C1=8,
        C2=2,
        C3=8,
        C4=2,
        max_evals=250,
        phi=0.00001,
        kernel=RBFSampler(gamma=0.4, n_components=150),
        repetition=10,
    )

    train_data = pd.read_csv(f'{DATA_DIR}/Border_train_data.csv')
    train_label = pd.read_csv(f'{DATA_DIR}/Border_train_label.csv')
    test_data = pd.read_csv(f'{DATA_DIR}/Border_test_data.csv')
    test_label = pd.read_csv(f'{DATA_DIR}/Border_test_label.csv')

    ifbtsvm = iFBTSVM(parameters=params, n_jobs=4)

    # Training
    num_points = 100
    before = time.monotonic()
    ifbtsvm.fit(X=train_data[:num_points].values,
                y=train_label[:num_points].values.reshape(train_label[:num_points].values.shape[0]))
    after = time.monotonic()
    elapsed = (after - before)
    accuracy_1 = ifbtsvm.score(X=test_data.values, y=test_label.values)

    # Update
    batch_size = int(len(train_data.values) / 100 * 5 + 0.5)  # 5% of original dataset
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


def coil():

    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.1,
        C1=4,
        C2=4,
        C3=4,
        C4=4,
        max_evals=250,
        phi=0.00001,
        kernel=RBFSampler(gamma=20, n_components=400),
        repetition=10,
    )

    train_data = pd.read_csv(f'{DATA_DIR}/Coil_train_data.csv')
    train_label = pd.read_csv(f'{DATA_DIR}/Coil_train_label.csv')
    test_data = pd.read_csv(f'{DATA_DIR}/Coil_test_data.csv')
    test_label = pd.read_csv(f'{DATA_DIR}/Coil_test_label.csv')

    ifbtsvm = iFBTSVM(parameters=params, n_jobs=4)

    # Training
    num_points = 500
    before = time.monotonic()
    ifbtsvm.fit(X=train_data[:num_points].values,
                y=train_label[:num_points].values.reshape(train_label[:num_points].values.shape[0]))
    after = time.monotonic()
    elapsed = (after - before)
    accuracy_1 = ifbtsvm.score(X=test_data.values, y=test_label.values)
    print(f'Done training ({np.around(elapsed, 3)}s), updating...')

    # Update
    batch_size = int(len(train_data.values) / 100 * 5 + 0.5)  # 5% of original dataset
    before = time.monotonic()
    ifbtsvm.update(X=train_data[num_points:].values,
                   y=train_label[num_points:].values.reshape(train_label[num_points:].values.shape[0]),
                   batch_size=batch_size)
    after = time.monotonic()
    u_elapsed = after - before

    # Prediction
    accuracy_2 = ifbtsvm.score(X=test_data.values, y=test_label.values)
    print(f'Coil\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t'
          f'Update (BatchSize|Accuracy|Time): {batch_size}|{np.around(accuracy_2 * 100.0, 3)}%|{np.around(u_elapsed, 3)}s')


def mnist():
    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.1,
        C1=10,
        C2=10,
        C3=10,
        C4=10,
        max_evals=250,
        phi=0.00001,
        kernel=RBFSampler(gamma=0.0002, n_components=2400),
        repetition=10,
    )

    train_data = pd.read_csv(f'{DATA_DIR}/MNIST_train_data.csv')
    train_label = pd.read_csv(f'{DATA_DIR}/MNIST_train_label.csv')
    test_data = pd.read_csv(f'{DATA_DIR}/MNIST_test_data.csv')
    test_label = pd.read_csv(f'{DATA_DIR}/MNIST_test_label.csv')

    ifbtsvm = iFBTSVM(parameters=params, n_jobs=4)

    # Training
    num_points = 30000
    before = time.monotonic()
    ifbtsvm.fit(X=train_data[:num_points].values,
                y=train_label[:num_points].values.reshape(train_label[:num_points].values.shape[0]))
    after = time.monotonic()
    elapsed = (after - before)
    accuracy_1 = ifbtsvm.score(X=test_data.values, y=test_label.values)

    # Update
    batch_size = int(len(train_data.values) / 100 * 10 + 0.5)  # 10% of original dataset
    before = time.monotonic()
    ifbtsvm.update(X=train_data[num_points:].values,
                   y=train_label[num_points:].values.reshape(train_label[num_points:].values.shape[0]),
                   batch_size=batch_size)
    after = time.monotonic()
    u_elapsed = after - before

    # Prediction
    accuracy_2 = ifbtsvm.score(X=test_data.values, y=test_label.values)
    print(f'MNIST\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t'
          f'Update (BatchSize|Accuracy|Time): {batch_size}|{np.around(accuracy_2 * 100.0, 3)}%|{np.around(u_elapsed, 3)}s')


def outdoor():

    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.1,
        C1=10,
        C2=1,
        C3=10,
        C4=1,
        max_evals=250,
        phi=0.00001,
        kernel=None,  # RBFSampler(gamma=0.001, n_components=500),
        repetition=10,
    )

    train_data = pd.read_csv(f'{DATA_DIR}/Outdoor_train_data.csv')
    train_label = pd.read_csv(f'{DATA_DIR}/Outdoor_train_label.csv')
    test_data = pd.read_csv(f'{DATA_DIR}/Outdoor_test_data.csv')
    test_label = pd.read_csv(f'{DATA_DIR}/Outdoor_test_label.csv')

    ifbtsvm = iFBTSVM(parameters=params, n_jobs=4)

    # Training
    num_points = 300
    before = time.monotonic()
    ifbtsvm.fit(X=train_data[:num_points].values,
                y=train_label[:num_points].values.reshape(train_label[:num_points].values.shape[0]))
    after = time.monotonic()
    elapsed = (after - before)
    accuracy_1 = ifbtsvm.score(X=test_data.values, y=test_label.values)

    # Update
    batch_size = int(len(train_data.values) / 100 * 5 + 0.5)  # 5% of original dataset
    before = time.monotonic()
    ifbtsvm.update(X=train_data[num_points:].values,
                   y=train_label[num_points:].values.reshape(train_label[num_points:].values.shape[0]),
                   batch_size=batch_size)
    after = time.monotonic()
    u_elapsed = after - before

    # Prediction
    accuracy_2 = ifbtsvm.score(X=test_data.values, y=test_label.values)
    print(f'Outdoor\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t'
          f'Update (BatchSize|Accuracy|Time): {batch_size}|{np.around(accuracy_2 * 100.0, 3)}%|{np.around(u_elapsed, 3)}s')


def overlap():

    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.1,
        C1=8,
        C2=2,
        C3=8,
        C4=2,
        max_evals=250,
        phi=0.00001,
        kernel=RBFSampler(gamma=0.4, n_components=150),
        repetition=10,
    )

    train_data = pd.read_csv(f'{DATA_DIR}/Overlap_train_data.csv')
    train_label = pd.read_csv(f'{DATA_DIR}/Overlap_train_label.csv')
    test_data = pd.read_csv(f'{DATA_DIR}/Overlap_test_data.csv')
    test_label = pd.read_csv(f'{DATA_DIR}/Overlap_test_label.csv')

    ifbtsvm = iFBTSVM(parameters=params, n_jobs=4)

    # Training
    num_points = 100
    before = time.monotonic()
    ifbtsvm.fit(X=train_data[:num_points].values,
                y=train_label[:num_points].values.reshape(train_label[:num_points].values.shape[0]))
    after = time.monotonic()
    elapsed = (after - before)
    accuracy_1 = ifbtsvm.score(X=test_data.values, y=test_label.values)

    # Update
    batch_size = int(len(train_data.values) / 100 * 5 + 0.5)  # 5% of original dataset
    before = time.monotonic()
    ifbtsvm.update(X=train_data[num_points:].values,
                   y=train_label[num_points:].values.reshape(train_label[num_points:].values.shape[0]),
                   batch_size=batch_size)
    after = time.monotonic()
    u_elapsed = after - before

    # Prediction
    accuracy_2 = ifbtsvm.score(X=test_data.values, y=test_label.values)
    print(f'Overlap\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t'
          f'Update (BatchSize|Accuracy|Time): {batch_size}|{np.around(accuracy_2 * 100.0, 3)}%|{np.around(u_elapsed, 3)}s')


def hyper():

    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.1,
        C1=5,
        C2=4,
        C3=5,
        C4=4,
        max_evals=250,
        phi=0.00001,
        kernel=None,  # RBFSampler(gamma=0.4, n_components=150),
        repetition=10,
    )

    _data = pd.read_csv(f'{DATA_DIR}/HYPER/10K/HYPER10K.csv')
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
    print(f'Hyper\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t'
          f'Update (BatchSize|Accuracy|Time): {batch_size}|{np.around(accuracy_2 * 100.0, 3)}%|{np.around(u_elapsed, 3)}s')


def led():
    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.1,
        C1=8,
        C2=2,
        C3=8,
        C4=2,
        max_evals=250,
        phi=0.00001,
        kernel=None,  # RBFSampler(gamma=0.4, n_components=150),
        repetition=10,
    )

    _data = pd.read_csv(f'{DATA_DIR}/LED/10K/LED10K.csv')
    train_data = _data.values[:10000, 0:24]
    train_label = _data.values[:10000, 24:]
    test_data = _data.values[10000:, 0:24]
    test_label = _data.values[10000:, 24:]

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
    print(f'LED\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t'
          f'Update (BatchSize|Accuracy|Time): {batch_size}|{np.around(accuracy_2 * 100.0, 3)}%|{np.around(u_elapsed, 3)}s')


def rbf():
    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.1,
        C1=8,
        C2=2,
        C3=8,
        C4=2,
        max_evals=250,
        phi=0.00001,
        kernel=RBFSampler(gamma=0.45, n_components=300),
        repetition=10,
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


def rtg():
    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.1,
        C1=2.5,
        C2=2,
        C3=2.5,
        C4=2,
        max_evals=250,
        phi=0.00001,
        kernel=RBFSampler(gamma=0.6, n_components=1400),
        repetition=10,
    )

    _data = pd.read_csv(f'{DATA_DIR}/RTG/10K/RTG10K.csv')
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
    print(f'RTG\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t'
          f'Update (BatchSize|Accuracy|Time): {batch_size}|{np.around(accuracy_2 * 100.0, 3)}%|{np.around(u_elapsed, 3)}s')


def sea():
    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.1,
        C1=10,
        C2=1,
        C3=10,
        C4=1,
        max_evals=250,
        phi=0.00001,
        kernel=None,  # RBFSampler(gamma=0.6, n_components=1400),
        repetition=10,
    )

    _data = pd.read_csv(f'{DATA_DIR}/SEA/10K/SEA10K.csv')
    train_data = _data.values[:10000, 0:3]
    train_label = _data.values[:10000, 3:]
    test_data = _data.values[10000:, 0:3]
    test_label = _data.values[10000:, 3:]

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
    print(f'SEA\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t'
          f'Update (BatchSize|Accuracy|Time): {batch_size}|{np.around(accuracy_2 * 100.0, 3)}%|{np.around(u_elapsed, 3)}s')


def letter():
    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.1,
        C1=8,
        C2=2,
        C3=8,
        C4=2,
        max_evals=250,
        phi=0.00001,
        kernel=RBFSampler(gamma=0.01, n_components=350),
        repetition=10,
    )

    _data = pd.read_csv(f'{DATA_DIR}/letter-recognition.data')
    train_data = _data.values[:16000, 1:]
    train_label = _data.values[:16000, 0]
    test_data = _data.values[16000:, 1:]
    test_label = _data.values[16000:, 0]

    for i, lbl in enumerate(train_label):
        train_label[i] = ord(lbl) - 65  # '65' -> 'A'

    for i, lbl in enumerate(test_label):
        test_label[i] = ord(lbl) - 65  # '65' -> 'A'

    test_label = test_label.reshape(test_label.shape[0], 1).astype(np.int)

    ifbtsvm = iFBTSVM(parameters=params, n_jobs=4)

    # Training
    num_points = 1000
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
    print(f'Letter\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t'
          f'Update (BatchSize|Accuracy|Time): {batch_size}|{np.around(accuracy_2 * 100.0, 3)}%|{np.around(u_elapsed, 3)}s')


def dna():
    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.1,
        C1=4,
        C2=4,
        C3=4,
        C4=4,
        max_evals=250,
        phi=0.00001,
        kernel=RBFSampler(gamma=0.003, n_components=500),
        repetition=10,
    )

    train_data = np.zeros((1400, 180))
    train_label = np.zeros((1400,))
    with open(f'{DATA_DIR}/dna.scale.tr', 'r') as f_in:
        for i, line in enumerate(f_in):
            split = line.split(' ')
            train_label[i] = split[0]
            for s in split[1:]:
                if s == '\n':
                    continue
                feats = s.split(':')
                train_data[i, int(feats[0])-1] = int(feats[1])

    test_data = np.zeros((1186, 180))
    test_label = np.zeros((1186,))
    with open(f'{DATA_DIR}/dna.scale.t', 'r') as f_in:
        for i, line in enumerate(f_in):
            split = line.split(' ')
            test_label[i] = split[0]
            for s in split[1:]:
                if s == '\n':
                    continue
                feats = s.split(':')
                test_data[i, int(feats[0])-1] = int(feats[1])

    ifbtsvm = iFBTSVM(parameters=params, n_jobs=4)

    # Training
    num_points = 50
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
    print(f'DNA\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t'
          f'Update (BatchSize|Accuracy|Time): {batch_size}|{np.around(accuracy_2 * 100.0, 3)}%|{np.around(u_elapsed, 3)}s')


def usps():
    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.1,
        C1=8,
        C2=2,
        C3=8,
        C4=2,
        max_evals=250,
        phi=0.00001,
        kernel=RBFSampler(gamma=0.007, n_components=1000),
        repetition=10,
    )

    train_data = np.zeros((7291, 256))
    train_label = np.zeros((7291,))
    with open(f'{DATA_DIR}/dna.scale.tr', 'r') as f_in:
        for i, line in enumerate(f_in):
            split = line.split(' ')
            train_label[i] = split[0]
            for s in split[1:]:
                if s == '\n':
                    continue
                feats = s.split(':')
                train_data[i, int(feats[0]) - 1] = int(feats[1])

    test_data = np.zeros((2007, 256))
    test_label = np.zeros((2007,))
    with open(f'{DATA_DIR}/dna.scale.t', 'r') as f_in:
        for i, line in enumerate(f_in):
            split = line.split(' ')
            test_label[i] = split[0]
            for s in split[1:]:
                if s == '\n':
                    continue
                feats = s.split(':')
                test_data[i, int(feats[0]) - 1] = int(feats[1])

    ifbtsvm = iFBTSVM(parameters=params, n_jobs=4)

    # Training
    num_points = 1000
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
    print(f'USPS\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t'
          f'Update (BatchSize|Accuracy|Time): {batch_size}|{np.around(accuracy_2 * 100.0, 3)}%|{np.around(u_elapsed, 3)}s')


def isolet():
    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.1,
        C1=10,
        C2=10,
        C3=10,
        C4=10,
        max_evals=250,
        phi=0.00001,
        kernel=RBFSampler(gamma=0.002, n_components=1000),
        repetition=10,
    )

    _data = pd.read_csv(f'{DATA_DIR}/isolet1+2+3+4.data')
    train_data = _data.values[:, :617]
    train_label = _data.values[:, 617]

    _data = pd.read_csv(f'{DATA_DIR}/isolet5.data')
    test_data = _data.values[:, :617]
    test_label = _data.values[:, 617]

    ifbtsvm = iFBTSVM(parameters=params, n_jobs=4)

    # Training
    num_points = 500
    before = time.monotonic()
    ifbtsvm.fit(X=train_data[:num_points],
                y=train_label[:num_points])
    after = time.monotonic()
    elapsed = (after - before)
    accuracy_1 = ifbtsvm.score(X=test_data, y=test_label)

    # Update
    batch_size = int(len(train_data) / 100 * 5 + 0.5)  # 5% of original dataset
    before = time.monotonic()
    ifbtsvm.update(X=train_data[num_points:],
                   y=train_label[num_points:],
                   batch_size=batch_size)
    after = time.monotonic()
    u_elapsed = after - before

    # Prediction
    accuracy_2 = ifbtsvm.score(X=test_data, y=test_label)
    print(f'Isolet\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t'
          f'Update (BatchSize|Accuracy|Time): {batch_size}|{np.around(accuracy_2 * 100.0, 3)}%|{np.around(u_elapsed, 3)}s')


def gisette():

    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.1,
        C1=8,
        C2=2,
        C3=8,
        C4=2,
        max_evals=250,
        phi=0.00001,
        kernel=None,
        repetition=10,
    )

    train_data = pd.read_csv(f'{DATA_DIR}/gisette_train.data', delim_whitespace=True)
    train_label = pd.read_csv(f'{DATA_DIR}/gisette_train.labels', delim_whitespace=True)
    test_data = pd.read_csv(f'{DATA_DIR}/gisette_valid.data', delim_whitespace=True)
    test_label = pd.read_csv(f'{DATA_DIR}/gisette_valid.labels', delim_whitespace=True)

    ifbtsvm = iFBTSVM(parameters=params, n_jobs=4)

    # Training
    num_points = 500
    before = time.monotonic()
    ifbtsvm.fit(X=train_data[:num_points].values,
                y=train_label[:num_points].values.reshape(train_label[:num_points].values.shape[0]))
    after = time.monotonic()
    elapsed = (after - before)
    accuracy_1 = ifbtsvm.score(X=test_data.values, y=test_label.values)

    # Update
    batch_size = int(len(train_data.values) / 100 * 5 + 0.5)  # 5% of original dataset
    before = time.monotonic()
    ifbtsvm.update(X=train_data[num_points:].values,
                   y=train_label[num_points:].values.reshape(train_label[num_points:].values.shape[0]),
                   batch_size=batch_size)
    after = time.monotonic()
    u_elapsed = after - before

    # Prediction
    accuracy_2 = ifbtsvm.score(X=test_data.values, y=test_label.values)
    print(f'Gisette\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t'
          f'Update (BatchSize|Accuracy|Time): {batch_size}|{np.around(accuracy_2 * 100.0, 3)}%|{np.around(u_elapsed, 3)}s')


if __name__ == '__main__':
    border()
    # coil() - Doesn't converge on training
    overlap()
    outdoor()
    # mnist()
    hyper()
    led()

    rbf()
    rtg()
    sea()
    letter()
    dna()
    # usps()  # Error
    isolet()
    gisette()
