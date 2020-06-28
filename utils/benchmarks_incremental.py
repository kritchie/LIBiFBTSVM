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


def border(forget_score):

    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.1,
        C1=8,
        C2=2,
        C3=8,
        C4=2,
        max_iter=50,
        phi=0.00001,
        kernel=RBFSampler(gamma=0.1, n_components=1000),
        forget_score=forget_score,
    )

    train_data = pd.read_csv(f'{DATA_DIR}/Border_train_data.csv')
    train_label = pd.read_csv(f'{DATA_DIR}/Border_train_label.csv')
    test_data = pd.read_csv(f'{DATA_DIR}/Border_test_data.csv')
    test_label = pd.read_csv(f'{DATA_DIR}/Border_test_label.csv')

    ifbtsvm = iFBTSVM(parameters=params, n_jobs=4)

    # Training
    num_points = int(len(train_data.values) / 100 * 5 + 0.5)  # 5% of original dataset
    before = time.monotonic()
    ifbtsvm.fit(X=train_data[:num_points].values,
                y=train_label[:num_points].values.reshape(train_label[:num_points].values.shape[0]))
    after = time.monotonic()
    elapsed = (after - before)
    accuracy_1 = ifbtsvm.score(X=test_data.values, y=test_label.values)

    accuracy_1 = ifbtsvm.score(X=test_data.values, y=test_label.values)

    print(f'Border\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t')

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
          f'Update (BatchSize|Accuracy|Time): {batch_size}|{np.around(accuracy_2 * 100.0, 3)}%|{np.around(u_elapsed, 3)}s')
    return accuracy_2, (elapsed + u_elapsed)


def coil(forget_score):

    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.1,
        C1=10,
        C2=10,
        C3=10,
        C4=10,
        max_iter=50,
        phi=0,
        kernel=RBFSampler(gamma=50, n_components=500),
        forget_score=forget_score,
    )

    train_data = pd.read_csv(f'{DATA_DIR}/Coil_train_data.csv')
    train_label = pd.read_csv(f'{DATA_DIR}/Coil_train_label.csv')
    test_data = pd.read_csv(f'{DATA_DIR}/Coil_test_data.csv')
    test_label = pd.read_csv(f'{DATA_DIR}/Coil_test_label.csv')

    indices = np.arange(train_data.values.shape[0])
    np.random.shuffle(indices)

    train_data = train_data.values[indices]
    train_label = train_label.values[indices]

    ifbtsvm = iFBTSVM(parameters=params, n_jobs=4)

    # Training
    num_points = 1000
    before = time.monotonic()
    ifbtsvm.fit(X=train_data[:num_points],
                y=train_label[:num_points].reshape(train_label[:num_points].shape[0]))
    after = time.monotonic()
    elapsed = (after - before)
    accuracy_1 = ifbtsvm.score(X=test_data.values, y=test_label.values)

    print(f'Coil\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t')

    # Update
    batch_size = 90  # 5% of original dataset
    before = time.monotonic()
    ifbtsvm.update(X=train_data[num_points:],
                   y=train_label[num_points:].reshape(train_label[num_points:].shape[0]),
                   batch_size=batch_size)
    after = time.monotonic()
    u_elapsed = after - before

    # Prediction
    accuracy_2 = ifbtsvm.score(X=test_data.values, y=test_label.values)
    print(f'Coil\t'
          f'Update (BatchSize|Accuracy|Time): {batch_size}|{np.around(accuracy_2 * 100.0, 3)}%|{np.around(u_elapsed, 3)}s')
    return accuracy_2, (elapsed + u_elapsed)


def mnist(forget_score):
    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.1,
        C1=10,
        C2=10,
        C3=10,
        C4=10,
        max_iter=50,
        phi=0.00001,
        kernel=RBFSampler(gamma=0.01, n_components=1000),
        forget_score=forget_score,
    )

    train_data = pd.read_csv(f'{DATA_DIR}/MNIST_train_data.csv')
    train_label = pd.read_csv(f'{DATA_DIR}/MNIST_train_label.csv')
    test_data = pd.read_csv(f'{DATA_DIR}/MNIST_test_data.csv')
    test_label = pd.read_csv(f'{DATA_DIR}/MNIST_test_label.csv')

    ifbtsvm = iFBTSVM(parameters=params, n_jobs=4)

    # Training
    num_points = 10000
    before = time.monotonic()
    ifbtsvm.fit(X=train_data[:num_points].values,
                y=train_label[:num_points].values.reshape(train_label[:num_points].values.shape[0]))
    after = time.monotonic()
    elapsed = (after - before)
    accuracy_1 = ifbtsvm.score(X=test_data.values, y=test_label.values)

    print(f'MNIST\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t')

    # Update
    batch_size = int(len(train_data.values) / 100 * 5 + 0.5)  # 10% of original dataset
    before = time.monotonic()
    ifbtsvm.update(X=train_data[num_points:].values,
                   y=train_label[num_points:].values.reshape(train_label[num_points:].values.shape[0]),
                   batch_size=batch_size)
    after = time.monotonic()
    u_elapsed = after - before

    # Prediction
    accuracy_2 = ifbtsvm.score(X=test_data.values, y=test_label.values)
    print(f'MNIST\t'
          f'Update (BatchSize|Accuracy|Time): {batch_size}|{np.around(accuracy_2 * 100.0, 3)}%|{np.around(u_elapsed, 3)}s')
    return accuracy_2, (elapsed + u_elapsed)


def outdoor(forget_score):

    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.1,
        C1=10,
        C2=1,
        C3=10,
        C4=1,
        max_iter=50,
        phi=0.00001,
        kernel=RBFSampler(gamma=20, n_components=400),
        forget_score=forget_score,
    )

    train_data = pd.read_csv(f'{DATA_DIR}/Outdoor_train_data.csv')
    train_label = pd.read_csv(f'{DATA_DIR}/Outdoor_train_label.csv')
    test_data = pd.read_csv(f'{DATA_DIR}/Outdoor_test_data.csv')
    test_label = pd.read_csv(f'{DATA_DIR}/Outdoor_test_label.csv')

    # indices = np.arange(train_data.values.shape[0])
    # np.random.shuffle(indices)
    #
    # train_data = train_data.values[indices]
    # train_label = train_label.values[indices]

    ifbtsvm = iFBTSVM(parameters=params, n_jobs=4)

    # Training
    num_points = 1000
    before = time.monotonic()
    ifbtsvm.fit(X=train_data[:num_points].values,
                y=train_label[:num_points].values.reshape(train_label[:num_points].values.shape[0]))
    after = time.monotonic()
    elapsed = (after - before)
    accuracy_1 = ifbtsvm.score(X=test_data.values, y=test_label.values)

    print(f'Outdoor\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t')

    # Update
    batch_size = 500  # 5% of original dataset
    before = time.monotonic()
    ifbtsvm.update(X=train_data[num_points:].values,
                   y=train_label[num_points:].values.reshape(train_label[num_points:].values.shape[0]),
                   batch_size=batch_size)
    after = time.monotonic()
    u_elapsed = after - before

    # Prediction
    accuracy_2 = ifbtsvm.score(X=test_data.values, y=test_label.values)
    print(f'Outdoor\t'
          f'Update (BatchSize|Accuracy|Time): {batch_size}|{np.around(accuracy_2 * 100.0, 3)}%|{np.around(u_elapsed, 3)}s')
    return accuracy_2, (elapsed + u_elapsed)


def overlap(forget_score):

    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.1,
        C1=8,
        C2=2,
        C3=8,
        C4=2,
        max_iter=50,
        phi=0.00001,
        kernel=RBFSampler(gamma=1, n_components=3000),
        forget_score=forget_score,
    )

    train_data = pd.read_csv(f'{DATA_DIR}/Overlap_train_data.csv')
    train_label = pd.read_csv(f'{DATA_DIR}/Overlap_train_label.csv')
    test_data = pd.read_csv(f'{DATA_DIR}/Overlap_test_data.csv')
    test_label = pd.read_csv(f'{DATA_DIR}/Overlap_test_label.csv')

    ifbtsvm = iFBTSVM(parameters=params, n_jobs=4)

    # Training
    num_points = int(len(train_data.values) / 100 * 5 + 0.5)  # 5% of original dataset
    before = time.monotonic()
    ifbtsvm.fit(X=train_data[:num_points].values,
                y=train_label[:num_points].values.reshape(train_label[:num_points].values.shape[0]))
    after = time.monotonic()
    elapsed = (after - before)
    accuracy_1 = ifbtsvm.score(X=test_data.values, y=test_label.values)

    print(f'Overlap\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t')

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
          f'Update (BatchSize|Accuracy|Time): {batch_size}|{np.around(accuracy_2 * 100.0, 3)}%|{np.around(u_elapsed, 3)}s')
    return accuracy_2, (elapsed + u_elapsed)


def hyper(forget_score):

    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.1,
        C1=5,
        C2=4,
        C3=5,
        C4=4,
        max_iter=50,
        phi=0.00001,
        kernel=None,  # RBFSampler(gamma=0.4, n_components=150),
        forget_score=forget_score,
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

    print(f'Hyper\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t')

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
          f'Update (BatchSize|Accuracy|Time): {batch_size}|{np.around(accuracy_2 * 100.0, 3)}%|{np.around(u_elapsed, 3)}s')
    return accuracy_2, (elapsed + u_elapsed)


def hyper100K(forget_score):

    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.1,
        C1=5,
        C2=4,
        C3=5,
        C4=4,
        max_iter=50,
        phi=0.00001,
        kernel=None,  # RBFSampler(gamma=0.4, n_components=150),
        forget_score=forget_score,
    )

    _data = pd.read_csv(f'{DATA_DIR}/HYPER/100K/HYPER100K.csv')
    train_data = _data.values[:100000, 0:10]
    train_label = _data.values[:100000, 10:]
    test_data = _data.values[100000:, 0:10]
    test_label = _data.values[100000:, 10:]

    ifbtsvm = iFBTSVM(parameters=params, n_jobs=4)

    # Training
    num_points = 5000
    before = time.monotonic()
    ifbtsvm.fit(X=train_data[:num_points],
                y=train_label[:num_points].reshape(train_label[:num_points].shape[0]))
    after = time.monotonic()
    elapsed = (after - before)
    accuracy_1 = ifbtsvm.score(X=test_data, y=test_label)

    print(f'Hyper100k\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t')

    # Update
    batch_size = 5000
    before = time.monotonic()
    ifbtsvm.update(X=train_data[num_points:],
                   y=train_label[num_points:].reshape(train_label[num_points:].shape[0]),
                   batch_size=batch_size)
    after = time.monotonic()
    u_elapsed = after - before

    # Prediction
    accuracy_2 = ifbtsvm.score(X=test_data, y=test_label)
    print(f'Hyper100k\t'
          f'Update (BatchSize|Accuracy|Time): {batch_size}|{np.around(accuracy_2 * 100.0, 3)}%|{np.around(u_elapsed, 3)}s')
    return accuracy_2, (elapsed + u_elapsed)


def hyper1M(forget_score):

    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.1,
        C1=5,
        C2=4,
        C3=5,
        C4=4,
        max_iter=50,
        phi=0.00001,
        kernel=None,  # RBFSampler(gamma=0.4, n_components=150),
        forget_score=forget_score,
    )

    _data = pd.read_csv(f'{DATA_DIR}/HYPER/1M/HYPERtrain.csv')
    train_data = _data.values[:1000000, 0:10]
    train_label = _data.values[:1000000, 10:]
    test_data = _data.values[1000000:, 0:10]
    test_label = _data.values[1000000:, 10:]

    ifbtsvm = iFBTSVM(parameters=params, n_jobs=4)

    # Training
    num_points = 50000
    before = time.monotonic()
    ifbtsvm.fit(X=train_data[:num_points],
                y=train_label[:num_points].reshape(train_label[:num_points].shape[0]))
    after = time.monotonic()
    elapsed = (after - before)
    accuracy_1 = ifbtsvm.score(X=test_data, y=test_label)

    print(f'Hyper1M\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t')

    # Update
    batch_size = 100000
    before = time.monotonic()
    ifbtsvm.update(X=train_data[num_points:],
                   y=train_label[num_points:].reshape(train_label[num_points:].shape[0]),
                   batch_size=batch_size)
    after = time.monotonic()
    u_elapsed = after - before

    # Prediction
    accuracy_2 = ifbtsvm.score(X=test_data, y=test_label)
    print(f'Hyper1M\t'
          f'Update (BatchSize|Accuracy|Time): {batch_size}|{np.around(accuracy_2 * 100.0, 3)}%|{np.around(u_elapsed, 3)}s')
    return accuracy_2, (elapsed + u_elapsed)


def led(forget_score):
    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.1,
        C1=8,
        C2=2,
        C3=8,
        C4=2,
        max_iter=50,
        phi=0.00001,
        kernel=None,  # RBFSampler(gamma=0.4, n_components=150),
        forget_score=forget_score,
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

    print(f'LED\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t')

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
          f'Update (BatchSize|Accuracy|Time): {batch_size}|{np.around(accuracy_2 * 100.0, 3)}%|{np.around(u_elapsed, 3)}s')
    return accuracy_2, (elapsed + u_elapsed)


def led100K(forget_score):
    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.1,
        C1=8,
        C2=2,
        C3=8,
        C4=2,
        max_iter=50,
        phi=0.00001,
        kernel=None,  # RBFSampler(gamma=0.4, n_components=150),
        forget_score=forget_score,
    )

    _data = pd.read_csv(f'{DATA_DIR}/LED/100K/LED100K.csv')
    train_data = _data.values[:100000, 0:24]
    train_label = _data.values[:100000, 24:]
    test_data = _data.values[100000:, 0:24]
    test_label = _data.values[100000:, 24:]

    ifbtsvm = iFBTSVM(parameters=params, n_jobs=4)

    # Training
    num_points = 5000
    before = time.monotonic()
    ifbtsvm.fit(X=train_data[:num_points],
                y=train_label[:num_points].reshape(train_label[:num_points].shape[0]))
    after = time.monotonic()
    elapsed = (after - before)
    accuracy_1 = ifbtsvm.score(X=test_data, y=test_label)

    print(f'LED100K\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t')

    # Update
    batch_size = 5000
    before = time.monotonic()
    ifbtsvm.update(X=train_data[num_points:],
                   y=train_label[num_points:].reshape(train_label[num_points:].shape[0]),
                   batch_size=batch_size)
    after = time.monotonic()
    u_elapsed = after - before

    # Prediction
    accuracy_2 = ifbtsvm.score(X=test_data, y=test_label)
    print(f'LED100K\t'
          f'Update (BatchSize|Accuracy|Time): {batch_size}|{np.around(accuracy_2 * 100.0, 3)}%|{np.around(u_elapsed, 3)}s')
    return accuracy_2, (elapsed + u_elapsed)


def led1M(forget_score):
    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.1,
        C1=8,
        C2=2,
        C3=8,
        C4=2,
        max_iter=50,
        phi=0.00001,
        kernel=None,  # RBFSampler(gamma=0.4, n_components=150),
        forget_score=forget_score,
    )

    _data = pd.read_csv(f'{DATA_DIR}/LED/1M/LEDtrain.csv')
    train_data = _data.values
    _data = pd.read_csv(f'{DATA_DIR}/LED/1M/LEDlabel.csv')
    train_label = _data.values

    _data = pd.read_csv(f'{DATA_DIR}/LED/1M/LEDtest.csv')
    test_data = _data.values[:, 0:24]
    test_label = _data.values[:, -1]

    ifbtsvm = iFBTSVM(parameters=params, n_jobs=4)

    # Training
    num_points = 50000
    before = time.monotonic()
    ifbtsvm.fit(X=train_data[:num_points],
                y=train_label[:num_points].reshape(train_label[:num_points].shape[0]))
    after = time.monotonic()
    elapsed = (after - before)
    accuracy_1 = ifbtsvm.score(X=test_data, y=test_label)

    print(f'LED1M\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t')

    # Update
    batch_size = 10000  # 5% of original dataset
    before = time.monotonic()
    ifbtsvm.update(X=train_data[num_points:],
                   y=train_label[num_points:].reshape(train_label[num_points:].shape[0]),
                   batch_size=batch_size)
    after = time.monotonic()
    u_elapsed = after - before

    # Prediction
    accuracy_2 = ifbtsvm.score(X=test_data, y=test_label)
    print(f'LED1M\t'
          f'Update (BatchSize|Accuracy|Time): {batch_size}|{np.around(accuracy_2 * 100.0, 3)}%|{np.around(u_elapsed, 3)}s')
    return accuracy_2, (elapsed + u_elapsed)


def rbf(forget_score):
    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.1,
        C1=8,
        C2=2,
        C3=8,
        C4=2,
        max_iter=50,
        phi=0.00001,
        kernel=RBFSampler(gamma=1, n_components=1500),
        forget_score=forget_score,
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

    print(f'RBF\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t')

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
          f'Update (BatchSize|Accuracy|Time): {batch_size}|{np.around(accuracy_2 * 100.0, 3)}%|{np.around(u_elapsed, 3)}s')
    return accuracy_2, (elapsed + u_elapsed)


def rbf100K(forget_score):
    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.1,
        C1=8,
        C2=2,
        C3=8,
        C4=2,
        max_iter=50,
        phi=0.00001,
        kernel=RBFSampler(gamma=1, n_components=1000),
        forget_score=forget_score,
    )

    _data = pd.read_csv(f'{DATA_DIR}/RBF/100K/RBF100K.csv')
    train_data = _data.values[:100000, 0:10]
    train_label = _data.values[:100000, 10:]
    test_data = _data.values[100000:, 0:10]
    test_label = _data.values[100000:, 10:]

    ifbtsvm = iFBTSVM(parameters=params, n_jobs=4)

    # Training
    num_points = 5000
    before = time.monotonic()
    ifbtsvm.fit(X=train_data[:num_points],
                y=train_label[:num_points].reshape(train_label[:num_points].shape[0]))
    after = time.monotonic()
    elapsed = (after - before)
    accuracy_1 = ifbtsvm.score(X=test_data, y=test_label)

    print(f'RBF100K\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t')

    # Update
    batch_size = 5000
    before = time.monotonic()
    ifbtsvm.update(X=train_data[num_points:],
                   y=train_label[num_points:].reshape(train_label[num_points:].shape[0]),
                   batch_size=batch_size)
    after = time.monotonic()
    u_elapsed = after - before

    # Prediction
    accuracy_2 = ifbtsvm.score(X=test_data, y=test_label)
    print(f'RBF100K\t'
          f'Update (BatchSize|Accuracy|Time): {batch_size}|{np.around(accuracy_2 * 100.0, 3)}%|{np.around(u_elapsed, 3)}s')
    return accuracy_2, (elapsed + u_elapsed)


def rbf1M(forget_score):
    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.1,
        C1=8,
        C2=2,
        C3=8,
        C4=2,
        max_iter=50,
        phi=0.00001,
        kernel=RBFSampler(gamma=1, n_components=1500),
        forget_score=forget_score,
    )

    _data = pd.read_csv(f'{DATA_DIR}/RBF/1M/RBFtrain.csv')
    train_data = _data.values[:1000000, 0:10]
    train_label = _data.values[:1000000, 10:]
    test_data = _data.values[1000000:, 0:10]
    test_label = _data.values[1000000:, 10:]

    ifbtsvm = iFBTSVM(parameters=params, n_jobs=4)

    # Training
    num_points = 5000
    before = time.monotonic()
    ifbtsvm.fit(X=train_data[:num_points],
                y=train_label[:num_points].reshape(train_label[:num_points].shape[0]))
    after = time.monotonic()
    elapsed = (after - before)
    accuracy_1 = ifbtsvm.score(X=test_data, y=test_label)

    print(f'RBF1M\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t')

    # Update
    batch_size = 1000
    before = time.monotonic()
    ifbtsvm.update(X=train_data[num_points:],
                   y=train_label[num_points:].reshape(train_label[num_points:].shape[0]),
                   batch_size=batch_size)
    after = time.monotonic()
    u_elapsed = after - before

    # Prediction
    accuracy_2 = ifbtsvm.score(X=test_data, y=test_label)
    print(f'RBF1M\t'
          f'Update (BatchSize|Accuracy|Time): {batch_size}|{np.around(accuracy_2 * 100.0, 3)}%|{np.around(u_elapsed, 3)}s')
    return accuracy_2, (elapsed + u_elapsed)


def rtg(forget_score):
    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.1,
        C1=2.5,
        C2=2,
        C3=2.5,
        C4=2,
        max_iter=50,
        phi=0.00001,
        kernel=RBFSampler(gamma=1, n_components=1000),
        forget_score=forget_score,
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

    print(f'RTG\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t')

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
          f'Update (BatchSize|Accuracy|Time): {batch_size}|{np.around(accuracy_2 * 100.0, 3)}%|{np.around(u_elapsed, 3)}s')
    return accuracy_2, (elapsed + u_elapsed)


def rtg100K(forget_score):
    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.1,
        C1=2.5,
        C2=2,
        C3=2.5,
        C4=2,
        max_iter=50,
        phi=0.00001,
        kernel=RBFSampler(gamma=1, n_components=1000),
        forget_score=forget_score,
    )

    _data = pd.read_csv(f'{DATA_DIR}/RTG/100K/RTG100K.csv')
    train_data = _data.values[:100000, 0:10]
    train_label = _data.values[:100000, 10:]
    test_data = _data.values[100000:, 0:10]
    test_label = _data.values[100000:, 10:]

    ifbtsvm = iFBTSVM(parameters=params, n_jobs=4)

    # Training
    num_points = 5000
    before = time.monotonic()
    ifbtsvm.fit(X=train_data[:num_points],
                y=train_label[:num_points].reshape(train_label[:num_points].shape[0]))
    after = time.monotonic()
    elapsed = (after - before)
    accuracy_1 = ifbtsvm.score(X=test_data, y=test_label)

    print(f'RTG100K\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t')

    # Update
    batch_size = 5000
    before = time.monotonic()
    ifbtsvm.update(X=train_data[num_points:],
                   y=train_label[num_points:].reshape(train_label[num_points:].shape[0]),
                   batch_size=batch_size)
    after = time.monotonic()
    u_elapsed = after - before

    # Prediction
    accuracy_2 = ifbtsvm.score(X=test_data, y=test_label)
    print(f'RTG100K\t'
          f'Update (BatchSize|Accuracy|Time): {batch_size}|{np.around(accuracy_2 * 100.0, 3)}%|{np.around(u_elapsed, 3)}s')
    return accuracy_2, (elapsed + u_elapsed)


def rtg1M(forget_score):
    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.1,
        C1=2.5,
        C2=2,
        C3=2.5,
        C4=2,
        max_iter=50,
        phi=0.00001,
        kernel=RBFSampler(gamma=1, n_components=1000),
        forget_score=forget_score,
    )

    _data = pd.read_csv(f'{DATA_DIR}/RTG/1M/RTGtrain.csv')
    train_data = _data.values[:1000000, 0:10]
    train_label = _data.values[:1000000, 10:]
    test_data = _data.values[1000000:, 0:10]
    test_label = _data.values[1000000:, 10:]

    ifbtsvm = iFBTSVM(parameters=params, n_jobs=4)

    # Training
    num_points = 50000
    before = time.monotonic()
    ifbtsvm.fit(X=train_data[:num_points],
                y=train_label[:num_points].reshape(train_label[:num_points].shape[0]))
    after = time.monotonic()
    elapsed = (after - before)
    accuracy_1 = ifbtsvm.score(X=test_data, y=test_label)

    print(f'RTG1M\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t')

    # Update
    batch_size = 10000
    before = time.monotonic()
    ifbtsvm.update(X=train_data[num_points:],
                   y=train_label[num_points:].reshape(train_label[num_points:].shape[0]),
                   batch_size=batch_size)
    after = time.monotonic()
    u_elapsed = after - before

    # Prediction
    accuracy_2 = ifbtsvm.score(X=test_data, y=test_label)
    print(f'RTG1M\t'
          f'Update (BatchSize|Accuracy|Time): {batch_size}|{np.around(accuracy_2 * 100.0, 3)}%|{np.around(u_elapsed, 3)}s')
    return accuracy_2, (elapsed + u_elapsed)


def sea(forget_score):
    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.1,
        C1=10,
        C2=1,
        C3=10,
        C4=1,
        max_iter=50,
        phi=0.00001,
        kernel=None,  # RBFSampler(gamma=0.6, n_components=1400),
        forget_score=forget_score,
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

    print(f'SEA\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t')

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
          f'Update (BatchSize|Accuracy|Time): {batch_size}|{np.around(accuracy_2 * 100.0, 3)}%|{np.around(u_elapsed, 3)}s')
    return accuracy_2, (elapsed + u_elapsed)


def sea100K(forget_score):
    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.1,
        C1=10,
        C2=1,
        C3=10,
        C4=1,
        max_iter=50,
        phi=0.00001,
        kernel=None,  # RBFSampler(gamma=0.6, n_components=1400),
        forget_score=forget_score,
    )

    _data = pd.read_csv(f'{DATA_DIR}/SEA/100K/SEA100K.csv')
    train_data = _data.values[:100000, 0:3]
    train_label = _data.values[:100000, 3:]
    test_data = _data.values[100000:, 0:3]
    test_label = _data.values[100000:, 3:]

    ifbtsvm = iFBTSVM(parameters=params, n_jobs=4)

    # Training
    num_points = 5000
    before = time.monotonic()
    ifbtsvm.fit(X=train_data[:num_points],
                y=train_label[:num_points].reshape(train_label[:num_points].shape[0]))
    after = time.monotonic()
    elapsed = (after - before)
    accuracy_1 = ifbtsvm.score(X=test_data, y=test_label)

    print(f'SEA100K\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t')

    # Update
    batch_size = 5000
    before = time.monotonic()
    ifbtsvm.update(X=train_data[num_points:],
                   y=train_label[num_points:].reshape(train_label[num_points:].shape[0]),
                   batch_size=batch_size)
    after = time.monotonic()
    u_elapsed = after - before

    # Prediction
    accuracy_2 = ifbtsvm.score(X=test_data, y=test_label)
    print(f'SEA100K\t'
          f'Update (BatchSize|Accuracy|Time): {batch_size}|{np.around(accuracy_2 * 100.0, 3)}%|{np.around(u_elapsed, 3)}s')
    return accuracy_2, (elapsed + u_elapsed)


def sea1M(forget_score):
    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.1,
        C1=10,
        C2=1,
        C3=10,
        C4=1,
        max_iter=50,
        phi=0.00001,
        kernel=None,  # RBFSampler(gamma=0.6, n_components=1400),
        forget_score=forget_score,
    )

    _data = pd.read_csv(f'{DATA_DIR}/SEA/1M/SEAtrain.csv')
    train_data = _data.values[:1000000, 0:3]
    train_label = _data.values[:1000000, 3:]
    test_data = _data.values[1000000:, 0:3]
    test_label = _data.values[1000000:, 3:]

    ifbtsvm = iFBTSVM(parameters=params, n_jobs=4)

    # Training
    num_points = 50000
    before = time.monotonic()
    ifbtsvm.fit(X=train_data[:num_points],
                y=train_label[:num_points].reshape(train_label[:num_points].shape[0]))
    after = time.monotonic()
    elapsed = (after - before)
    accuracy_1 = ifbtsvm.score(X=test_data, y=test_label)

    print(f'SEA1M\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t')

    # Update
    batch_size = 10000
    before = time.monotonic()
    ifbtsvm.update(X=train_data[num_points:],
                   y=train_label[num_points:].reshape(train_label[num_points:].shape[0]),
                   batch_size=batch_size)
    after = time.monotonic()
    u_elapsed = after - before

    # Prediction
    accuracy_2 = ifbtsvm.score(X=test_data, y=test_label)
    print(f'SEA1M\t'
          f'Update (BatchSize|Accuracy|Time): {batch_size}|{np.around(accuracy_2 * 100.0, 3)}%|{np.around(u_elapsed, 3)}s')
    return accuracy_2, (elapsed + u_elapsed)


def letter(forget_score):
    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.1,
        C1=8,
        C2=2,
        C3=8,
        C4=2,
        max_iter=50,
        phi=0.00001,
        kernel=RBFSampler(gamma=0.03, n_components=500),
        forget_score=forget_score,
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

    print(f'Letter\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t')

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
          f'Update (BatchSize|Accuracy|Time): {batch_size}|{np.around(accuracy_2 * 100.0, 3)}%|{np.around(u_elapsed, 3)}s')
    return accuracy_2, (elapsed + u_elapsed)


def dna(forget_score):
    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.1,
        C1=4,
        C2=4,
        C3=4,
        C4=4,
        max_iter=50,
        phi=0.00001,
        kernel=RBFSampler(gamma=0.01, n_components=1000),
        forget_score=forget_score,
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
    num_points = int(len(train_data) / 100 * 5 + 0.5)  # 5% of original dataset
    before = time.monotonic()
    ifbtsvm.fit(X=train_data[:num_points],
                y=train_label[:num_points].reshape(train_label[:num_points].shape[0]))
    after = time.monotonic()
    elapsed = (after - before)
    accuracy_1 = ifbtsvm.score(X=test_data, y=test_label)


    print(f'DNA\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t')

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
          f'Update (BatchSize|Accuracy|Time): {batch_size}|{np.around(accuracy_2 * 100.0, 3)}%|{np.around(u_elapsed, 3)}s')
    return accuracy_2, (elapsed + u_elapsed)


def usps(forget_score):
    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.1,
        C1=8,
        C2=2,
        C3=8,
        C4=2,
        max_iter=50,
        phi=0.00001,
        kernel=RBFSampler(gamma=0.01, n_components=1000),
        forget_score=forget_score,
    )

    train_data = np.zeros((7291, 256))
    train_label = np.zeros((7291,))
    with open(f'{DATA_DIR}/usps', 'r') as f_in:
        for i, line in enumerate(f_in):
            split = line.split(' ')
            train_label[i] = split[0]
            for s in split[1:]:
                if s == '\n':
                    continue
                feats = s.split(':')
                train_data[i, int(feats[0]) - 1] = float(feats[1])

    test_data = np.zeros((2007, 256))
    test_label = np.zeros((2007,))
    with open(f'{DATA_DIR}/usps.t', 'r') as f_in:
        for i, line in enumerate(f_in):
            split = line.split(' ')
            test_label[i] = split[0]
            for s in split[1:]:
                if s == '\n':
                    continue
                feats = s.split(':')
                test_data[i, int(feats[0]) - 1] = float(feats[1])

    ifbtsvm = iFBTSVM(parameters=params, n_jobs=4)

    indices = np.arange(train_data.shape[0])
    np.random.shuffle(indices)
    train_data = train_data[indices]
    train_label = train_label[indices]

    # Training
    num_points = 1000
    before = time.monotonic()
    ifbtsvm.fit(X=train_data[:num_points],
                y=train_label[:num_points].reshape(train_label[:num_points].shape[0]))
    after = time.monotonic()
    elapsed = (after - before)
    accuracy_1 = ifbtsvm.score(X=test_data, y=test_label)

    print(f'USPS\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t')

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
          f'Update (BatchSize|Accuracy|Time): {batch_size}|{np.around(accuracy_2 * 100.0, 3)}%|{np.around(u_elapsed, 3)}s')
    return accuracy_2, (elapsed + u_elapsed)


def isolet(forget_score):
    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.1,
        C1=10,
        C2=10,
        C3=10,
        C4=10,
        max_iter=50,
        phi=0.00001,
        kernel=RBFSampler(gamma=0.001, n_components=1000),
        forget_score=forget_score,
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

    print(f'Isolet\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t')

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
          f'Update (BatchSize|Accuracy|Time): {batch_size}|{np.around(accuracy_2 * 100.0, 3)}%|{np.around(u_elapsed, 3)}s')
    return accuracy_2, (elapsed + u_elapsed)


def gisette(forget_score):

    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.1,
        C1=8,
        C2=2,
        C3=8,
        C4=2,
        max_iter=50,
        phi=0.00001,
        kernel=None,
        forget_score=forget_score,
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

    print(f'Gisette\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t')

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
    return accuracy_2, (elapsed + u_elapsed)


def susy(forget_score):

    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.1,
        C1=10,
        C2=2,
        C3=10,
        C4=2,
        max_iter=50,
        phi=0.00001,
        kernel=RBFSampler(gamma=0.2, n_components=300),
        forget_score=forget_score,
    )

    _data = pd.read_csv(f'/media/karl/DataLake/SUSY.csv', delim_whitespace=True)

    train_data = _data.values[:4500000, 1:]
    train_label = _data.values[:4500000, 0]
    test_data = _data.values[4500000:, 1:]
    test_label = _data.values[4500000:, 0]

    ifbtsvm = iFBTSVM(parameters=params, n_jobs=4)

    # Training
    num_points = 100000
    before = time.monotonic()
    ifbtsvm.fit(X=train_data[:num_points].values,
                y=train_label[:num_points].values.reshape(train_label[:num_points].values.shape[0]))
    after = time.monotonic()
    elapsed = (after - before)
    accuracy_1 = ifbtsvm.score(X=test_data.values, y=test_label.values)

    print(f'SUSY\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t')

    # Update
    batch_size = 100000
    before = time.monotonic()
    ifbtsvm.update(X=train_data[num_points:].values,
                   y=train_label[num_points:].values.reshape(train_label[num_points:].values.shape[0]),
                   batch_size=batch_size)
    after = time.monotonic()
    u_elapsed = after - before

    # Prediction
    accuracy_2 = ifbtsvm.score(X=test_data.values, y=test_label.values)
    print(f'SUSY\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t'
          f'Update (BatchSize|Accuracy|Time): {batch_size}|{np.around(accuracy_2 * 100.0, 3)}%|{np.around(u_elapsed, 3)}s')
    return accuracy_2, (elapsed + u_elapsed)


def wesad(forget_score):
    params = Hyperparameters(
        epsilon=1e-10,
        fuzzy=0.1,
        C1=10,
        C2=2,
        C3=10,
        C4=2,
        max_iter=50,
        phi=0.00001,
        kernel=None, #RBFSampler(gamma=0.2, n_components=300),
        forget_score=forget_score,
    )

    # TODO load WESAD data
    _data = pd.read_csv(f'/media/karl/DataLake/', delim_whitespace=True)

    train_data = _data.values[:4500000, 1:]
    train_label = _data.values[:4500000, 0]
    test_data = _data.values[4500000:, 1:]
    test_label = _data.values[4500000:, 0]

    ifbtsvm = iFBTSVM(parameters=params, n_jobs=4)

    # Training
    num_points = 1537900
    before = time.monotonic()
    ifbtsvm.fit(X=train_data[:num_points].values,
                y=train_label[:num_points].values.reshape(train_label[:num_points].values.shape[0]))
    after = time.monotonic()
    elapsed = (after - before)
    accuracy_1 = ifbtsvm.score(X=test_data.values, y=test_label.values)

    print(f'WESAD\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t')

    # Update
    batch_size = 1537900
    before = time.monotonic()
    ifbtsvm.update(X=train_data[num_points:].values,
                   y=train_label[num_points:].values.reshape(train_label[num_points:].values.shape[0]),
                   batch_size=batch_size)
    after = time.monotonic()
    u_elapsed = after - before

    # Prediction
    accuracy_2 = ifbtsvm.score(X=test_data.values, y=test_label.values)
    print(f'WESAD\t'
          f'Training (DataPoints|Accuracy|Time): {num_points}|{np.around(accuracy_1 * 100.0, 3)}%|{np.around(elapsed, 3)}s\t'
          f'Update (BatchSize|Accuracy|Time): {batch_size}|{np.around(accuracy_2 * 100.0, 3)}%|{np.around(u_elapsed, 3)}s')
    return accuracy_2, (elapsed + u_elapsed)


if __name__ == '__main__':
    from datetime import datetime, timezone

    for forget_score in [1, 2, 4, 10]:
        print(f"Forget score {forget_score}")
        with open(f'./logs/benchmarks_{forget_score}_inc_{str(datetime.now(tz=timezone.utc))}.log', 'w') as f_out:
            for dataset in [led100K, led1M, hyper100K, hyper1M, rtg100K, rtg1M, rbf100K, rbf1M, sea100K, sea1M]:
                res = []
                tmg = []
                for i in range(10):
                    _res, _time = dataset(forget_score)
                    res.append(_res)
                    tmg.append(_time)
                res = np.asarray(res)
                tmg = np.asarray(tmg)
                print(f'{dataset.__name__} ACC: mean:{res.mean()} stdev:{res.std()} max:{np.max(res)} min:{np.min(res)}')
                f_out.write(f'{dataset.__name__} ACC: mean:{res.mean()} stdev:{res.std()} '
                            f'max:{np.max(res)} min:{np.min(res)}\n')
                print(f'{dataset.__name__} TIME: mean:{tmg.mean()} stdev:{tmg.std()} max:{np.max(tmg)} min:{np.min(tmg)}')
                f_out.write(f'{dataset.__name__} TIME: mean:{tmg.mean()} stdev:{tmg.std()} '
                            f'max:{np.max(tmg)} min:{np.min(tmg)}\n')
