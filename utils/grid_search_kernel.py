"""
This file contains code to run benchmarks against the LIBiFBTSVM.

Those benchmarks and evaluation are based on (de Mello, A. R., Stemmer, M. R., & Koerich, A. L., 2019)
"""
import os
import numpy as np
import pandas as pd
import time

from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import train_test_split

from libifbtsvm import iFBTSVM, Hyperparameters


DATA_DIR = os.getenv('DATA_DIR', './data')


def border():

    gammas = [1, 0.1, 0.01, 0.001, 0.0001]
    components = [10, 100, 1000, 3000]

    acc = 0
    best_params = {}

    train_data = pd.read_csv(f'{DATA_DIR}/Border_train_data.csv')
    train_label = pd.read_csv(f'{DATA_DIR}/Border_train_label.csv')

    train_data, test_data, train_label, test_label = train_test_split(
        train_data.values,
        train_label.values,
        test_size=0.33,
        shuffle=True
    )

    with open(f'./logs/border_{int(time.time())}.tsv', 'w') as f_out:
        f_out.write(f'Accuracy\tGamma\tNComponents\tElapsed(s)\n')

        for gamma in gammas:
            for n_comp in components:

                params = Hyperparameters(
                    epsilon=1e-10,
                    fuzzy=0.1,
                    C1=8,
                    C2=2,
                    C3=8,
                    C4=2,
                    max_iter=250,
                    phi=0,
                    kernel=RBFSampler(gamma=gamma, n_components=n_comp),
                    forget_score=10,
                )

                ifbtsvm = iFBTSVM(parameters=params, n_jobs=4)

                # Training
                before = time.monotonic()
                ifbtsvm.fit(X=train_data, y=train_label.reshape(train_label.shape[0]))
                after = time.monotonic()
                elapsed = (after - before)

                # Prediction
                accuracy = ifbtsvm.score(X=test_data, y=test_label)

                if accuracy > acc:
                    acc = accuracy
                    best_params = {'gamma': gamma, 'n_comp': n_comp}

                print(f'Completed run with: {dict(gamma=gamma, n_comp=n_comp)} - Acc : {accuracy} - Elapsed: {np.around(elapsed, 3)}')

                f_out.write(f'{accuracy}\t{gamma}\t{n_comp}\t{np.around(elapsed, 3)}\n')


def coil():

    gammas = [100, 50, 20, 10, 1]
    components = [100, 500, 1000]

    acc = 0
    best_params = {}

    with open(f'./logs/coil_{int(time.time())}.tsv', 'w') as f_out:
        f_out.write(f'Accuracy\tGamma\tNComponents\tElapsed(s)\n')

        for gamma in gammas:
            for n_comp in components:

                params = Hyperparameters(
                    epsilon=1e-10,
                    fuzzy=0.1,
                    C1=10,
                    C2=10,
                    C3=10,
                    C4=10,
                    max_iter=250,
                    phi=0,
                    kernel=RBFSampler(gamma=gamma, n_components=n_comp),
                    forget_score=10,
                )

                train_data = pd.read_csv(f'{DATA_DIR}/Coil_train_data.csv')
                train_label = pd.read_csv(f'{DATA_DIR}/Coil_train_label.csv')

                train_data, test_data, train_label, test_label = train_test_split(
                    train_data.values,
                    train_label.values,
                    test_size=0.33,
                    shuffle=True
                )

                ifbtsvm = iFBTSVM(parameters=params, n_jobs=4)

                # Training
                before = time.monotonic()
                ifbtsvm.fit(X=train_data, y=train_label.reshape(train_label.shape[0]))
                after = time.monotonic()
                elapsed = (after - before)

                # Prediction
                accuracy = ifbtsvm.score(X=test_data, y=test_label)

                if accuracy > acc:
                    acc = accuracy
                    best_params = {'gamma': gamma, 'n_comp': n_comp}

                print(f'Completed run with: {dict(gamma=gamma, n_comp=n_comp)} - Acc : {accuracy} - Elapsed: {np.around(elapsed, 3)}')

                f_out.write(f'{accuracy}\t{gamma}\t{n_comp}\t{np.around(elapsed, 3)}\n')


def mnist():
    gammas = [1, 0.1, 0.01, 0.001]
    components = [100, 500, 1000]

    acc = 0

    with open(f'./logs/mnist_{int(time.time())}.tsv', 'w') as f_out:
        f_out.write(f'Accuracy\tGamma\tNComponents\tElapsed(s)\n')

        for gamma in gammas:
            for n_comp in components:

                params = Hyperparameters(
                    epsilon=1e-10,
                    fuzzy=0.1,
                    C1=10,
                    C2=10,
                    C3=10,
                    C4=10,
                    max_iter=250,
                    phi=0,
                    kernel=RBFSampler(gamma=gamma, n_components=n_comp),
                    forget_score=10,
                )

                train_data = pd.read_csv(f'{DATA_DIR}/MNIST_train_data.csv')
                train_label = pd.read_csv(f'{DATA_DIR}/MNIST_train_label.csv')

                train_data, test_data, train_label, test_label = train_test_split(
                    train_data.values,
                    train_label.values,
                    test_size=0.50,
                    shuffle=True
                )

                ifbtsvm = iFBTSVM(parameters=params, n_jobs=4)

                # Training
                before = time.monotonic()
                ifbtsvm.fit(X=train_data, y=train_label.reshape(train_label.shape[0]))
                after = time.monotonic()
                elapsed = (after - before)

                # Prediction
                accuracy = ifbtsvm.score(X=test_data, y=test_label)

                if accuracy > acc:
                    acc = accuracy
                    best_params = {'gamma': gamma, 'n_comp': n_comp}

                print(f'Completed run with: {dict(gamma=gamma, n_comp=n_comp)} - Acc : {accuracy} - Elapsed: {np.around(elapsed, 3)}')

                print(f't{accuracy}\t{gamma}\t{n_comp}\t{np.around(elapsed, 3)}\n')


def outdoor():
    gammas = [5, 1, 0.1]
    components = [10, 100, 1000]

    acc = 0
    best_params = {}

    with open(f'./logs/outdoor_{int(time.time())}.tsv', 'w') as f_out:
        f_out.write(f'Accuracy\tGamma\tNComponents\tElapsed(s)\n')

        for gamma in gammas:
            for n_comp in components:

                params = Hyperparameters(
                    epsilon=1e-10,
                    fuzzy=0.1,
                    C1=10,
                    C2=1,
                    C3=10,
                    C4=1,
                    max_iter=250,
                    phi=0,
                    kernel=RBFSampler(gamma=gamma, n_components=n_comp),
                    forget_score=10,
                )

                train_data = pd.read_csv(f'{DATA_DIR}/Outdoor_train_data.csv')
                train_label = pd.read_csv(f'{DATA_DIR}/Outdoor_train_label.csv')

                train_data, test_data, train_label, test_label = train_test_split(
                    train_data.values,
                    train_label.values,
                    test_size=0.33,
                    shuffle=True
                )

                ifbtsvm = iFBTSVM(parameters=params, n_jobs=4)

                # Training
                before = time.monotonic()
                ifbtsvm.fit(X=train_data, y=train_label.reshape(train_label.shape[0]))
                after = time.monotonic()
                elapsed = (after - before)

                # Prediction
                accuracy = ifbtsvm.score(X=test_data, y=test_label)

                if accuracy > acc:
                    acc = accuracy
                    best_params = {'gamma': gamma, 'n_comp': n_comp}

                print(f'Completed run with: {dict(gamma=gamma, n_comp=n_comp)} - Acc : {accuracy} - Elapsed: {np.around(elapsed, 3)}')

                f_out.write(f'{accuracy}\t{gamma}\t{n_comp}\t{np.around(elapsed, 3)}\n')


def overlap():
    gammas = [1, 0.1, 0.01, 0.001, 0.0001]
    components = [10, 100, 1000, 3000]

    acc = 0
    best_params = {}

    with open(f'./logs/overlap_{int(time.time())}.tsv', 'w') as f_out:
        f_out.write(f'Accuracy\tGamma\tNComponents\tElapsed(s)\n')

        for gamma in gammas:
            for n_comp in components:

                params = Hyperparameters(
                    epsilon=1e-10,
                    fuzzy=0.1,
                    C1=8,
                    C2=2,
                    C3=8,
                    C4=2,
                    max_iter=250,
                    phi=0,
                    kernel=RBFSampler(gamma=gamma, n_components=n_comp),
                    forget_score=10,
                )

                train_data = pd.read_csv(f'{DATA_DIR}/Overlap_train_data.csv')
                train_label = pd.read_csv(f'{DATA_DIR}/Overlap_train_label.csv')

                train_data, test_data, train_label, test_label = train_test_split(
                    train_data.values,
                    train_label.values,
                    test_size=0.33,
                    shuffle=True
                )

                ifbtsvm = iFBTSVM(parameters=params, n_jobs=4)

                # Training
                before = time.monotonic()
                ifbtsvm.fit(X=train_data, y=train_label.reshape(train_label.shape[0]))
                after = time.monotonic()
                elapsed = (after - before)

                # Prediction
                accuracy = ifbtsvm.score(X=test_data, y=test_label)

                if accuracy > acc:
                    acc = accuracy
                    best_params = {'gamma': gamma, 'n_comp': n_comp}

                print(f'Completed run with: {dict(gamma=gamma, n_comp=n_comp)} - Acc : {accuracy} - Elapsed: {np.around(elapsed, 3)}')

                f_out.write(f'{accuracy}\t{gamma}\t{n_comp}\t{np.around(elapsed, 3)}\n')


def rbf():
    gammas = [10, 5, 1, 0.1]
    components = [500, 750, 1000, 1250, 1500]

    acc = 0
    best_params = {}

    with open(f'./logs/rbf_{int(time.time())}.tsv', 'w') as f_out:
        f_out.write(f'Accuracy\tGamma\tNComponents\tElapsed(s)\n')

        for gamma in gammas:
            for n_comp in components:

                params = Hyperparameters(
                    epsilon=1e-10,
                    fuzzy=0.1,
                    C1=8,
                    C2=2,
                    C3=8,
                    C4=2,
                    max_iter=250,
                    phi=0,
                    kernel=RBFSampler(gamma=gamma, n_components=n_comp),
                    forget_score=10,
                )

                _data = pd.read_csv(f'{DATA_DIR}/RBF/10K/RBF10K.csv')
                train_data = _data.values[:10000, 0:10]
                train_label = _data.values[:10000, 10:]

                train_data, test_data, train_label, test_label = train_test_split(
                    train_data,
                    train_label,
                    test_size=0.33,
                    shuffle=True
                )

                ifbtsvm = iFBTSVM(parameters=params, n_jobs=4)

                # Training
                before = time.monotonic()
                ifbtsvm.fit(X=train_data, y=train_label.reshape(train_label.shape[0]))
                after = time.monotonic()
                elapsed = (after - before)

                # Prediction
                accuracy = ifbtsvm.score(X=test_data, y=test_label)

                if accuracy > acc:
                    acc = accuracy
                    best_params = {'gamma': gamma, 'n_comp': n_comp}

                print(f'Completed run with: {dict(gamma=gamma, n_comp=n_comp)} - Acc : {accuracy} - Elapsed: {np.around(elapsed, 3)}')

                f_out.write(f'{accuracy}\t{gamma}\t{n_comp}\t{np.around(elapsed, 3)}\n')


def rtg():
    gammas = [1, 0.1, 0.01, 0.001, 0.0001]
    components = [10, 100, 1000, 3000]

    acc = 0
    best_params = {}

    with open(f'./logs/rtg_{int(time.time())}.tsv', 'w') as f_out:
        f_out.write(f'Accuracy\tGamma\tNComponents\tElapsed(s)\n')

        for gamma in gammas:
            for n_comp in components:

                params = Hyperparameters(
                    epsilon=1e-10,
                    fuzzy=0.1,
                    C1=2.5,
                    C2=2,
                    C3=2.5,
                    C4=2,
                    max_iter=250,
                    phi=0,
                    kernel=RBFSampler(gamma=gamma, n_components=n_comp),
                    forget_score=10,
                )

                _data = pd.read_csv(f'{DATA_DIR}/RTG/10K/RTG10K.csv')
                train_data = _data.values[:10000, 0:10]
                train_label = _data.values[:10000, 10:]

                train_data, test_data, train_label, test_label = train_test_split(
                    train_data,
                    train_label,
                    test_size=0.33,
                    shuffle=True
                )

                ifbtsvm = iFBTSVM(parameters=params, n_jobs=4)

                # Training
                before = time.monotonic()
                ifbtsvm.fit(X=train_data, y=train_label.reshape(train_label.shape[0]))
                after = time.monotonic()
                elapsed = (after - before)

                # Prediction
                accuracy = ifbtsvm.score(X=test_data, y=test_label)

                if accuracy > acc:
                    acc = accuracy
                    best_params = {'gamma': gamma, 'n_comp': n_comp}

                print(f'Completed run with: {dict(gamma=gamma, n_comp=n_comp)} - Acc : {accuracy} - Elapsed: {np.around(elapsed, 3)}')

                f_out.write(f'{accuracy}\t{gamma}\t{n_comp}\t{np.around(elapsed, 3)}\n')


def letter():
    gammas = [1, 0.1, 0.01, 0.001, 0.0001]
    components = [10, 100, 1000, 3000]

    acc = 0
    best_params = {}

    _data = pd.read_csv(f'{DATA_DIR}/letter-recognition.data')
    train_data = _data.values[:16000, 1:]
    train_label = _data.values[:16000, 0]

    train_data, test_data, train_label, test_label = train_test_split(
        train_data,
        train_label,
        test_size=0.33,
        shuffle=True
    )

    for i, lbl in enumerate(train_label):
        train_label[i] = ord(lbl) - 65  # '65' -> 'A'

    for i, lbl in enumerate(test_label):
        test_label[i] = ord(lbl) - 65  # '65' -> 'A'

    with open(f'./logs/letter_{int(time.time())}.tsv', 'w') as f_out:
        f_out.write(f'Accuracy\tGamma\tNComponents\tElapsed(s)\n')

        for gamma in gammas:
            for n_comp in components:

                params = Hyperparameters(
                    epsilon=1e-10,
                    fuzzy=0.1,
                    C1=8,
                    C2=2,
                    C3=8,
                    C4=2,
                    max_iter=250,
                    phi=0,
                    kernel=RBFSampler(gamma=gamma, n_components=n_comp),
                    forget_score=10,
                )

                test_label = test_label.reshape(test_label.shape[0], 1).astype(np.int)

                ifbtsvm = iFBTSVM(parameters=params, n_jobs=4)

                # Training
                before = time.monotonic()
                ifbtsvm.fit(X=train_data, y=train_label)
                after = time.monotonic()
                elapsed = (after - before)

                # Prediction
                accuracy = ifbtsvm.score(X=test_data, y=test_label.reshape(test_label.shape[0], 1))

                if accuracy > acc:
                    acc = accuracy
                    best_params = {'gamma': gamma, 'n_comp': n_comp}

                print(f'Completed run with: {dict(gamma=gamma, n_comp=n_comp)} - Acc : {accuracy} - Elapsed: {np.around(elapsed, 3)}')

                f_out.write(f'{accuracy}\t{gamma}\t{n_comp}\t{np.around(elapsed, 3)}\n')


def dna():
    gammas = [1, 0.1, 0.01, 0.001, 0.0001]
    components = [10, 100, 1000, 3000]

    acc = 0
    best_params = {}

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
                train_data[i, int(feats[0]) - 1] = int(feats[1])

    train_data, test_data, train_label, test_label = train_test_split(
        train_data,
        train_label,
        test_size=0.33,
        shuffle=True
    )

    with open(f'./logs/dna_{int(time.time())}.tsv', 'w') as f_out:
        f_out.write(f'Accuracy\tGamma\tNComponents\tElapsed(s)\n')

        for gamma in gammas:
            for n_comp in components:

                params = Hyperparameters(
                    epsilon=1e-10,
                    fuzzy=0.1,
                    C1=4,
                    C2=4,
                    C3=4,
                    C4=4,
                    max_iter=250,
                    phi=0,
                    kernel=RBFSampler(gamma=gamma, n_components=n_comp),
                    forget_score=10,
                )

                ifbtsvm = iFBTSVM(parameters=params, n_jobs=4)

                # Training
                before = time.monotonic()
                ifbtsvm.fit(X=train_data, y=train_label)
                after = time.monotonic()
                elapsed = (after - before)

                # Prediction
                accuracy = ifbtsvm.score(X=test_data, y=test_label)

                if accuracy > acc:
                    acc = accuracy
                    best_params = {'gamma': gamma, 'n_comp': n_comp}

                print(f'Completed run with: {dict(gamma=gamma, n_comp=n_comp)} - Acc : {accuracy} - Elapsed: {np.around(elapsed, 3)}')

                f_out.write(f'{accuracy}\t{gamma}\t{n_comp}\t{np.around(elapsed, 3)}\n')


def usps():
    gammas = [1, 0.1, 0.01, 0.001, 0.0001]
    components = [10, 100, 1000, 3000]

    acc = 0
    best_params = {}

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

    train_data, test_data, train_label, test_label = train_test_split(
        train_data,
        train_label,
        test_size=0.33,
        shuffle=True
    )

    with open(f'./logs/usps_{int(time.time())}.tsv', 'w') as f_out:
        f_out.write(f'Accuracy\tGamma\tNComponents\tElapsed(s)\n')

        for gamma in gammas:
            for n_comp in components:

                params = Hyperparameters(
                    epsilon=1e-10,
                    fuzzy=0.1,
                    C1=8,
                    C2=2,
                    C3=8,
                    C4=2,
                    max_iter=250,
                    phi=0,
                    kernel=RBFSampler(gamma=gamma, n_components=n_comp),
                    forget_score=10,
                )

                ifbtsvm = iFBTSVM(parameters=params, n_jobs=4)

                # Training
                before = time.monotonic()
                ifbtsvm.fit(X=train_data, y=train_label)
                after = time.monotonic()
                elapsed = (after - before)

                # Prediction
                accuracy = ifbtsvm.score(X=test_data, y=test_label)

                if accuracy > acc:
                    acc = accuracy
                    best_params = {'gamma': gamma, 'n_comp': n_comp}

                print(f'Completed run with: {dict(gamma=gamma, n_comp=n_comp)} - Acc : {accuracy} - Elapsed: {np.around(elapsed, 3)}')

                f_out.write(f'{accuracy}\t{gamma}\t{n_comp}\t{np.around(elapsed, 3)}\n')


def isolet():
    gammas = [1, 0.1, 0.01, 0.001, 0.0001]
    components = [10, 100, 1000, 3000]

    acc = 0
    best_params = {}

    with open(f'./logs/isolet_{int(time.time())}.tsv', 'w') as f_out:
        f_out.write(f'Accuracy\tGamma\tNComponents\tElapsed(s)\n')

        for gamma in gammas:
            for n_comp in components:

                params = Hyperparameters(
                    epsilon=1e-10,
                    fuzzy=0.1,
                    C1=10,
                    C2=10,
                    C3=10,
                    C4=10,
                    max_iter=250,
                    phi=0,
                    kernel=RBFSampler(gamma=gamma, n_components=n_comp),
                    forget_score=10,
                )

                _data = pd.read_csv(f'{DATA_DIR}/isolet1+2+3+4.data')
                train_data = _data.values[:, :617]
                train_label = _data.values[:, 617]

                train_data, test_data, train_label, test_label = train_test_split(
                    train_data,
                    train_label,
                    test_size=0.33,
                    shuffle=True
                )

                ifbtsvm = iFBTSVM(parameters=params, n_jobs=4)

                # Training
                before = time.monotonic()
                ifbtsvm.fit(X=train_data, y=train_label.reshape(train_label.shape[0]))
                after = time.monotonic()
                elapsed = (after - before)

                # Prediction
                accuracy = ifbtsvm.score(X=test_data, y=test_label)

                if accuracy > acc:
                    acc = accuracy
                    best_params = {'gamma': gamma, 'n_comp': n_comp}

                print(f'Completed run with: {dict(gamma=gamma, n_comp=n_comp)} - Acc : {accuracy} - Elapsed: {np.around(elapsed, 3)}')

                f_out.write(f'{accuracy}\t{gamma}\t{n_comp}\t{np.around(elapsed, 3)}\n')


if __name__ == '__main__':
    # border()
    # coil()
    # overlap()
    # outdoor()
    # mnist()
    # rbf()
    rtg()
    letter()
    dna()
    usps()
    isolet()
