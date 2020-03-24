
import numpy as np
import pytest

from libifbtsvm import iFBTSVM

from libifbtsvm.models.ifbtsvm import (
    FuzzyMembership,
    Hyperparameters,
    Hyperplane,
)


def test_generate_sub_samples(dataset_3_classes):

    parameters = Hyperparameters()

    model = iFBTSVM(parameters=parameters)

    sub_data_sets = model._generate_sub_sets(X=dataset_3_classes.X, y=dataset_3_classes.y)

    dag_1 = next(sub_data_sets)
    truth_1 = [np.array([0.9, 1.0, 1.1]), np.array(['1', '1', '1']),
               np.array([10.9, 11.0, 11.1]), np.array(['2', '2', '2'])]

    for i in range(len(truth_1)):
        assert np.array_equal(dag_1[i], truth_1[i])

    dag_2 = next(sub_data_sets)
    truth_2 = [np.array([0.9, 1.0, 1.1]), np.array(['1', '1', '1']),
               np.array([110.9, 111.0, 111.1]), np.array(['3', '3', '3'])]

    for i in range(len(truth_2)):
        assert np.array_equal(dag_2[i], truth_2[i])

    dag_3 = next(sub_data_sets)
    truth_3 = [np.array([10.9, 11.0, 11.1]), np.array(['2', '2', '2']),
               np.array([110.9, 111.0, 111.1]), np.array(['3', '3', '3'])]

    for i in range(len(truth_3)):
        assert np.array_equal(dag_3[i], truth_3[i])

    with pytest.raises(StopIteration):
        _ = next(sub_data_sets)


def test_fit_dag_step(mocker):

    z = np.zeros((1, 1))
    fuzzy_membership = FuzzyMembership(noise_n=z, noise_p=z, sp=z, sn=z)
    subset = (z, z, z, z)

    _mock_plane = Hyperplane()
    _mock_plane.weights = np.ones((1, 1))

    _mock_params = Hyperparameters()
    _mock_params.C1 = 1
    _mock_params.C2 = 2
    _mock_params.C3 = 3
    _mock_params.C4 = 4

    mocker.patch('libifbtsvm.libifbtsvm.fuzzy_membership', return_value=fuzzy_membership)
    mocker.patch('libifbtsvm.libifbtsvm.train_model', return_value=_mock_plane)

    model = iFBTSVM._fit_dag_step(subset, _mock_params)
    assert model.get('hyperplaneP').weights[0] == -1
    assert model.get('hyperplaneN').weights[0] == -1
