
import numpy as np
import pytest

from libifbtsvm import iFBTSVM


def test_generate_sub_samples(dataset_3_classes):

    model = iFBTSVM(parameters={})

    sub_data_sets = model.generate_sub_sets(X=dataset_3_classes.X, y=dataset_3_classes.y)

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
