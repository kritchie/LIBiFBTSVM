
import numpy as np
import pytest

from libifbtsvm.functions import train_model
from libifbtsvm.models.ifbtsvm import Hyperparameters


def test_train_model():

    H_p = np.ones((5, 5))
    H_n = np.ones((5, 5))

    CCx = np.ones((5, 1))
    C = 2

    _mock_params = Hyperparameters()
    _mock_params.max_iter = 1
    _mock_params.epsilon = 0.0001

    model = train_model(parameters=_mock_params, H=H_p, G=H_n, C=C, CCx=CCx)
    assert np.array_equal(model.alpha, np.array([1., 1., 1., 1., 1.]))

    _truth = [np.array(val) for val in [-1, -0.8, -0.6, -0.4, -0.2]]

    for i in range(5):
        assert model.projected_gradients[i] == pytest.approx(_truth[i])
