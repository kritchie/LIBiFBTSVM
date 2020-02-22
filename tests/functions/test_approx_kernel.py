from collections import namedtuple

import pytest

from libifbtsvm.functions.approx_kernel import approx_kernel
from libifbtsvm.kernels import Kernel


def test_approx_kernel_invalid_kernel(valid_features_a):
    """
    Test assert
    """
    with pytest.raises(ValueError) as ve:
        approx_kernel(kernel=None, x=valid_features_a, y=valid_features_a)

    assert 'Kernel cannot be null' == ve.value.args[0]

    NotCallable = namedtuple('NotCallable', 'fit_transform')
    with pytest.raises(ValueError) as ve:
        approx_kernel(kernel=NotCallable(fit_transform=42), x=valid_features_a, y=valid_features_a)

    assert 'Kernel must implement a "fit_transform(X, y)" method to be usable.' == ve.value.args[0]

    CallableButWrong = namedtuple('CallableButWrong', 'not_fit_transform')

    def fit_transform():
        return True

    with pytest.raises(ValueError) as ve:
        approx_kernel(kernel=CallableButWrong(not_fit_transform=fit_transform),
                      x=valid_features_a, y=valid_features_a)

    assert 'Kernel must implement a "fit_transform(X, y)" method to be usable.' == ve.value.args[0]


def test_approx_kernel_invalid_features(invalid_features_a, valid_features_a):
    """
    Test passing an invalid features array results in a ValueError exception.
    """
    with pytest.raises(ValueError) as ve:
        approx_kernel(kernel=Kernel.RBF, x=invalid_features_a, y=valid_features_a)

    assert 'Unsupported feature type' in ve.value.args[0]


def test_approx_kernel_custom(sum_kernel, valid_features_a):
    """
    Test an arbitrary user-defined kernel for feature approximation.
    """
    values = approx_kernel(kernel=sum_kernel, x=valid_features_a, y=valid_features_a)
    assert values.tolist()[0] == [2, 4, 6, 8, 10]


def test_approx_kernel_rbf(valid_features_a):
    """
    Test the feature approximation using an RBF kernel.
    """
    values = approx_kernel(kernel=Kernel.RBF, x=valid_features_a, y=valid_features_a)
    assert values.shape == (1, 100)


def test_approx_kernel_achi2(valid_features_a):
    """
    Test the feature approximation using an ACHI2 kernel.
    """
    values = approx_kernel(kernel=Kernel.ACHI2, x=valid_features_a, y=valid_features_a)
    assert values.shape == (1, 95)
