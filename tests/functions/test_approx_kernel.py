import pytest

from libifbtsvm.functions.approx_kernel import approx_kernel
from libifbtsvm.kernels import Kernel


def test_approx_kernel_invalid_features(invalid_features_a, valid_features_a):
    """
    Asserts passing an invalid kernel type results in a ValueError exception.
    """
    with pytest.raises(ValueError) as ve:
        approx_kernel(kernel=Kernel.RBF, x=invalid_features_a, y=valid_features_a)

    assert 'Unsupported feature type' in ve.value.args[0]


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
