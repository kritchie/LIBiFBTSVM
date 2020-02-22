import numpy as np

from libifbtsvm.kernels import Kernel


def approx_kernel(kernel: Kernel, x: np.array, y: np.array):
    """
    Returns kernel approximated features.

    :param kernel: Kernel for approximated features computation.
    :param x: Numpy array of non-approximated features.
    :param y: Numpy array of non-approximated features.
    :return: The computer approximate features.
    """

    if not kernel:
        raise ValueError('Kernel cannot be null')

    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError(f'Unsupported feature type, must be of type: {type(np.array)}')

    return kernel(X=x, y=y)


