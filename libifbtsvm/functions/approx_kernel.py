import numpy as np


def approx_kernel(kernel, x: np.array, y: np.array) -> np.ndarray:
    """
    Returns kernel approximated features.

    :param kernel: Kernel for approximated features computation.
                   This is an arbitrary object on which the 'fit_transform' method
                   must be defined.

    :param x: Numpy array of features.
    :param y: Numpy array of features.
    :return: The computed approximate features.
    """
    if not kernel:
        raise ValueError('Kernel cannot be null')

    _op = getattr(kernel, 'fit_transform', None)
    if not _op or not callable(_op):
        raise ValueError('Kernel must implement a "fit_transform(X, y)" method to be usable.')

    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError(f'Unsupported feature type, must be of type: {type(np.array)}')

    return kernel.fit_transform(X=x, y=y)
