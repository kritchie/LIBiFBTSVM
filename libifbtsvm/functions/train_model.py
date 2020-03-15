
import numpy as np


class Hyperplane(object):
    alpha: np.ndarray
    weights: np.ndarray
    iterations: int

    # FIXME verify pg type
    from typing import Any
    pg: Any
