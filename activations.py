"""
A collection of activation functions, and their derivatives
"""

import numpy as np


def relu(z):
    """Compute the rectified linear activation of unit z.
    N.B. Freshly allocates a new array each function called.

    :param x: The array of values for a unit, pre-activation.
    :return: ReLu evaluated at z.
    """

    return np.maximum(z, 0)


def d_relu(x):
    """Compute derivative of ReLu w/r/t x.

    :param x: scalar value we wish to evaluate the gradient at.
    :return: d_relu/d x
    """

    if x > 0:
        return 1
    else:
        return 0
