"""
A collection of activation functions, and their derivatives
"""

import numpy as np


def relu(z):
    """Compute the rectified linear activation of unit z.
    N.B. Freshly allocates a new array each function called.

    :param z: The array of values for a unit, pre-activation.
    :return: ReLu evaluated at z.
    """

    return np.maximum(z, 0)


def d_relu(x):
    """Compute derivative of ReLu w/r/t x.

    :param x: vector value we wish to evaluate the gradient at.
    :return: d_relu/d x
    """

    x[x <= 0] = 0
    x[x > 0] = 1

    return x


def sigmoid(z):
    """Compute the sigmoid activation of z

    :param z: The array of values for a unit, pre-activation.
    :return: the sigmoid activation of z
    """

    return 1.0 / (1.0 + np.exp(-z))


def d_sigmoid(x):
    """Compute the deriv. of sigmoid w/r/t x

    :param x: scalar value we wish to evaluate the gradient at.
    :return: d_sigmoid/d x
    """
    return sigmoid(x) * (1 - sigmoid(x))
