import numpy as np


class DeepNet(object):
    def __init__(self, architecture):
        """Initalises the weights, biases, etc.

        :param architecture: A list containing the network's desired
            layer architecture. [2, 5, 5, 1] would contain two 5 unit hidden
            layers, for example.
        :return: The parameters dictionary with initialised weights + biases.
        """

        self.parameters = {}  # stores the weights, biases, etc.
        L = len(architecture)  # no. of layers
        a = architecture

        """Initialise all the weights and biases in the network, given
        the architecture requested. Biases are zero initalised. Weights
        are randomly initalised from Gaussian dist.
        """
        for l in range(1, L):
            self.parameters['W{}'.format(l)] = np.random.randn(a[l],
                                                               a[l-1]) * 0.01
            self.parameters['b{}'.format(l)] = np.zeros((a[l], 1))

    def get_parameters(self):
        """Returns the net's params.

        :return: net's params.
        """

        return self.parameters
