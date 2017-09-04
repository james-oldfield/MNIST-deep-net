import numpy as np
from activations import relu, d_relu, sigmoid, d_sigmoid


class DeepNet(object):
    def __init__(self, architecture):
        """Initalise the weights, biases, etc.

        :param architecture: A list containing the network's desired
            layer architecture. [2, 5, 5, 1] would contain two 5 unit hidden
            layers, for example.
        :return: The parameters dictionary with initialised weights + biases.
        """

        self.parameters = {}  # stores the weights, biases, etc.
        self.L = len(architecture)  # no. of layers
        a = architecture

        """Initialise all the weights and biases in the network, given
        the architecture requested. Biases are zero initalised. Weights
        are randomly initalised from Gaussian dist.
        """
        for l in range(1, self.L):
            self.parameters['W{}'.format(l)] = np.random.randn(a[l],
                                                               a[l-1]) * 0.01
            self.parameters['b{}'.format(l)] = np.zeros((a[l], 1))

    def get_parameters(self):
        """Return the net's params.

        :return: net's params.
        """

        return self.parameters

    def activate(self, A_prev, W, b, activ_fn=relu):
        """Compute a layer's activation values, given the previous layer's.

        :param A_prev: previous layer's activations.
        :param W: Weights max to compute next layer.
        :param b: ditto above, but for bias term.
        :param activ_fn: desired activation function.
        :return A, (cache): activation for this layer + useful
            values from this layer for backprop.
        """

        Z = np.dot(W, A_prev) + b
        A = activ_fn(Z)

        return A, ((A_prev, W, b), Z)

    def feedforward(self, A):
        """Forward propagate using ReLu activation functions up to
        layer (L-1), and then a sigmoid activation for the output.

        :param A: The training data set (or minibatch subset)
        """

        caches = []

        # Compute each layer's activations using ReLu, up until L-1.
        # Store the relevant values in `cache` list for use with bprop.
        for l in range(1, self.L):
            A_prev = A
            W = self.parameters['W{}'.format(l)]
            b = self.parameters['b{}'.format(l)]

            A, cache = self.activate(A_prev, W, b)
            caches.append(cache)

        # Compute the output layer's value using sigmoid.
        WL = self.parameters['W{}'.format(self.L)]
        bL = self.parameters['b{}'.format(self.L)]
        AL, cache = self.activate(A, WL, bL, activ_fn=sigmoid)

        caches.append(cache)

        return AL, caches
