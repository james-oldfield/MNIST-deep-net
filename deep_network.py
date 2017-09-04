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

        # store the reference to derivative of correct function
        if activ_fn is relu:
            d_func = d_relu
        else:
            d_func = d_sigmoid

        return A, (A_prev, W, b, Z, d_func)

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

    def backpropagate(self, AL, Y, caches):
        """Backpropagate the error to all the weights, etc.
        """

        grads = {}
        L = self.L
        Y = Y.reshape(AL.shape)

        # Store the deriv. wrt the output layer
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        current_cache = caches[-1]

        # store the gradients of the output layer.
        grads['dA{}'.format(L)],
        grads["dW{}".format(L)],
        grads["db{}".format(L)] = self.get_prevlayer_gradient(
            dAL, current_cache, activ_fn=sigmoid)

        for l in reversed(range(L-1)):
            current_cache = caches[l]

            dA_prev_temp, dW_temp, db_temp = self.get_prevlayer_gradient(
                grads["dA{}".format(L)], current_cache)

            grads["dA{}".format(l+1)] = dA_prev_temp
            grads["dW{}".format(l+1)] = dW_temp
            grads["db{}".format(l+1)] = db_temp

        return grads

    def get_prevlayer_gradient(self, dA, cache, activ_fn=relu):
        """Compute the gradient of layer (l-1) w/r/t cost, given
        the derivative of the cost function w/r/t layer `l`.

        :param dA: gradient of cost function w/r/t current layer.
        :param cache: the cache of the layer's values.
        :param activ_fn: the activation function used in this layer.
        :return: The derivative of cost fn, w/r/t desired vals.
        """

        A_prev, W, b, Z, d_func = cache
        dZ = np.multiply(dA, d_func(Z))

        m = A_prev.shape[1]

        dW = (1/m) * np.dot(dZ, A_prev.T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db

    def get_cost(self, YHat, Y):
        """Compute the Cross Entropy cost, given our prediction and ground truth y.

        :param YHat: our approximation of Y.
        :param Y: the actual value of Y.
        :return: the cost, as specified by the cross entropy function.
        """
        m = Y.shape[1]  # no. of examples

        # Compute the Cross Entropy cost, given the predicted values
        # and the ground truth values
        cost = -(1/m) * np.sum(np.nan_to_num(
            Y * np.log(YHat) + (1-Y) * np.log(1-YHat)))

        return np.squeeze(cost)
