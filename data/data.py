import pickle
import gzip
import numpy as np


def get_MNIST():
    """Return the MNIST data
    :return: A tuple (training, CV, test) data sets.
    """
    f = gzip.open('./data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f,
                                                            encoding='latin1')
    f.close()

    return (training_data, validation_data, test_data)


def load_data_wrapper():
    tr_d, va_d, te_d = get_MNIST()

    training_results = [one_hot(y) for y in tr_d[1]]
    training_data = (tr_d[0], training_results)

    validation_results = [one_hot(y) for y in va_d[1]]
    validation_data = (va_d[0], validation_results)

    test_results = [one_hot(y) for y in te_d[1]]
    test_data = (te_d[0], test_results)

    return (training_data, validation_data, test_data)


def one_hot(j):
    e = np.zeros((10, 1))
    e[j] = 1.0

    return e