import pickle
import gzip


def get_MNIST():
    """Return the MNIST data
    :return: A tuple (training, CV, test) data sets.
    """
    f = gzip.open('./data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f,
                                                            encoding='latin1')
    f.close()

    return (training_data, validation_data, test_data)