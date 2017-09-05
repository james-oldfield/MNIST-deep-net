from data import data
import deep_network

training_data, _, _ = data.load_data_wrapper()

nn = deep_network.DeepNet([784, 30, 10])

nn.SGD(training_data, 30, 10, 3.0, debug=True)
