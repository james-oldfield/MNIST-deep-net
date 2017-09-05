from data import data
import deep_network

training_data, _, test_data = data.load_data_wrapper()

nn = deep_network.DeepNet([784, 30, 30, 10])
nn.SGD(training_data, 30, 10, 1.0, test_data=test_data)
