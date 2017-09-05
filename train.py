from data import data
import deep_network

training_data, _, test_data = data.load_data_wrapper()

nn = deep_network.DeepNet([784, 30, 30, 10])
nn.SGD(
    training_data,
    num_epochs=20,
    mini_batch_size=10,
    eta=0.01,
    test_data=test_data,
    save_costs=True)
