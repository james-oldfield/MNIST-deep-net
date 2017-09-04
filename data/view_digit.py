"""
View a random digit from MNIST dataset, or supply an index in CLI.
"""
import matplotlib.pyplot as plt

from sys import argv
from random import randint
from data import get_MNIST

training_set, _, _ = get_MNIST()

# optional CLI index
if len(argv) < 2:
    index = randint(0, len(training_set[1]))
else:
    index = argv[1]

# rid of rank 1 array format
random_digit = training_set[0][index].reshape(28, -1)

print('this digit is a {}'.format(training_set[1][index]))

plt.imshow(random_digit)
plt.show()
