from BP import NeuralNetwork

import numpy as np

nn = NeuralNetwork([2, 2, 1], 'tanh')

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

y = np.array([1, 0, 0, 1])

nn.fit(x, y, 0.1, 10000)

for i in [[0, 0], [0, 1], [1, 0], [1, 1]]:
    print(i, nn.predict(i))