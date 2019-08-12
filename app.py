import numpy as np

from Neural import Neural

input_data = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
    [1, 0, 1],
])
# input_result = np.array([[0, 0, 1, 1]]).T
input_result = np.array([[0, 1, 1, 1]]).T

network = Neural()

print('random starting weights: ')
print(network.synaptic_weights)

network.train(input_data, input_result)

# print('Synaptic weights after training')
# print(network.synaptic_weights)

# print('Outputs after training: ')
# print(network.output)

A = str(input("Input 1: "))
B = str(input("Input 2: "))
C = str(input("Input 3: "))

print("New Situation input data = ", A, B, C)
print("Output Data: ")
print(network.think(np.array([A, B, C])))
