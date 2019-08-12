import numpy as np


class Neural:
    def __init__(self):
        np.random.seed(1)
        arr = np.random.random((3, 1))
        print(2*arr-1)
        self.synaptic_weights = 2 * arr - 1
        self.val = 0

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1-x)

    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.__sigmoid(np.dot(inputs, self.synaptic_weights))
        return output

    def train(self, training_in, training_out, training_it=10000):
        for iteration in range(training_it):

            output = self.think(training_in)
            error = training_out - output
            adjustments = np.dot(training_in.T, error *
                                 self.__sigmoid_derivative(output))

            self.synaptic_weights += adjustments
            self.output = output


# if __name__ == "__main__":
#     network = Neural()
#     print("Random synaptic Weights: ")
#     prin
