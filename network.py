"""A feedforward network of sigmoid neurons, written from scratch with numpy."""
import numpy as np

class Network:
    """Fully connected network: weights and biases for every non-input layer."""

    def __init__(self, sizes):
        """Create random weights and biases for layer sizes, e.g. [784, 30, 10].

        weights[i] has shape (n_out, n_in): one row per neuron of the next layer.
        biases[i] has shape (n_out, 1). The input layer has no biases.
        """
        self.sizes = sizes
        self.weights = []
        self.biases = []

        for n_in, n_out in zip(self.sizes[:-1], self.sizes[1:]):
            self.weights.append(np.random.randn(n_out, n_in))
            self.biases.append(np.random.randn(n_out, 1))


    def feedforward(self, a):
        """Pass a column a of shape (sizes[0], 1) through all layers.

        Returns the output layer activations, shape (sizes[-1], 1).
        """
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(w @ a + b)
        return a


    def evaluate(self, test_data):
        """Return how many (x, y) pairs the network classifies correctly.

        The answer is the index of the brightest output neuron (argmax).
        y must be an int label, so use test or validation data, not training.
        """
        counter = 0

        for x, y in test_data:
            output = self.feedforward(x)
            output_answer = np.argmax(output)
            if output_answer == y:
                counter += 1

        return counter

def sigmoid(z):
    """Elementwise 1 / (1 + e^-z); works on numbers and numpy arrays."""
    return 1/(1 + np.exp(-z))
