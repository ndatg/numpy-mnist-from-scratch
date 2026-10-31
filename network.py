"""A feedforward network of sigmoid neurons, written from scratch with numpy."""
import numpy as np
import random

class Network:
    """Fully connected network: weights and biases for every non-input layer."""

    def __init__(self, sizes):
        """Create random weights and biases for layer sizes, e.g. [784, 30, 10].

        weights[i] has shape (n_out, n_in): one row per neuron of the next layer.
        biases[i] has shape (n_out, 1). The input layer has no biases.
        """
        self.sizes = sizes
        self.num_layers = len(sizes)
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


    def cost_derivative(self, output_activations, y):
        """Return dC/da = a - y: the signed error of every output neuron.

        For the quadratic cost C = 1/2 * ||a - y||^2 (the 1/2 cancels here).
        Both arguments are (10, 1) columns; y is one-hot. Used by backprop.
        """
        return (output_activations-y)


    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train with mini-batch stochastic gradient descent.

        training_data: list of (x, y) pairs, y one-hot (10, 1). A shallow copy
        is shuffled each epoch, so the caller's list keeps its order.
        Each epoch: shuffle, cut into mini-batches, one update_mini_batch per
        batch (eta = learning rate). If test_data is given, print evaluate()
        after every epoch (slower, but shows whether the net is learning).
        """
        training_data = list(training_data)
        training_data_len = len(training_data)

        for epoch in range(epochs):
            random.shuffle(training_data)
            for i in range(0, training_data_len, mini_batch_size):
                self.update_mini_batch(training_data[i:i + mini_batch_size], eta)
            if test_data is not None:
                print(f"Epoch {epoch}: {self.evaluate(test_data)} / {len(test_data)}")
            else:
                print(f"Epoch {epoch} finished")


    def update_mini_batch(self, batch, eta):
        """One gradient descent step on a mini-batch of (x, y) pairs.

        Sums the backprop gradients of all pairs into nabla_b / nabla_w
        (array + array per layer, never list + list), then moves every
        weight and bias against the average: w -> w - eta/len(batch) * nabla.
        New lists are assigned to self.weights / self.biases.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        factor = eta / len(batch)

        for x, y in batch:
            delta_b, delta_w = self.backprop(x, y)

            nabla_b = [acc+d for acc, d in zip(nabla_b, delta_b)]
            nabla_w = [acc+d for acc, d in zip(nabla_w, delta_w)]

        self.weights = [w - factor * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - factor * nb for b, nb in zip(self.biases, nabla_b)]


    def backprop(self, x, y):
        """Return (nabla_b, nabla_w): the gradient of C_x for one pair (x, y).

        x is a (sizes[0], 1) input column, y a one-hot (sizes[-1], 1) column;
        update_mini_batch calls this once per pair. The forward pass stores
        every z and activation; the backward pass goes from the last layer to
        the first with BP1-BP4 (Nielsen, ch. 2). Both lists have the same
        shapes as self.biases / self.weights. Weights are not changed here.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activations, zs = [x], []

        # forward pass
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activations[-1]) + b
            activations.append(sigmoid(z))
            zs.append(z)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_derivative(zs[-1]) # BP1
        nabla_b[-1] = delta * 1 # BP3
        nabla_w[-1] = np.dot(delta, activations[-2].T) # BP4

        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l+1].T, delta) * sigmoid_derivative(zs[-l]) # BP2
            nabla_b[-l] = delta * 1 # BP3
            nabla_w[-l] = np.dot(delta, activations[-l-1].T) # BP4

        return (nabla_b, nabla_w)


def sigmoid(z):
    """Elementwise 1 / (1 + e^-z); works on numbers and numpy arrays."""
    return 1/(1 + np.exp(-z))

def sigmoid_derivative(z):
    """Elementwise sigma'(z) = sigma(z) * (1 - sigma(z)); used by backprop."""
    return sigmoid(z)*(1 - sigmoid(z))

