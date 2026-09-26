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

            nabla_w = [acc+d for acc, d in zip(nabla_w, delta_w)]
            nabla_b = [acc+d for acc, d in zip(nabla_b, delta_b)]

        self.weights = [w - factor * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - factor * nb for b, nb in zip(self.biases, nabla_b)]



    # COPIED from Nielsen, "Neural Networks and Deep Learning", ch. 1
    # (github.com/mnielsen/neural-networks-and-deep-learning, src/network.py).
    # Black box for now: to be deleted and rewritten from scratch after ch. 2.
    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)


def sigmoid(z):
    """Elementwise 1 / (1 + e^-z); works on numbers and numpy arrays."""
    return 1/(1 + np.exp(-z))

# COPIED from Nielsen, ch. 1 (src/network.py), together with backprop.
def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1 - sigmoid(z))

