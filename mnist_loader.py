"""Load MNIST and convert it to (x, y) pairs for the network."""
import gzip, pickle
import numpy as np

class MNISTLoader:
    """Reads the MNIST pickle and prepares training/validation/test splits."""

    def __init__(self, path='data/mnist.pkl.gz'):
        """Store the path; the file is not read until load_data() is called."""
        self.path = path
        self.data = None
        self.prepared_data = None

    def load_data(self):
        """Read the gzipped pickle and return the raw (train, valid, test) tuple.

        Each split is (images, labels): images (N, 784) float32, labels (N,) int.
        """
        with gzip.open(self.path, 'rb') as f:
            self.data = pickle.load(f, encoding='latin1')
        return self.data

    def prepare_data(self):
        """Return (training, validation, test) as lists of (x, y) pairs.

        x is a (784, 1) column. y is a one-hot (10, 1) column for training,
        and a plain int label for validation and test. Loads data if needed.
        """
        if self.data is None:
            self.load_data()

        res = []
        for i, raw in enumerate(self.data):
            pairs = []
            for x, y in zip(raw[0], raw[1]):
                if i == 0:
                    y = one_hot(y)
                pairs.append((x.reshape(784, 1), y))
            res.append(pairs)

        self.prepared_data = tuple(res)
        return self.prepared_data

def one_hot(y):
    """Return a (10, 1) column with 1.0 at index y and 0.0 elsewhere."""
    res = np.zeros((10, 1))
    res[y][0] = 1.0
    return res

