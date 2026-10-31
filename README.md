# A neural network from scratch — following Nielsen

A learning project: a fully connected neural network in plain numpy that learns
to recognize MNIST handwritten digits. Written by hand while working through
Michael Nielsen's
[*Neural Networks and Deep Learning*](http://neuralnetworksanddeeplearning.com)
— no PyTorch, no ready-made layers, so that every line is understood.

**Goal (milestone 1):** own `network.py`, **≥ 97%** on the MNIST test set,
backpropagation derived on paper and implemented independently.

## Status

| Chapter | What | Status |
|---|---|---|
| 1 | loader, network, `feedforward`, `evaluate`, `SGD`, `update_mini_batch` | ✅ done, **95.1%** |
| 2 | derive BP1–BP4, write own `backprop` | ✅ done, **95.35%** |
| 3 | cross-entropy cost, regularization, weight initialization | ⏳ target ≥ 97% |

## Results

All runs: `[784, 30, 10]`, 30 epochs, mini-batch 10, η = 3.0, accuracy on the
10,000 test images after every epoch.

| Date | `backprop` | Epoch 1 | Best | Last |
|---|---|---|---|---|
| 2026-09-26 | copied from the book | 90.94% | 95.10% (epoch 25) | 94.89% |
| 2026-10-31 | own, run 1 | 82.05% | **95.35%** (epoch 28) | 94.95% |
| 2026-10-31 | own, run 2 | 81.51% | 94.26% (epoch 27) | 94.19% |
| 2026-10-31 | own, run 3 | 90.39% | 94.93% (epoch 27) | 94.88% |

Runs differ by about one percentage point with the same code: only the random
initial weights and the shuffling change. So own vs copied `backprop` is a tie,
not a win. The own `backprop` also passes a gradient check against finite
differences (max difference 2.3e-11 on a `[5, 4, 3, 2]` network). Two of three
runs started slowly (~82% after epoch 1) — likely saturated sigmoid neurons,
which chapter 3 addresses. "Best" is the best of 30 test-set evaluations, so it
is optimistic; "Last" is the fairer number.

For reference: an untrained network scores 7–10% (random guessing);
the book reports 95.42% for the same setup (best of three runs).

## Layout

| File | Contents |
|---|---|
| `mnist_loader.py` | `MNISTLoader`: reads `data/mnist.pkl.gz` and returns three lists of `(x, y)` pairs; `one_hot` |
| `network.py` | `Network`: weights and biases, forward pass, accuracy, training with stochastic gradient descent, `backprop` (BP1–BP4) |
| `data/` | put `mnist.pkl.gz` here (ignored by git) |

Data format — the contract between the loader and the network:

| Split | Pairs | `x` | `y` |
|---|---|---|---|
| `training` | 50,000 | column `(784, 1)` | one-hot column `(10, 1)` |
| `validation` | 10,000 | column `(784, 1)` | int 0–9 |
| `test` | 10,000 | column `(784, 1)` | int 0–9 |

## Setup

Python 3.13, numpy.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy

mkdir -p data
curl -L -o data/mnist.pkl.gz \
  https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz
```

## Usage

From the project root, with `.venv` active:

```python
from mnist_loader import MNISTLoader
import network

training, validation, test = MNISTLoader().prepare_data()
net = network.Network([784, 30, 10])
net.SGD(training, 30, 10, 3.0, test)   # epochs, mini-batch size, eta, eval data
```

## References

- Michael A. Nielsen, *Neural Networks and Deep Learning*, Determination Press, 2015.
- MNIST (LeCun, Cortes, Burges), in the format from the
  [book's repository](https://github.com/mnielsen/neural-networks-and-deep-learning).
