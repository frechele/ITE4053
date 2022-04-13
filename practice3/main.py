from collections import namedtuple
import numpy as np
from typing import Tuple, List


Dataset = namedtuple('Dataset', ['x', 'y'])


# Hyper-parameters
K = 5000
K_print = 500


def binary_cross_entropy(input: np.ndarray, target: np.ndarray, eps: float = 1e-10) -> float:
    return -np.mean(target * np.log(input + eps) + (1 - target) * np.log(1 - input + eps))

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1. / (1 + np.exp(-x))

def sigmoid_prime(z: np.ndarray) -> np.ndarray:
    return z * (1 - z)

W1 = np.random.randn(1, 1)
b1 = np.random.randn(1, 1)
W2 = np.random.randn(1, 1)
b2 = np.random.randn(1, 1)

def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.dot(weight, x) + bias

def generate_samples(m: int):
    x = np.zeros(shape=(1, m))
    y = np.zeros(shape=(1, m))

    for i in range(m):
        x[:, i] = np.random.uniform(0, 360, size=(1,))
        y[:, i] = 1 if np.sin(np.radians(x[:, i])) > 0 else 0

    return Dataset(x, y.astype(np.int32))


def model(input: np.ndarray, target: np.ndarray, alpha: float, train: bool) -> Tuple[float, float]:
    global W1, W2, b1, b2

    # input normalize
    input = (input - 180) / 360

    Z1 = linear(input, W1, b1)
    A1 = sigmoid(Z1)
    Z2 = linear(A1, W2, b2)
    A2 = sigmoid(Z2)

    batch_size = input.shape[1]

    cost = binary_cross_entropy(A2, target)
    acc = np.mean((A2 > 0.5) == target)

    if train:
        dZ2 = A2 - target
        dW2 = np.dot(dZ2, A1.T) / batch_size
        db2 = np.sum(dZ2, axis=1, keepdims=True) / batch_size

        dZ1 = np.dot(W2.T, dZ2) * (A1 * (1 - A1))
        dW1 = np.dot(dZ1, input.T) / batch_size
        db1 = np.sum(dZ1, axis=1, keepdims=True) / batch_size

        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * db1
        W2 = W2 - alpha * dW2
        b2 = b2 - alpha * db2

    return cost, acc


def run_experiment(lr: float, train_dataset: Dataset, test_dataset: Dataset, f):
    global W1, b1, W2, b2

    np.random.seed(12345)

    W1 = np.random.normal(size=(1, 1)) * np.sqrt(2.)
    b1 = np.zeros((1, 1))
    W2 = np.random.normal(size=(1, 1)) * np.sqrt(2.)
    b2 = np.zeros((1, 1))

    for k in range(1, 1+K):
        train_cost, train_acc = model(train_dataset.x, train_dataset.y, lr, True)
        test_cost, test_acc = model(test_dataset.x, test_dataset.y, lr, False)

        f.write(f'{k}/{K} iter - train_cost {train_cost} train_acc {train_acc * 100} test_cost {test_cost} test_acc {test_acc * 100} ')

        if k % K_print == 0 or k == 1:
            f.write('W1 {} b1 {} '.format(W1, b1))
            f.write('W2 {} b2 {}'.format(W2, b2))

        f.write('\n')


if __name__ == '__main__':
    train_dataset = generate_samples(10000)
    test_dataset = generate_samples(1000)

    np.save('dataset.npy', test_dataset.x)

    # lr_candidates = [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 1e-7, 1e-8]
    lr_candidates = [1.0]

    for lr in lr_candidates:
        with open('result_lr_{}'.format(lr), 'wt') as f:
            print('run experiment {}...'.format(lr), end='', flush=True)
            run_experiment(lr, train_dataset, test_dataset, f)
            print('DONE')
    
