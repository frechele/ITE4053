from collections import namedtuple
import numpy as np
from typing import Tuple
import pickle


Dataset = namedtuple('Dataset', ['x', 'y'])


# Hyper-parameters
K = 5000
K_print = 500


def binary_cross_entropy(input: np.ndarray, target: np.ndarray, eps: float = 1e-10) -> float:
    return -np.mean(target * np.log(input + eps) + (1 - target) * np.log(1 - input + eps))


class LogisticRegression:
    def __init__(self, in_features: int):
        self.weight = np.random.randn(in_features)
        self.bias = np.random.random()

    def forward(self, x: np.ndarray) -> float:
        z = np.dot(self.weight.T, x) + self.bias

        return 1. / (1 + np.exp(-z))

    def train(self, input: np.ndarray, target: np.ndarray, alpha: float) -> Tuple[float, float]:
        preds = self.forward(input.T)
        eps = 1e-10

        cost = binary_cross_entropy(preds, target)
        acc = np.mean((preds > 0.5) == target)

        dZ = preds - target
        dw = input * dZ.reshape((-1, 1))
        db = dZ

        self.weight -= alpha * np.mean(dw, axis=0)
        self.bias -= alpha * np.mean(db, axis=0)

        return cost, acc

    def test(self, input: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
        preds = self.forward(input.T)
        eps = 1e-10

        cost = binary_cross_entropy(preds, target)
        acc = np.mean((preds > 0.5) == target)

        return cost, acc


def generate_samples(m: int):
    x = np.zeros(shape=(m, 1))
    y = np.zeros(shape=(m,))

    for i in range(m):
        x[i] = np.random.uniform(0, 360, size=(1,))
        y[i] = 1 if np.sin(np.radians(x[i])) > 0 else 0

    return Dataset(x, y)


def run_experiment(lr: float, train_dataset: Dataset, test_dataset: Dataset, f):
    np.random.seed(12345)

    model = LogisticRegression(1)

    for k in range(1, 1+K):
        train_cost, train_acc = model.train(train_dataset.x, train_dataset.y, lr)
        test_cost, test_acc = model.test(test_dataset.x, test_dataset.y)

        f.write(f'{k}/{K} iter - train_cost {train_cost} train_acc {train_acc * 100} test_cost {test_cost} test_acc {test_acc * 100} ')

        if k % K_print == 0 or k == 1:
            f.write('W {} '.format(model.weight))
            f.write('b {}'.format(model.bias))

        f.write('\n')


if __name__ == '__main__':
    train_dataset = generate_samples(1000)
    test_dataset = generate_samples(1000)

    # lr_candidates = [20.0, 15.0, 10.0, 8.0, 4.0, 2.0, 1.0, 0.1, 0.01, 0.001, 0.0001]
    # lr_candidates = [0.00001, 0.000001, 1e-7, 1e-8]
    lr_candidates = [0.0001]

    for lr in lr_candidates:
        with open('result_lr_{}'.format(lr), 'wt') as f:
            print('run experiment {}...'.format(lr), end='', flush=True)
            run_experiment(lr, train_dataset, test_dataset, f)
            print('DONE')
    