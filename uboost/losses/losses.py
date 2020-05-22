import typing

import numpy as np


class Loss:
    name = "BaseLoss"

    def __call__(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return self.eval(y, p)

    def eval(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def gradient(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def hessian_diagonal(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def hessian(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    pass


class MSELoss(Loss):
    name = "mse"

    def eval(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return (p - y) ** 2

    def gradient(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return 2 * (p - y)

    def hessian_diagonal(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return 2 + 0 * y

    pass


class BinaryCrossEntropy(Loss):
    name = "binary_cross_entropy"
    epsilon = 1e-8

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        pr = 1. / (1. + np.exp(-z))
        pr = np.clip(pr, self.epsilon, 1 - self.epsilon)
        return pr

    def eval(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        pr = self.sigmoid(p)
        return -y * np.log(self.epsilon + pr) - (1 - y) * np.log(self.epsilon + 1 - pr)

    def gradient(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        pr = self.sigmoid(p)
        return pr - y

    def hessian_diagonal(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        pr = self.sigmoid(p)
        return pr * (1 - pr)

    pass


class SoftmaxCrossEntropy(Loss):
    name = "softmax_cross_entropy"

    def softmax(self, z: np.ndarray) -> np.ndarray:
        e = np.exp(z - z.max(axis=-1, keepdims=True))
        pr = e / e.sum(axis=-1, keepdims=True)
        return pr

    def eval(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        pr = self.softmax(p)
        true_pr = np.sum(y * pr, axis=-1, keepdims=True)
        return -np.log(true_pr)

    def gradient(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        pr = self.softmax(p)
        return pr - y

    def hessian_diagonal(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        pr = self.softmax(p)
        return pr * (1 - pr)

    def hessian(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        pr = self.softmax(p)
        m = pr.shape[-1]
        eye = np.eye(m).reshape((pr.ndim - 1) * (1,) + (m, m))
        h = np.expand_dims(pr, axis=-1) * (eye - np.expand_dims(pr, axis=-2))
        return h

    pass


class L2RegularizedLoss(Loss):

    def __init__(self, base_loss: Loss, alpha=0.01):
        self.base_loss: Loss = base_loss
        self.alpha: float = alpha
        pass

    def eval(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return self.base_loss.eval(y, p) + self.alpha * p ** 2

    def gradient(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return self.base_loss.gradient(y, p) + self.alpha * 2.0 * p

    def hessian_diagonal(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return self.base_loss.hessian_diagonal(y, p) + self.alpha * 2.0

    def hessian(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return self.base_loss.hessian(y, p) + self.alpha * 2.0

    pass


class L1RegularizedLoss(Loss):

    def __init__(self, base_loss: Loss, alpha=0.01):
        self.base_loss: Loss = base_loss
        self.alpha: float = alpha
        pass

    def eval(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return self.base_loss.eval(y, p) + self.alpha * np.abs(p)

    def gradient(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return self.base_loss.gradient(y, p) + self.alpha * np.sign(p)

    def hessian_diagonal(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return self.base_loss.hessian_diagonal(y, p)

    def hessian(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return self.base_loss.hessian(y, p)

    pass


_losses = {
    'mse': MSELoss,
    'binary_cross_entropy': lambda: L2RegularizedLoss(BinaryCrossEntropy(), alpha=1e-6),
    'softmax_cross_entropy': SoftmaxCrossEntropy
}


def get_loss(name):
    return _losses[name]()


def get_losses_list():
    return _losses.keys()
