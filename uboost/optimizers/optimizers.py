import numpy as np
import typing

from uboost import losses


class LossOptimizer:
    loss: losses.Loss
    y: np.ndarray

    def init(self, loss: losses.Loss, y: np.ndarray):
        self.loss = loss
        self.y = y
        pass

    def step(self, p: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    pass


class GradientDescentOptimizer(LossOptimizer):

    def step(self, p):
        return -self.loss.gradient(self.y, p)

    pass


class NewtonOptimizer(LossOptimizer):

    def step(self, p):
        return -self.loss.gradient(self.y, p) / self.loss.hessian_diagonal(self.y, p)

    pass


class MomentumOptimizer(LossOptimizer):
    base_optimizer: LossOptimizer
    momentum: float
    velocity: np.ndarray

    def __init__(self, base_optimizer: LossOptimizer = GradientDescentOptimizer(), momentum: float = 0.5):
        self.base_optimizer = base_optimizer
        self.momentum = momentum
        pass

    def init(self, loss: losses.Loss, y: np.ndarray):
        super(MomentumOptimizer, self).init(loss, y)
        self.base_optimizer.init(loss, y)
        self.velocity = 0.0 * y
        pass

    def step(self, p):
        g = self.base_optimizer.step(p)
        v_new = g + self.momentum * self.velocity
        self.velocity = v_new
        return v_new

    pass


class NesterovOptimizer(LossOptimizer):
    base_optimizer: LossOptimizer
    learning_rate: float
    momentum: float
    velocity: np.ndarray

    def __init__(self, base_optimizer: LossOptimizer = GradientDescentOptimizer(), learning_rate: float = 0.3,
                 momentum: float = 0.5):
        self.base_optimizer = base_optimizer
        self.momentum = momentum
        self.learning_rate = learning_rate
        pass

    def init(self, loss: losses.Loss, y: np.ndarray):
        super(NesterovOptimizer, self).init(loss, y)
        self.base_optimizer.init(loss, y)
        self.velocity = 0.0 * y
        pass

    def step(self, p):
        g = self.base_optimizer.step(p + self.learning_rate * self.momentum * self.velocity)
        v_new = g + self.momentum * self.velocity
        self.velocity = v_new
        return v_new

    pass


class MultiStepOptimizer(LossOptimizer):
    base_optimizer: LossOptimizer
    learning_rate: float
    n_steps: int

    def __init__(self, base_optimizer: LossOptimizer = GradientDescentOptimizer(),
                 n_steps: int = 2,
                 learning_rate: typing.Union[float, None] = None):
        self.base_optimizer = base_optimizer
        self.learning_rate = 1.0 / n_steps if learning_rate is None else learning_rate
        self.n_steps = n_steps
        pass

    def init(self, loss: losses.Loss, y: np.ndarray):
        super(MultiStepOptimizer, self).init(loss, y)
        self.base_optimizer.init(loss, y)
        pass

    def step(self, p):
        p_new = p.copy()
        for _ in range(self.n_steps):
            p_new += self.learning_rate * self.base_optimizer.step(p_new)
            pass
        v_total = p_new - p
        return v_total

    pass
