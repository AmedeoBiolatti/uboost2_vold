import numpy as np
import typing
import random


class Sampler:
    _indexes: typing.List[int]

    def fit(self, x: np.ndarray, y=None):
        self._indexes = list(range(x.shape[0]))
        pass

    def sample(self, x: np.ndarray = None, y: np.ndarray = None) -> typing.List[int]:
        return self._indexes

    pass


class RandomSampler(Sampler):

    def __init__(self, sub_sample: typing.Union[float, int] = 1.0):
        self.sub_sample: typing.Union[float, int] = sub_sample
        pass

    def sample(self, x: np.ndarray = None, y: np.ndarray = None) -> typing.List[int]:
        indexes = self._indexes if x is None else list(range(x.shape[0]))
        n = self.sub_sample
        assert n > 0.0
        if n > 1.0:
            n = int(n)
        else:
            n = int(n * len(indexes))
            pass
        n = max(1, n)
        indexes = random.sample(indexes, n)
        return indexes

    pass
