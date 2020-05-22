import numpy as np
from random import sample
import typing


class BaseTransformer:
    input_dim: typing.Tuple
    output_dim: typing

    def fit(self, x: np.ndarray):
        self.input_dim = x.shape[1:]
        self.output_dim = self.input_dim
        pass

    def transform(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    pass


class IdentityTransformer(BaseTransformer):

    def transform(self, x: np.ndarray) -> np.ndarray:
        return x

    pass


class ColumnSelectorTransformer(BaseTransformer):
    columns: list

    def __init__(self, n=1.0):
        self.n = n
        pass

    def fit(self, x: np.ndarray):
        num = x.shape[-1]
        n = self.n
        if isinstance(self.n, float):
            if n <= 1:
                n = int(num * n)
            else:
                n = int(n)
                pass
            pass
        n = max(1, n)
        columns = sample(range(num), n)
        columns.sort()
        self.columns = columns
        self.input_dim = x.shape[1:]
        self.output_dim = self.input_dim[:-1] + (n,)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return x[..., self.columns]

    pass
