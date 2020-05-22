import numpy as np
import typing
import tqdm

from uboost import metrics, transformers


class BaseLearner:
    input_dim: typing.Tuple
    output_dim: typing.Tuple

    def fit(self, x: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    pass


class BaseClassifier(BaseLearner):
    link_function: typing.Callable[[np.ndarray], np.ndarray]

    def predict_proba(self, x):
        return self.link_function(self.predict(x))

    pass


class IterativeLearner(BaseLearner):
    current_iteration: int = -1
    history: typing.Dict[str, np.ndarray]

    pass


class BaseBooster(IterativeLearner):
    """
    Genral boosting algorithm than train learner iteratively
    """
    # basic
    objective: str = "mse"
    n_learners: int = 10
    base_learner: typing.Callable

    learners: typing.Dict[int, BaseLearner]
    epsilons: typing.Dict[int, float]
    base_score: float = 0.0
    # internals
    x: np.ndarray
    y: np.ndarray
    x_transformer: typing.Dict[int, transformers.BaseTransformer]

    def __init__(self):
        self.learners = dict()
        self.epsilons = dict()
        self.x_transformer = dict()
        self.history = dict()
        self.callbacks = list()
        pass

    def _build_z(self, x: np.ndarray, k: int) -> np.ndarray:
        return x

    def _predict_kth(self, x: np.ndarray, k: int):
        prediction_dim = (x.shape[0],) + self.output_dim
        z = self._build_z(x, k)
        return self.epsilons[k] * self.learners[k].predict(z).reshape(prediction_dim)

    def predict_score(self, x: np.ndarray) -> np.ndarray:
        prediction_dim = (x.shape[0],) + self.output_dim
        prediction = self.base_score + np.zeros(prediction_dim)
        for k in self.learners.keys():
            prediction += self._predict_kth(x, k)
            pass
        return prediction

    def _eval(self, eval_set=None, eval_metric=None):
        train_set = [("", self.x, self.y)]
        eval_set = train_set if eval_set is None else (train_set + eval_set)
        train_metric = [("loss", metrics.get_metric(self.objective))]
        eval_metric = train_metric if eval_metric is None else (train_metric + eval_metric)
        if eval_set is not None:
            i = 0
            for s in eval_set:
                if len(s) == 2:
                    x, y = s
                    set_name = "val_%d" % i if i > 0 else "val"
                    i += 1
                else:
                    set_name, x, y = s
                    pass
                p = self.predict_score(x)
                for metric in eval_metric:
                    if isinstance(metric, tuple):
                        metric_name, metric = metric
                    elif isinstance(metric, str):
                        metric_name = metric
                        metric = metrics.get_metric(metric_name)
                    else:
                        raise NotImplementedError
                    metric_value = np.array(metric(y, p)).mean().reshape(1)
                    name = "%s-%s" % (set_name, metric_name) if len(set_name) > 0 else metric_name
                    if name in self.history:
                        self.history[name] = np.append(self.history[name], metric_value)
                    else:
                        self.history[name] = metric_value
                    pass
                pass
            pass
        pass

    def _init_booster(self, x, y):
        self.x = x
        self.y = y
        self.input_dim = x.shape[1:]
        self.output_dim = y.shape[1:]
        self.learners = dict()
        self.epsilons = dict()
        self.x_transformer = dict()
        self.history = dict()
        pass

    def _fit_learner(self):
        raise NotImplementedError

    def fit(self, x: np.ndarray, y: np.ndarray, eval_set=None, eval_metric=None, progress_bar=True, *args, **kwargs):
        self._init_booster(x, y)
        iter_range = range(self.current_iteration + 1, self.n_learners)
        if progress_bar:
            iter_range = tqdm.tqdm(iter_range)
            pass
        for self.current_iteration in iter_range:
            self._fit_learner()
            self._eval(eval_set, eval_metric)
            pass
        return self.history

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_score(x)

    pass
