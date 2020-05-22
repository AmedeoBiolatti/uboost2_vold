import numpy as np
import typing
from sklearn import tree, base

from uboost import core, losses, metrics, transformers, optimizers, samplers, utils


class GradientBooster(core.BaseBooster):
    # basic parameters
    loss: losses.Loss
    loss_optimizer: optimizers.LossOptimizer
    learning_rate: float = 0.3
    base_learner: typing.Union[base.RegressorMixin, core.BaseLearner]

    # data managing parameters
    # col_sample: float = 1.0
    data_sampler: samplers.Sampler
    data_transformer_builder: typing.Callable = transformers.ColumnSelectorTransformer

    # advanced params
    learner_fit_params: typing.Dict
    data_transformer_builder_params: typing.Dict

    # model internals
    prediction: np.ndarray

    # methods
    def __init__(self):
        super(GradientBooster, self).__init__()
        self.learner_fit_params = dict()
        self.loss_optimizer = optimizers.NewtonOptimizer()
        self.base_learner = tree.DecisionTreeRegressor(max_depth=3)
        self.data_sampler = samplers.RandomSampler()
        self.data_transformer_builder = transformers.ColumnSelectorTransformer
        self.data_transformer_builder_params = dict()
        pass

    def _init_booster(self, x, y):
        super(GradientBooster, self)._init_booster(x, y)
        self.loss = losses.get_loss(self.objective)
        if self.objective in ["mse"]:
            self.base_score = self.y.mean()
        elif self.objective in ["binary_cross_entropy"]:
            prob = self.y.mean()
            self.base_score = np.log(prob / (1 - prob))
            pass
        self.prediction = np.zeros(x.shape[0:1] + self.output_dim) + self.base_score
        self.loss_optimizer.init(self.loss, self.y)
        self.data_sampler.fit(self.x, self.y)
        pass

    def _build_z(self, x: np.ndarray, k: int) -> np.ndarray:
        if k not in self.x_transformer.keys():
            self.x_transformer[k] = self.data_transformer_builder(**self.data_transformer_builder_params).fit(x)
            pass
        return self.x_transformer[k].transform(x)

    def _build_target(self, y: np.ndarray, p: np.ndarray, k: int) -> np.ndarray:
        return self.loss_optimizer.step(p)

    def _build_training_data(self, k: int):
        training_subset = self.data_sampler.sample()
        z = self._build_z(self.x, k)
        target = self._build_target(self.y, self.prediction, k)
        return z[training_subset], target[training_subset]

    def _fit_learner(self):
        z, target = self._build_training_data(self.current_iteration)
        learner = utils.init_learner(self.base_learner)  # self.base_learner(**self.learner_builder_params)
        learner.fit(z, target, **self.learner_fit_params)
        self.learners[self.current_iteration] = learner
        self.epsilons[self.current_iteration] = self.learning_rate
        self.prediction += self._predict_kth(self.x, self.current_iteration)
        pass

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_score(x)

    pass


