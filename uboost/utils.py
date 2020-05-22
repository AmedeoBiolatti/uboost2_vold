import typing
import sklearn.base
import types
import sys


def init_learner(learner_builder: typing.Union[sklearn.base.RegressorMixin], *args, **kwargs):
    learner = None
    if isinstance(learner_builder, sklearn.base.RegressorMixin):
        learner = sklearn.base.clone(learner_builder)
    elif isinstance(learner_builder, types.LambdaType):
        learner = learner_builder(*args, **kwargs)
    else:
        if "tensorflow" in sys.modules:
            from tensorflow.keras import models
            if isinstance(learner_builder, models.Model):
                learner = models.clone_model(learner_builder)
                learner.compile(loss=learner_builder.loss, optimizer=learner_builder.optimizer,
                                metrics=learner_builder.metrics)
                pass
        pass
    if learner is None:
        raise TypeError
    return learner
