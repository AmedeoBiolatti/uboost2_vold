import numpy as np
from sklearn import metrics
import time

from uboost import losses

_metrics_numpy = {
    "mse": metrics.mean_squared_error,
    "rmse": lambda y, p: metrics.mean_squared_error(y, p) ** 0.5,
    "mae": metrics.mean_absolute_error,
    "auc": metrics.roc_auc_score,
    "time": time.time
}

for loss_name in losses.get_losses_list():
    _metrics_numpy[loss_name] = losses.get_loss(loss_name)
    pass


def get_metric(name):
    if name in _metrics_numpy.keys():
        metric = _metrics_numpy[name]
    else:
        raise NotImplementedError
    return metric
