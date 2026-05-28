from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score


MetricFn = Callable[[object, object], float]


def roc_auc(y_true: object, y_pred: object) -> float:
    return float(roc_auc_score(y_true, y_pred))


def rmse(y_true: object, y_pred: object) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


METRICS: dict[str, MetricFn] = {
    "roc_auc": roc_auc,
    "rmse": rmse,
}


def get_metric(name: str) -> MetricFn:
    try:
        return METRICS[name]
    except KeyError as exc:
        known = ", ".join(sorted(METRICS))
        raise ValueError(f"Unknown metric {name!r}. Known metrics: {known}") from exc
