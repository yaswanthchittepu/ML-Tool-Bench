"""Official competition metrics. Signature: ``metric(predictions, labels) -> float``."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score


def rmse(preds, labels) -> float:
    return float(np.sqrt(np.mean((preds - labels) ** 2)))


def rmsle(preds, labels) -> float:
    """RMSLE on non-negative predictions (the grader clips negatives to 0 first)."""
    preds, labels = np.asarray(preds, dtype=float), np.asarray(labels, dtype=float)
    return float(np.sqrt(np.mean((np.log1p(preds) - np.log1p(labels)) ** 2)))


def rmsle_paper(preds, labels) -> float:
    """The paper's RMSLE: on pandas Series, rows with a prediction below -1 become NaN and are
    silently skipped by the mean, and a prediction of exactly -1 gives inf."""
    return float(np.sqrt(np.mean((np.log(preds + 1) - np.log(labels + 1)) ** 2)))


def accuracy(preds, labels) -> float:
    return float((preds == labels).mean())


def auc(preds, labels) -> float:
    return float(roc_auc_score(labels, preds))


def mcc(preds, labels) -> float:
    return float(matthews_corrcoef(labels, preds))


def macro_f1(preds, labels) -> float:
    return float(f1_score(labels, preds, average="macro"))


def normalized_gini(preds, labels) -> float:
    """Porto Seguro's normalized Gini coefficient."""
    def gini(actual, pred):
        table = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=float)
        table = table[np.lexsort((table[:, 2], -1 * table[:, 1]))]
        cumulative = table[:, 0].cumsum().sum() / table[:, 0].sum()
        return (cumulative - (len(actual) + 1) / 2.0) / len(actual)

    return float(gini(labels, preds) / gini(labels, labels))


METRICS: dict[str, Callable] = {
    "rmse": rmse, "rmsle": rmsle, "accuracy": accuracy, "auc": auc,
    "mcc": mcc, "macro_f1": macro_f1, "normalized_gini": normalized_gini,
}
LOWER_IS_BETTER = {"rmse", "rmsle"}
