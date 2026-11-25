"""External clustering scores: accuracy under the best label matching, NMI and ARI."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def contingency(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    _, yt = np.unique(y_true, return_inverse=True)
    _, yp = np.unique(y_pred, return_inverse=True)
    m = np.zeros((yp.max() + 1, yt.max() + 1), dtype=np.int64)
    np.add.at(m, (yp, yt), 1)
    return m


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of points whose cluster maps to their class under the best one-to-one matching."""
    m = contingency(y_true, y_pred)
    r, c = linear_sum_assignment(-m)
    return float(m[r, c].sum() / len(y_true))


def scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "acc": accuracy(y_true, y_pred),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
    }
