import numpy as np
import sklearn.metrics


def f1_score(y_true: np.ndarray,
             y_pred: np.ndarray,
             threshold: float = 0.5,
             average: str = 'samples',
             sample_weight=None):
    return sklearn.metrics.f1_score(y_true,
                                    np.where(y_pred > threshold, 1., 0.),
                                    average=average,
                                    sample_weight=sample_weight)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray):
    correct = (y_true == y_pred.argmax(axis=1)).sum()
    return 100 * correct / len(y_true)
