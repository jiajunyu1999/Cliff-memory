from __future__ import annotations

import math

import numpy as np


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(np.mean(np.square(y_true - y_pred))))


def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, cliff_mask: np.ndarray
) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    cliff_mask = np.asarray(cliff_mask, dtype=bool).reshape(-1)
    if not (len(y_true) == len(y_pred) == len(cliff_mask)):
        raise ValueError("metric inputs have different lengths")
    if not cliff_mask.any() or cliff_mask.all():
        raise ValueError("both cliff and non-cliff test molecules are required")
    return {
        "n_test": int(len(y_true)),
        "n_cliff": int(cliff_mask.sum()),
        "rmse": _rmse(y_true, y_pred),
        "cliff_rmse": _rmse(y_true[cliff_mask], y_pred[cliff_mask]),
        "noncliff_rmse": _rmse(y_true[~cliff_mask], y_pred[~cliff_mask]),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "cliff_mae": float(np.mean(np.abs(y_true[cliff_mask] - y_pred[cliff_mask]))),
    }

