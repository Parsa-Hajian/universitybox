"""
Forecast evaluation metrics.

All functions accept array-like y_true and y_pred of equal length.

Metrics implemented
-------------------
MAE    — Mean Absolute Error
RMSE   — Root Mean Squared Error
MAPE   — Mean Absolute Percentage Error   (undefined if y_true = 0)
sMAPE  — Symmetric MAPE
MASE   — Mean Absolute Scaled Error       (requires training series)
CRPS   — Continuous Ranked Probability Score for Gaussian intervals
"""
from __future__ import annotations

import numpy as np
from typing import Optional


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _check(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _check(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _check(y_true, y_pred)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    sMAPE = (2/n) Σ |y − ŷ| / (|y| + |ŷ|)   ∈ [0, 200%]
    """
    y_true, y_pred = _check(y_true, y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 0
    if not mask.any():
        return 0.0
    return float(np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: Optional[np.ndarray] = None,
    period: int = 1,
) -> float:
    """
    MASE = MAE / MAE_naïve

    Naïve benchmark: seasonal random walk  ŷ_t = y_{t-P}
    (reduces to random walk when period=1).

    Parameters
    ----------
    y_train : training series used to compute naïve MAE scale.
              If None, y_true itself is used (in-sample MASE).
    period  : seasonal period P for the naïve benchmark.
    """
    y_true, y_pred = _check(y_true, y_pred)
    if y_train is None:
        y_train = y_true
    y_train = np.asarray(y_train, dtype=float)
    scale = np.mean(np.abs(y_train[period:] - y_train[:-period]))
    if scale == 0:
        scale = 1e-8
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def crps_gaussian(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> float:
    """
    CRPS for Gaussian predictive distribution N(μ, σ²).

    Closed form (Gneiting & Raftery 2007):
        CRPS(N(μ,σ), y) = σ { (y-μ)/σ [2Φ((y-μ)/σ) − 1]
                              + 2φ((y-μ)/σ) − 1/√π }

    where Φ is the standard normal CDF and φ its PDF.

    Returns the mean CRPS over all forecast horizons.
    """
    from scipy.stats import norm

    y_true = np.asarray(y_true, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sigma = np.maximum(sigma, 1e-8)

    z = (y_true - mu) / sigma
    crps_i = sigma * (
        z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi)
    )
    return float(np.mean(crps_i))


def summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: Optional[np.ndarray] = None,
    period: int = 1,
) -> dict:
    """
    Return all metrics as a dict.
    """
    return {
        "MAE":   mae(y_true, y_pred),
        "RMSE":  rmse(y_true, y_pred),
        "MAPE":  mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "MASE":  mase(y_true, y_pred, y_train=y_train, period=period),
    }


def _check(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}.")
    return a, b
