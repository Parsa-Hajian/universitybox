"""
Abstract base class for all UniversityBox forecasters.
"""
from __future__ import annotations

import abc
import numpy as np
from typing import Optional, Tuple


class BaseForecaster(abc.ABC):
    """
    Minimal sklearn-compatible interface for all UBox forecasters.

    Every concrete forecaster must implement:
        fit(y, ...)  -> self
        forecast(h)  -> np.ndarray

    Optionally implement:
        predict_interval(h, level) -> (lower, upper)
        score(y_true, y_pred)      -> float
    """

    def __repr__(self) -> str:
        params = ", ".join(
            f"{k}={v!r}" for k, v in self.get_params().items()
        )
        return f"{self.__class__.__name__}({params})"

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def fit(self, y: np.ndarray, **kwargs) -> "BaseForecaster":
        """Fit the model to time series y (shape: [n_timepoints])."""

    @abc.abstractmethod
    def forecast(self, h: int) -> np.ndarray:
        """Return point forecasts for horizons 1 … h."""

    # ------------------------------------------------------------------
    # Default implementations (may be overridden)
    # ------------------------------------------------------------------

    def predict_interval(
        self, h: int, level: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (lower, upper) prediction interval arrays of length h.
        Default: ±1.96 * in-sample RMSE (Gaussian, homoscedastic).
        Override for richer uncertainty quantification.
        """
        if not hasattr(self, "_insample_rmse"):
            raise RuntimeError("Call fit() before predict_interval().")
        from scipy.stats import norm
        z = norm.ppf(0.5 + level / 2)
        fc = self.forecast(h)
        margin = z * self._insample_rmse * np.sqrt(np.arange(1, h + 1))
        return fc - margin, fc + margin

    def score(
        self, y_true: np.ndarray, y_pred: Optional[np.ndarray] = None
    ) -> float:
        """Return MASE (Mean Absolute Scaled Error) relative to naïve."""
        y_true = np.asarray(y_true, dtype=float)
        if y_pred is None:
            y_pred = self.forecast(len(y_true))
        naive_mae = np.mean(np.abs(np.diff(y_true)))
        if naive_mae == 0:
            naive_mae = 1e-8
        return float(np.mean(np.abs(y_true - y_pred)) / naive_mae)

    def get_params(self) -> dict:
        """Return constructor parameters (mirrors sklearn convention)."""
        import inspect
        sig = inspect.signature(self.__init__)
        return {
            k: getattr(self, k, v.default)
            for k, v in sig.parameters.items()
            if v.default is not inspect.Parameter.empty
        }

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_y(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D, got shape {y.shape}.")
        if len(y) < 4:
            raise ValueError("y must have at least 4 observations.")
        if not np.all(np.isfinite(y)):
            raise ValueError("y contains NaN or Inf values.")
        return y
