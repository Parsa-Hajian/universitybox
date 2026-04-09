"""
universitybox.forecast
======================

Time-series forecasting tools.

Quick start
-----------
    from universitybox.forecast import DNA

    model = DNA(period=4)
    model.fit(y)
    fc = model.forecast(h=4)
    lower, upper = model.predict_interval(h=4, level=0.95)

Classes
-------
DNA         — Dynamic Nonlinear Adaptive forecaster (flagship model)

Submodules
----------
_decomposition  — D-stage: Henderson filter + Fourier decomposition
_nonlinear      — N-stage: RBF / Ridge regression on residuals
_adaptive       — A-stage: Local Linear Trend Kalman filter
_metrics        — MAE, RMSE, MAPE, sMAPE, MASE, CRPS
_base           — Abstract BaseForecaster interface
"""

from .dna import DNA
from . import _metrics as metrics
from ._base import BaseForecaster

__all__ = ["DNA", "metrics", "BaseForecaster"]
