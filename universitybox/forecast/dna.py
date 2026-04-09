"""
DNA — Dynamic Nonlinear Adaptive Forecaster
============================================

Usage
-----
    from universitybox.forecast import DNA

    model = DNA(period=4)
    model.fit(y_train)

    point_forecast      = model.forecast(h=4)
    lower, upper        = model.predict_interval(h=4, level=0.95)
    metrics             = model.evaluate(y_test)

Mathematical summary
--------------------
DNA is a three-stage hierarchical forecaster that decomposes the
observed time series y into progressively finer structure:

    Stage D (Decomposition):
        y_t = μ_t + s_t + ε_t
        μ_t via Henderson filter; s_t via Fourier OLS.

    Stage N (Nonlinear):
        ε_t ≈ f(Φ(x_t))
        f fitted by Ridge regression on a rich feature dictionary
        Φ = [polynomial | AR lags | RBF kernels].

    Stage A (Adaptive):
        r_t = ε_t − f̂(x_t)
        r_t modelled by a Local Linear Trend (LLT) Kalman filter.

    Ensemble combination:
        ŷ_{t+h} = α μ̂_{t+h} + β ŝ_{t+h} + γ f̂_{t+h} + δ â_{t+h}

        Weights [α,β,γ,δ] are computed by inverse-variance stacking
        on the in-sample fitted values of each stage.

    Prediction intervals:
        Analytical Gaussian:  ŷ ± z_{α/2} σ_h
        Bootstrap:            empirical quantiles from B resampled
                              residual trajectories.

See MATH.md for full derivations.

Parameters
----------
period : int or 'auto'
    Dominant seasonal period (e.g. 4 quarterly, 12 monthly, 7 daily).
    'auto' estimates P from the periodogram.
trend_window : int or 'auto'
    Henderson filter half-length m. 'auto' → max(3, period//2).
n_fourier : int
    Fourier harmonics K for the seasonal model. Default 3.
poly_degree : int
    Polynomial degree for N-stage feature map. Default 2.
n_lags : int
    AR lag count for N-stage feature map. Default 4.
n_rbf : int
    Number of RBF centres (k-means++ selected). Default 10.
rbf_gamma : float or 'auto'
    RBF bandwidth γ. 'auto' uses median heuristic.
ridge_alpha : float
    Ridge regularisation λ. Default 1e-3.
kalman_q_level : float
    Kalman level process noise q_l. Default 1e-4.
kalman_q_slope : float
    Kalman slope process noise q_b. Default 1e-6.
kalman_obs_var : float
    Kalman observation noise R. Default 1e-2.
kalman_mle : bool
    If True, estimate Kalman noise params by MLE. Default False.
ensemble : {'iv', 'equal', 'ols'}
    Combination method for component forecasts. Default 'iv'.
ci_method : {'analytical', 'bootstrap'}
    Prediction interval method. Default 'analytical'.
ci_bootstrap_n : int
    Bootstrap replications (ci_method='bootstrap'). Default 500.
random_state : int or None
    Seed for reproducibility.
"""
from __future__ import annotations

import warnings
import numpy as np
from typing import Optional, Tuple, Union

from ._base import BaseForecaster
from ._decomposition import (
    decompose,
    estimate_period,
    forecast_components,
)
from ._nonlinear import NonlinearFeatureMap, NonlinearStage
from ._adaptive import AdaptiveStage
from . import _metrics as metrics


# ──────────────────────────────────────────────────────────────────────
# DNA forecaster
# ──────────────────────────────────────────────────────────────────────

class DNA(BaseForecaster):
    """
    Dynamic Nonlinear Adaptive (DNA) time series forecaster.

    See module docstring for full mathematical formulation and parameters.
    """

    def __init__(
        self,
        period: Union[int, str] = "auto",
        trend_window: Union[int, str] = "auto",
        n_fourier: int = 3,
        poly_degree: int = 2,
        n_lags: int = 4,
        n_rbf: int = 10,
        rbf_gamma: Union[float, str] = "auto",
        ridge_alpha: float = 1e-3,
        kalman_q_level: float = 1e-4,
        kalman_q_slope: float = 1e-6,
        kalman_obs_var: float = 1e-2,
        kalman_mle: bool = False,
        ensemble: str = "iv",
        ci_method: str = "analytical",
        ci_bootstrap_n: int = 500,
        random_state: Optional[int] = None,
    ):
        self.period = period
        self.trend_window = trend_window
        self.n_fourier = n_fourier
        self.poly_degree = poly_degree
        self.n_lags = n_lags
        self.n_rbf = n_rbf
        self.rbf_gamma = rbf_gamma
        self.ridge_alpha = ridge_alpha
        self.kalman_q_level = kalman_q_level
        self.kalman_q_slope = kalman_q_slope
        self.kalman_obs_var = kalman_obs_var
        self.kalman_mle = kalman_mle
        self.ensemble = ensemble
        self.ci_method = ci_method
        self.ci_bootstrap_n = ci_bootstrap_n
        self.random_state = random_state

        self._fitted = False

    # ── fit ──────────────────────────────────────────────────────────

    def fit(self, y: np.ndarray, **kwargs) -> "DNA":
        """
        Fit the DNA model to time series y.

        Parameters
        ----------
        y : array-like, shape (n,)
            Observed univariate time series.

        Returns
        -------
        self
        """
        y = self._validate_y(np.asarray(y, dtype=float))
        self._y = y
        n = len(y)
        rng = np.random.default_rng(self.random_state)

        # ── Resolve auto-parameters ──────────────────────────────
        if self.period == "auto":
            self._period = estimate_period(y)
        else:
            self._period = int(self.period)

        if self.trend_window == "auto":
            m = max(3, (self._period // 2) * 2 + 1)
            self._trend_window = m
        else:
            self._trend_window = int(self.trend_window)

        # ── D-Stage: Decomposition ────────────────────────────────
        trend, seasonal, sea_coef, residual = decompose(
            y,
            period=self._period,
            trend_window=self._trend_window,
            n_fourier=self.n_fourier,
        )
        self._trend = trend
        self._seasonal = seasonal
        self._sea_coef = sea_coef

        # ── N-Stage: Nonlinear ────────────────────────────────────
        feat_map = NonlinearFeatureMap(
            poly_degree=self.poly_degree,
            n_lags=self.n_lags,
            n_rbf=min(self.n_rbf, n),
            rbf_gamma=self.rbf_gamma,
            random_state=self.random_state,
        )
        feat_map.fit(residual)
        self._feat_map = feat_map

        nl_stage = NonlinearStage(feat_map, ridge_alpha=self.ridge_alpha)
        nl_stage.fit(residual)
        self._nl_stage = nl_stage

        nl_insample = nl_stage.in_sample()
        r2 = residual - nl_insample  # second-order residual

        # ── A-Stage: Adaptive ─────────────────────────────────────
        a_stage = AdaptiveStage(
            q_level=self.kalman_q_level,
            q_slope=self.kalman_q_slope,
            obs_var=self.kalman_obs_var,
            mle=self.kalman_mle,
        )
        a_stage.fit(r2)
        self._a_stage = a_stage

        # ── Ensemble weights ──────────────────────────────────────
        components = {
            "trend":    trend,
            "seasonal": seasonal,
            "nonlin":   nl_insample,
            "adaptive": a_stage.in_sample(),
        }
        self._weights = self._compute_weights(y, components)

        # ── In-sample fitted values ───────────────────────────────
        self._fitted_values = (
            self._weights["trend"]    * trend
            + self._weights["seasonal"] * seasonal
            + self._weights["nonlin"]   * nl_insample
            + self._weights["adaptive"] * a_stage.in_sample()
        )

        # ── Store residual stats for CI ───────────────────────────
        resid_full = y - self._fitted_values
        self._insample_rmse = float(np.sqrt(np.mean(resid_full**2)))
        self._resid_full = resid_full

        self._fitted = True
        return self

    # ── forecast ─────────────────────────────────────────────────────

    def forecast(self, h: int) -> np.ndarray:
        """
        Point forecast for horizons 1 … h.

        Parameters
        ----------
        h : int, forecast horizon (must be ≥ 1)

        Returns
        -------
        fc : np.ndarray, shape (h,)
        """
        self._check_fitted()
        h = int(h)
        if h < 1:
            raise ValueError("h must be ≥ 1.")

        trend_fc, seasonal_fc = forecast_components(
            self._trend, self._sea_coef, self._period, self.n_fourier, h
        )
        nl_fc = self._nl_stage.forecast(h)
        a_fc  = self._a_stage.forecast(h)

        fc = (
            self._weights["trend"]    * trend_fc
            + self._weights["seasonal"] * seasonal_fc
            + self._weights["nonlin"]   * nl_fc
            + self._weights["adaptive"] * a_fc
        )
        return fc

    # ── prediction intervals ─────────────────────────────────────────

    def predict_interval(
        self, h: int, level: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction interval [lower, upper] for horizons 1…h.

        Analytical method (default):
            σ_h = σ_1 · √h  (random-walk uncertainty propagation)
            CI  = ŷ_{t+h} ± z_{α/2} · σ_h

        Bootstrap method:
            Resample residuals with replacement, re-forecast B times,
            take empirical quantiles at (1−level)/2 and (1+level)/2.

        Parameters
        ----------
        h     : forecast horizon
        level : coverage probability (default 0.95)

        Returns
        -------
        lower, upper : np.ndarray, each shape (h,)
        """
        self._check_fitted()
        fc = self.forecast(h)

        if self.ci_method == "bootstrap":
            return self._bootstrap_interval(fc, h, level)
        else:
            return self._analytical_interval(fc, h, level)

    # ── evaluate ─────────────────────────────────────────────────────

    def evaluate(
        self, y_test: np.ndarray, period: Optional[int] = None
    ) -> dict:
        """
        Compute forecast accuracy metrics against a held-out test set.

        Automatically calls forecast(h=len(y_test)).

        Returns
        -------
        dict with keys: MAE, RMSE, MAPE, sMAPE, MASE
        """
        self._check_fitted()
        y_test = np.asarray(y_test, dtype=float)
        h = len(y_test)
        y_pred = self.forecast(h)
        p = period if period is not None else self._period
        return metrics.summary(y_test, y_pred, y_train=self._y, period=p)

    # ── fitted values / residuals ─────────────────────────────────────

    @property
    def fitted_values(self) -> np.ndarray:
        """In-sample fitted values ŷ_1, ..., ŷ_n."""
        self._check_fitted()
        return self._fitted_values

    @property
    def residuals(self) -> np.ndarray:
        """In-sample residuals y - ŷ."""
        self._check_fitted()
        return self._resid_full

    @property
    def components(self) -> dict:
        """
        Return a dict of in-sample component arrays:
            trend, seasonal, nonlinear, adaptive
        """
        self._check_fitted()
        return {
            "trend":    self._trend,
            "seasonal": self._seasonal,
            "nonlinear": self._nl_stage.in_sample(),
            "adaptive": self._a_stage.in_sample(),
        }

    @property
    def weights(self) -> dict:
        """Ensemble weights for each component."""
        self._check_fitted()
        return self._weights

    # ── summary ──────────────────────────────────────────────────────

    def summary(self) -> str:
        """Print a human-readable model summary."""
        self._check_fitted()
        n = len(self._y)
        w = self._weights
        lines = [
            "=" * 60,
            "  DNA — Dynamic Nonlinear Adaptive Forecaster",
            "=" * 60,
            f"  Series length    : {n}",
            f"  Seasonal period  : {self._period}",
            f"  Trend window (m) : {self._trend_window}",
            f"  Fourier harmonics: {self.n_fourier}",
            f"  Poly degree      : {self.poly_degree}",
            f"  AR lags          : {self.n_lags}",
            f"  RBF centres      : {len(self._feat_map._centres)}",
            f"  Ridge α          : {self.ridge_alpha}",
            f"  Kalman MLE       : {self.kalman_mle}",
            f"  Ensemble method  : {self.ensemble}",
            "",
            "  Component weights:",
            f"    Trend    α = {w['trend']:.4f}",
            f"    Seasonal β = {w['seasonal']:.4f}",
            f"    Nonlin   γ = {w['nonlin']:.4f}",
            f"    Adaptive δ = {w['adaptive']:.4f}",
            "",
            f"  In-sample RMSE   : {self._insample_rmse:.4f}",
            "=" * 60,
        ]
        result = "\n".join(lines)
        print(result)
        return result

    # ── internal helpers ──────────────────────────────────────────────

    def _compute_weights(self, y: np.ndarray, components: dict) -> dict:
        """
        Compute ensemble combination weights by the chosen method.

        iv  : w_i = (1/σ²_i) / Σ (1/σ²_j)    (inverse-variance)
        equal : w_i = 0.25 for all i
        ols : w = argmin ‖y − Cw‖²            (non-negative LS)
        """
        keys = ["trend", "seasonal", "nonlin", "adaptive"]
        n = len(y)

        if self.ensemble == "equal":
            return {k: 0.25 for k in keys}

        if self.ensemble == "iv":
            variances = {}
            for k in keys:
                resid = y - components[k]
                variances[k] = max(np.var(resid), 1e-12)
            inv_var = {k: 1.0 / v for k, v in variances.items()}
            total = sum(inv_var.values())
            return {k: inv_var[k] / total for k in keys}

        if self.ensemble == "ols":
            # Non-negative least squares stacking
            from scipy.optimize import nnls
            C = np.column_stack([components[k] for k in keys])
            w_raw, _ = nnls(C, y)
            w_sum = w_raw.sum()
            if w_sum == 0:
                return {k: 0.25 for k in keys}
            return dict(zip(keys, w_raw / w_sum))

        raise ValueError(f"Unknown ensemble method: '{self.ensemble}'. "
                         "Choose 'iv', 'equal', or 'ols'.")

    def _analytical_interval(
        self, fc: np.ndarray, h: int, level: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        from scipy.stats import norm
        z = norm.ppf(0.5 + level / 2)
        sigma_h = self._insample_rmse * np.sqrt(np.arange(1, h + 1))
        return fc - z * sigma_h, fc + z * sigma_h

    def _bootstrap_interval(
        self, fc: np.ndarray, h: int, level: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.random_state)
        resid = self._resid_full
        B = self.ci_bootstrap_n
        boot_fc = np.zeros((B, h))
        for b in range(B):
            boot_resid = rng.choice(resid, size=h, replace=True)
            # cumulative sum to propagate shock
            boot_fc[b] = fc + np.cumsum(boot_resid)
        alpha = (1 - level) / 2
        lower = np.quantile(boot_fc, alpha, axis=0)
        upper = np.quantile(boot_fc, 1 - alpha, axis=0)
        return lower, upper

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call fit() before calling this method.")
