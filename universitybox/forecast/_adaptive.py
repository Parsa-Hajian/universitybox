"""
A-Stage: Adaptive Kalman filter module for the DNA forecaster.

Mathematical formulation
------------------------
After D and N stage corrections, a second-order residual remains:

    r_t = ε̂_t − f̂(x_t)

We model r_t with a Linear Gaussian State-Space Model (LG-SSM)
and infer the latent state via the Kalman filter.

─── State-space model ───────────────────────────────────────────────────
We use a Local Linear Trend (LLT) model:

    State vector: x_t = [l_t, b_t]ᵀ   (level, slope)

    Transition equation:
        x_t = F x_{t-1} + G w_t,    w_t ~ N(0, Q)

    Observation equation:
        r_t = H x_t + v_t,           v_t ~ N(0, R)

    where
        F = [[1, 1],    G = I_2,    H = [1, 0]
             [0, 1]]

        Q = diag(q_l, q_b)  — process noise for level and slope
        R = σ²_v            — observation noise variance

─── Kalman filter recursion ─────────────────────────────────────────────
Initialisation (diffuse prior):
    x̂_{1|0} = [r_1, 0]ᵀ
    P_{1|0} = diag(10⁶, 10⁶)

For t = 1, ..., n:

    Prediction step:
        x̂_{t|t-1} = F x̂_{t-1|t-1}                           (Pred-1)
        P_{t|t-1}  = F P_{t-1|t-1} Fᵀ + G Q Gᵀ              (Pred-2)

    Innovation:
        v_t = r_t − H x̂_{t|t-1}                              (Inn)
        S_t = H P_{t|t-1} Hᵀ + R                              (Inn-Cov)

    Kalman gain:
        K_t = P_{t|t-1} Hᵀ S_t⁻¹                             (Gain)

    Update step:
        x̂_{t|t} = x̂_{t|t-1} + K_t v_t                       (Upd-1)
        P_{t|t}  = (I − K_t H) P_{t|t-1}                     (Upd-2)
        (Joseph form for numerical stability)

─── h-step ahead forecast ───────────────────────────────────────────────
    x̂_{n+h|n} = F^h x̂_{n|n}                                  (Forecast)

    P_{n+h|n}  = F^h P_{n|n} (Fᵀ)^h + Σ_{j=0}^{h-1} F^j G Q Gᵀ (Fᵀ)^j

    ŷ_{n+h|n} = H x̂_{n+h|n}

─── Parameter estimation (MLE) ──────────────────────────────────────────
Log-likelihood of the LLT model via prediction-error decomposition:

    log L(Q, R) = −(n/2) log(2π) − (1/2) Σ_{t=1}^n [log S_t + v_t²/S_t]

Optimised via scipy.optimize.minimize with Nelder-Mead (gradient-free,
robust for small n). Parameters are log-transformed for positivity:
    θ = [log q_l, log q_b, log R].

If MLE is skipped (mle=False), the user-supplied q and r are used
directly. This is faster and recommended when n < 20.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg
from typing import Optional, Tuple


# ──────────────────────────────────────────────────────────────────────
# Local Linear Trend Kalman filter
# ──────────────────────────────────────────────────────────────────────

class KalmanLLT:
    """
    Local Linear Trend Kalman filter / smoother.

    Parameters
    ----------
    q_level : process noise variance for the level component
    q_slope : process noise variance for the slope component
    obs_var : observation noise variance R
    """

    def __init__(
        self,
        q_level: float = 1e-4,
        q_slope: float = 1e-6,
        obs_var: float = 1e-2,
    ):
        self.q_level = q_level
        self.q_slope = q_slope
        self.obs_var = obs_var

        # System matrices
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self._build_noise()

    def _build_noise(self):
        self.Q = np.diag([self.q_level, self.q_slope])
        self.R = np.array([[self.obs_var]])

    # ── Kalman filter pass ───────────────────────────────────────────

    def filter(self, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run the Kalman filter on series r.

        Parameters
        ----------
        r : shape (n,) — second-order residuals

        Returns
        -------
        x_filt : filtered states x̂_{t|t}, shape (n, 2)
        P_last : posterior covariance at t=n, shape (2, 2)
        log_lik: marginal log-likelihood
        """
        n = len(r)
        F, H, Q, R = self.F, self.H, self.Q, self.R

        # Diffuse initialisation
        x = np.array([r[0], 0.0])
        P = np.eye(2) * 1e6

        x_filt = np.zeros((n, 2))
        log_lik = 0.0

        for t in range(n):
            # ── Prediction ─────────────────────────────────────
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q

            # ── Innovation ─────────────────────────────────────
            y_t = np.array([r[t]])
            v = y_t - H @ x_pred                    # scalar innovation
            S = H @ P_pred @ H.T + R                # innovation variance

            # ── Log-likelihood accumulation ────────────────────
            S_inv = np.linalg.inv(S)
            log_lik += -0.5 * (np.log(np.linalg.det(S)) + float(v.T @ S_inv @ v))

            # ── Kalman gain ────────────────────────────────────
            K = P_pred @ H.T @ S_inv

            # ── Update (Joseph form for stability) ─────────────
            IKH = np.eye(2) - K @ H
            x = x_pred + K @ v
            P = IKH @ P_pred @ IKH.T + K @ R @ K.T

            x_filt[t] = x

        log_lik -= (n / 2) * np.log(2 * np.pi)
        return x_filt, P, log_lik

    # ── h-step forecast ─────────────────────────────────────────────

    def forecast(self, x_last: np.ndarray, P_last: np.ndarray, h: int) -> np.ndarray:
        """
        Produce h-step forecasts H F^k x_{n|n} for k = 1..h.

        Parameters
        ----------
        x_last : state at t = n, shape (2,)
        P_last : covariance at t = n, shape (2, 2)
        h      : forecast horizon

        Returns
        -------
        fc : point forecasts, shape (h,)
        """
        F, H = self.F, self.H
        x = x_last.copy()
        fc = np.zeros(h)
        for k in range(h):
            x = F @ x
            fc[k] = float((H @ x).item())
        return fc

    # ── MLE parameter estimation ────────────────────────────────────

    @classmethod
    def fit_mle(cls, r: np.ndarray, **kwargs) -> "KalmanLLT":
        """
        Estimate (q_level, q_slope, obs_var) by maximising the
        Kalman filter log-likelihood via Nelder-Mead.

        Parameters
        ----------
        r : shape (n,) — second-order residual series
        **kwargs : passed to KalmanLLT constructor as initial values

        Returns
        -------
        Fitted KalmanLLT instance.
        """
        from scipy.optimize import minimize

        def neg_log_lik(log_params):
            ql, qb, rv = np.exp(log_params)
            model = cls(q_level=ql, q_slope=qb, obs_var=rv)
            _, _, ll = model.filter(r)
            return -ll

        # Initial log-params from kwargs or defaults
        q0 = np.log([
            kwargs.get("q_level", 1e-4),
            kwargs.get("q_slope", 1e-6),
            kwargs.get("obs_var", max(np.var(r), 1e-8)),
        ])

        result = minimize(neg_log_lik, q0, method="Nelder-Mead",
                         options={"maxiter": 2000, "xatol": 1e-6})

        ql, qb, rv = np.exp(result.x)
        return cls(q_level=ql, q_slope=qb, obs_var=rv)


# ──────────────────────────────────────────────────────────────────────
# A-Stage wrapper
# ──────────────────────────────────────────────────────────────────────

class AdaptiveStage:
    """
    A-Stage: fit Kalman LLT on the second-order residual and produce
    adaptive correction forecasts.

    Parameters
    ----------
    q_level : level process noise (overridden if mle=True)
    q_slope : slope process noise (overridden if mle=True)
    obs_var : observation noise (overridden if mle=True)
    mle     : if True, estimate noise parameters by MLE
    """

    def __init__(
        self,
        q_level: float = 1e-4,
        q_slope: float = 1e-6,
        obs_var: float = 1e-2,
        mle: bool = False,
    ):
        self.q_level = q_level
        self.q_slope = q_slope
        self.obs_var = obs_var
        self.mle = mle
        self._kalman: Optional[KalmanLLT] = None
        self._x_last: Optional[np.ndarray] = None
        self._P_last: Optional[np.ndarray] = None

    def fit(self, r2: np.ndarray) -> "AdaptiveStage":
        """
        Fit the Kalman filter on second-order residual r2 = ε̂ − f̂.
        """
        if self.mle:
            self._kalman = KalmanLLT.fit_mle(
                r2, q_level=self.q_level,
                q_slope=self.q_slope, obs_var=self.obs_var
            )
        else:
            self._kalman = KalmanLLT(
                q_level=self.q_level,
                q_slope=self.q_slope,
                obs_var=self.obs_var,
            )

        x_filt, P_last, _ = self._kalman.filter(r2)
        self._x_last = x_filt[-1]
        self._P_last = P_last
        self._in_sample = (
            self._kalman.H @ x_filt.T
        ).ravel()
        return self

    def in_sample(self) -> np.ndarray:
        return self._in_sample

    def forecast(self, h: int) -> np.ndarray:
        return self._kalman.forecast(self._x_last, self._P_last, h)
