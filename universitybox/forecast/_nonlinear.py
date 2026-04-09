"""
N-Stage: Nonlinear basis-expansion module for the DNA forecaster.

Mathematical formulation
------------------------
Given the D-stage residual ε̂ = (ε̂_1, ..., ε̂_n), the N-stage fits a
function f in a Reproducing Kernel Hilbert Space (RKHS) ℋ:

    ε̂_t ≈ f(x_t) + η_t,     η_t ~ N(0, σ²_η)         (3)

where x_t ∈ ℝ^d is a feature vector derived from the series history.

─── Feature map Φ ───────────────────────────────────────────────────────
We construct Φ: ℝ → ℝ^D via three dictionaries:

1. Polynomial basis (degree p):
        φ^poly(t) = [1, t/n, (t/n)², ..., (t/n)^p]      ∈ ℝ^{p+1}

2. Autoregressive lags (lags 1..L):
        φ^lag(t)  = [ε̂_{t-1}/σ, ..., ε̂_{t-L}/σ]        ∈ ℝ^L

3. Radial Basis Functions (RBF / squared exponential kernel):
        φ^rbf_j(t) = exp(−γ ‖ε̂_t − c_j‖²),   j = 1..J  ∈ ℝ^J

   Centres c_j selected by k-means++ on the residual history.
   Bandwidth γ estimated by the median heuristic:
        γ = 1 / (2 · median²(‖ε̂_i − ε̂_j‖))

Full feature vector (concatenation):
        Φ(t) = [φ^poly(t) | φ^lag(t) | φ^rbf(t)]        ∈ ℝ^D
        D = (p+1) + L + J

─── Ridge regression ────────────────────────────────────────────────────
Let Φ ∈ ℝ^{n×D} be the design matrix (row t → Φ(t)).
The regularised least-squares estimator:

    θ* = argmin_{θ ∈ ℝ^D} ‖ε̂ − Φθ‖² + λ‖θ‖²           (4)

Closed-form solution (normal equations):

    θ* = (ΦᵀΦ + λ I_D)⁻¹ Φᵀ ε̂

Computational note: solved via scipy.linalg.solve with Cholesky
factorisation (O(D³) in parameter dimension, O(nD) for Φᵀε).

Dual form (when D >> n, rare in practice):
    θ* = Φᵀ (ΦΦᵀ + λ I_n)⁻¹ ε̂

─── k-means++ seed selection ────────────────────────────────────────────
RBF centres are chosen by k-means++ (Arthur & Vassilvitskii 2007):

    1. c_1 ~ Uniform({ε̂_t})
    2. For j = 2, ..., J:
          D(x) = min_{i<j} ‖x − c_i‖²
          c_j  ~ Categorical(p_t ∝ D(ε̂_t))

This gives an O(log J) approximation guarantee over random initialisation.

─── Forecast extrapolation ──────────────────────────────────────────────
At horizon h:
  • Polynomial: evaluate φ^poly at t = n + h
  • Lag:        use the recursively predicted residuals ε̂_{n+1}, ...
                (set to 0 for h > L, as residuals are mean-zero by design)
  • RBF:        evaluate φ^rbf at the last observed residual centroid
                (assumes residual process stays near its stationary mean)

    f̂_{t+h} = Φ(n+h)ᵀ θ*                                (5)
"""
from __future__ import annotations

import numpy as np
from scipy import linalg
from typing import Tuple, Optional


# ──────────────────────────────────────────────────────────────────────
# k-means++ seed selection
# ──────────────────────────────────────────────────────────────────────

def _kmeanspp_seeds(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    k-means++ initialisation on 1-D array X.

    Returns k centre values from {X_i}.
    """
    X = X.reshape(-1, 1)
    n = len(X)
    first_idx = rng.integers(0, n)
    centres = [X[first_idx]]

    for _ in range(k - 1):
        D2 = np.array([
            min(np.sum((x - c) ** 2) for c in centres)
            for x in X
        ])
        probs = D2 / D2.sum()
        idx = rng.choice(n, p=probs)
        centres.append(X[idx])

    return np.array(centres).ravel()


# ──────────────────────────────────────────────────────────────────────
# Feature map
# ──────────────────────────────────────────────────────────────────────

class NonlinearFeatureMap:
    """
    Build and apply the composite feature map Φ for the N-stage.

    Parameters
    ----------
    poly_degree : degree p of polynomial component
    n_lags      : L autoregressive lags of the residual
    n_rbf       : J RBF centres
    rbf_gamma   : bandwidth γ ('auto' → median heuristic)
    random_state: seed for k-means++ centre selection
    """

    def __init__(
        self,
        poly_degree: int = 2,
        n_lags: int = 4,
        n_rbf: int = 10,
        rbf_gamma: float | str = "auto",
        random_state: Optional[int] = None,
    ):
        self.poly_degree = poly_degree
        self.n_lags = n_lags
        self.n_rbf = n_rbf
        self.rbf_gamma = rbf_gamma
        self.random_state = random_state

        self._rng = np.random.default_rng(random_state)
        self._centres: Optional[np.ndarray] = None
        self._gamma: Optional[float] = None
        self._residual_std: float = 1.0
        self._n_fit: int = 0

    # ── fit ──────────────────────────────────────────────────────────

    def fit(self, residual: np.ndarray) -> "NonlinearFeatureMap":
        """Fit RBF centres and bandwidth from residual history."""
        n = len(residual)
        self._n_fit = n
        self._residual_std = max(residual.std(), 1e-8)

        # ── RBF centres via k-means++ ────────────────────────────
        k = min(self.n_rbf, n)
        self._centres = _kmeanspp_seeds(residual, k, self._rng)

        # ── RBF bandwidth: median heuristic ─────────────────────
        if self.rbf_gamma == "auto":
            diffs = np.abs(
                residual[:, None] - residual[None, :]
            ).ravel()
            med = np.median(diffs[diffs > 0])
            self._gamma = 1.0 / (2.0 * med**2 + 1e-12)
        else:
            self._gamma = float(self.rbf_gamma)

        return self

    # ── transform ────────────────────────────────────────────────────

    def transform(
        self,
        residual: np.ndarray,
        n_offset: int = 0,
    ) -> np.ndarray:
        """
        Build design matrix Φ ∈ ℝ^{n × D} for the provided residual array.

        Parameters
        ----------
        residual  : shape (n,) — the residual series to featurise
        n_offset  : starting time index (0 for in-sample, n for out-of-sample)
        """
        n = len(residual)
        t = np.arange(n_offset, n_offset + n, dtype=float)
        n_total = self._n_fit + n_offset + n

        cols = []

        # ── 1. Polynomial ────────────────────────────────────────
        t_norm = t / max(n_total - 1, 1)
        for deg in range(self.poly_degree + 1):
            cols.append(t_norm**deg)

        # ── 2. AR lags ───────────────────────────────────────────
        L = self.n_lags
        for lag in range(1, L + 1):
            col = np.zeros(n)
            for i in range(n):
                src_idx = n_offset + i - lag
                if 0 <= src_idx < len(residual):
                    col[i] = residual[src_idx] / self._residual_std
                # else: 0 (zero-pad before series start)
            cols.append(col)

        # ── 3. RBF ──────────────────────────────────────────────
        for c in self._centres:
            phi = np.exp(-self._gamma * (residual - c) ** 2)
            cols.append(phi)

        return np.column_stack(cols)

    def transform_future(self, h: int) -> np.ndarray:
        """
        Feature matrix for h future steps.

        Lags are set to 0 (residuals are mean-zero; best linear
        predictor beyond the observed window is the mean).
        RBF evaluated at the series' mean (0 after centring).
        """
        n = self._n_fit
        t = np.arange(n, n + h, dtype=float)
        n_total = n + h

        cols = []

        # Polynomial
        t_norm = t / max(n_total - 1, 1)
        for deg in range(self.poly_degree + 1):
            cols.append(t_norm**deg)

        # AR lags → 0 for all h > 0
        for _ in range(self.n_lags):
            cols.append(np.zeros(h))

        # RBF at residual mean (0 after normalisation)
        for c in self._centres:
            phi = np.exp(-self._gamma * (0.0 - c) ** 2) * np.ones(h)
            cols.append(phi)

        return np.column_stack(cols)


# ──────────────────────────────────────────────────────────────────────
# N-stage fitter
# ──────────────────────────────────────────────────────────────────────

class NonlinearStage:
    """
    Fit and predict the nonlinear residual model via Ridge regression (eq. 4-5).

    Parameters
    ----------
    feature_map  : fitted NonlinearFeatureMap
    ridge_alpha  : L2 regularisation λ (default 1e-3)
    """

    def __init__(
        self,
        feature_map: NonlinearFeatureMap,
        ridge_alpha: float = 1e-3,
    ):
        self.feature_map = feature_map
        self.ridge_alpha = ridge_alpha
        self._theta: Optional[np.ndarray] = None
        self._in_sample_pred: Optional[np.ndarray] = None

    def fit(self, residual: np.ndarray) -> "NonlinearStage":
        """
        Fit Ridge model on the residual series (equation 4).
        """
        Phi = self.feature_map.transform(residual, n_offset=0)
        D = Phi.shape[1]

        A = Phi.T @ Phi + self.ridge_alpha * np.eye(D)
        b = Phi.T @ residual

        self._theta = linalg.solve(A, b, assume_a="pos")
        self._in_sample_pred = Phi @ self._theta
        return self

    def in_sample(self) -> np.ndarray:
        """Return in-sample fitted values f̂(x_t) for t = 1..n."""
        return self._in_sample_pred

    def forecast(self, h: int) -> np.ndarray:
        """Return out-of-sample forecasts f̂_{n+1..n+h} (equation 5)."""
        Phi_future = self.feature_map.transform_future(h)
        return Phi_future @ self._theta
