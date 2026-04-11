"""
universitybox.survey._copula
==============================
Stage 2: Gaussian Copula for preserving cross-question correlations.

Algorithm
---------
Given a list of fitted marginal models (one per column):

Fit:
  1. For each observed column, compute pseudo-uniform values u_j via the
     marginal's cdf() method (rank-based empirical CDF).
  2. Apply the probit transform: w_j = Phi^{-1}(u_j).
  3. Estimate the correlation matrix R of the normal scores w.
     With small N, use Ledoit-Wolf shrinkage to regularise R.

Synthesise:
  1. Draw Z ~ MVN(0, R)  (shape: (n_synthetic, p)).
  2. Convert back to uniform: u_j = Phi(Z_j).
  3. Apply each marginal's quantile() to map u_j -> column value.

References
----------
- Genest & Favre (2007) "Everything you always wanted to know about copula
  modeling but were afraid to ask." J. Hydrol. Eng.
- Ledoit & Wolf (2004) "A well-conditioned estimator for large-dimensional
  covariance matrices." J. Multivariate Anal.
"""
from __future__ import annotations

from typing import List, TYPE_CHECKING

import numpy as np
from scipy import stats
from scipy.linalg import cholesky

if TYPE_CHECKING:
    from ._marginals import CategoricalMarginal, OrdinalMarginal, ContinuousMarginal

MarginalType = "CategoricalMarginal | OrdinalMarginal | ContinuousMarginal"


def _ledoit_wolf_shrinkage(W: np.ndarray) -> np.ndarray:
    """
    Ledoit-Wolf analytic shrinkage estimator for a correlation matrix.

    W  : (n, p) matrix of normal scores (zero-centered by column).
    Returns a (p, p) regularised correlation matrix.

    Uses the Oracle Approximating Shrinkage (OAS) formula from
    Chen, Wiesel, Eldar & Hero (2010) as a fast analytic alternative.
    We fall back to simple sample correlation when n > 5*p (shrinkage
    has negligible effect in that regime).
    """
    n, p = W.shape
    # Sample correlation
    S = np.corrcoef(W, rowvar=False)
    if p == 1:
        return S

    if n > 5 * p:
        return S  # well-estimated regime — no shrinkage needed

    # OAS shrinkage coefficient
    mu = np.trace(S) / p
    # rho  (optimal shrinkage intensity toward mu * I)
    alpha = (
        (np.trace(S @ S) + np.trace(S) ** 2)
        / ((n + 1 - 2.0 / p) * (np.trace(S @ S) - np.trace(S) ** 2 / p))
    )
    rho = min(1.0, max(0.0, alpha))
    R_shrunk = (1.0 - rho) * S + rho * np.eye(p)
    return R_shrunk


class GaussianCopula:
    """
    Gaussian Copula: fits a joint dependency structure from observed data
    and synthesises new samples that preserve it.

    Parameters
    ----------
    marginals : list of fitted marginal models (one per survey column)
    """

    def __init__(self) -> None:
        self._R: np.ndarray | None = None
        self._p: int = 0

    # ------------------------------------------------------------------
    def fit(self, df_obs, marginals: List) -> "GaussianCopula":
        """
        Estimate the copula correlation matrix from observed data.

        Parameters
        ----------
        df_obs   : pandas DataFrame (n_obs x p), already validated.
        marginals: list of fitted marginal objects (same order as df_obs columns).
        """
        import pandas as pd  # noqa: F401

        n, p = df_obs.shape
        self._p = p

        # Step 1 — pseudo-uniform values via marginal CDFs
        U = np.empty((n, p), dtype=float)
        for j, (col_name, marginal) in enumerate(zip(df_obs.columns, marginals)):
            col = df_obs[col_name].values
            U[:, j] = marginal.cdf(col)

        # Step 2 — probit transform (clip to avoid ±inf)
        U = np.clip(U, 1e-6, 1 - 1e-6)
        W = stats.norm.ppf(U)  # (n, p)

        # Step 3 — regularised correlation matrix
        self._R = _ledoit_wolf_shrinkage(W)
        return self

    # ------------------------------------------------------------------
    def synthesise(
        self,
        n_synthetic: int,
        marginals: List,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Draw n_synthetic joint samples from the fitted copula.

        Returns
        -------
        out : (n_synthetic, p) numpy array (dtype=object to hold mixed types).
        """
        if self._R is None:
            raise RuntimeError("Call fit() before synthesise().")

        p = self._p

        # Cholesky of R for efficient MVN sampling
        try:
            L = cholesky(self._R, lower=True)
        except Exception:
            # If R is not PD (numerical issues), add small diagonal regularisation
            R_reg = self._R + 1e-6 * np.eye(p)
            L = cholesky(R_reg, lower=True)

        # Step 1 — draw Z ~ MVN(0, R)
        Z = rng.standard_normal((n_synthetic, p)) @ L.T  # (n_synthetic, p)

        # Step 2 — back to uniform
        U = stats.norm.cdf(Z)  # (n_synthetic, p)

        # Step 3 — apply inverse marginal CDFs (quantile functions)
        out = np.empty((n_synthetic, p), dtype=object)
        for j, marginal in enumerate(marginals):
            out[:, j] = marginal.quantile(U[:, j])

        return out
