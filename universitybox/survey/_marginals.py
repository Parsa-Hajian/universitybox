"""
universitybox.survey._marginals
================================
Stage 1: Per-column Bayesian marginal estimation.

For each question type we learn a posterior distribution over its parameters
using conjugate Bayesian inference, then draw posterior-predictive samples.

Categorical  — Dirichlet-Multinomial (Jeffreys prior: alpha_k = 1/K)
Ordinal      — Bayesian Ordinal Probit with Gibbs sampler (Albert & Chib 1993)
               Latent variable: z_i ~ N(mu, 1)
               Thresholds:      tau_0 = -inf, tau_1, ..., tau_{K-1}, tau_K = +inf
               Likelihood:      y_i = k  iff  tau_{k-1} < z_i <= tau_k
Continuous   — Normal-Inverse-Gamma (NIG) conjugate
               Prior: mu | sigma^2 ~ N(mu_0, sigma^2/kappa_0)
                      sigma^2 ~ InvGamma(alpha_0, beta_0)
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Categorical
# ---------------------------------------------------------------------------

class CategoricalMarginal:
    """
    Dirichlet-Multinomial model for a nominal categorical column.

    Prior: Dirichlet(alpha_k = 1/K) — Jeffreys prior for a K-category variable.
    Posterior: Dirichlet(alpha_k + n_k).
    Posterior-predictive samples: draw theta ~ Dirichlet posterior,
                                  then draw y ~ Categorical(theta).
    """

    def __init__(self, categories: List[str]) -> None:
        self.categories = list(categories)
        self.K = len(categories)
        self._cat_to_idx: Dict[str, int] = {c: i for i, c in enumerate(categories)}
        self._posterior_alpha: Optional[np.ndarray] = None

    def fit(self, col: np.ndarray) -> "CategoricalMarginal":
        """col: 1-D array of category strings (or ints mapped from categories)."""
        counts = np.zeros(self.K)
        for val in col:
            idx = self._cat_to_idx.get(str(val))
            if idx is not None:
                counts[idx] += 1
        jeffreys_alpha = 1.0 / self.K
        self._posterior_alpha = counts + jeffreys_alpha
        return self

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw n posterior-predictive category labels."""
        if self._posterior_alpha is None:
            raise RuntimeError("Call fit() before sample().")
        theta = rng.dirichlet(self._posterior_alpha)
        idx = rng.choice(self.K, size=n, p=theta)
        return np.array(self.categories)[idx]

    def cdf(self, col: np.ndarray) -> np.ndarray:
        """Empirical CDF values in [0,1] for use by the copula layer."""
        # Convert to integer indices, then use Dirichlet posterior mean for smoothed CDF
        if self._posterior_alpha is None:
            raise RuntimeError("Call fit() before cdf().")
        probs = self._posterior_alpha / self._posterior_alpha.sum()
        cum = np.concatenate([[0.0], np.cumsum(probs)])  # length K+1
        result = np.empty(len(col), dtype=float)
        for i, val in enumerate(col):
            idx = self._cat_to_idx.get(str(val), 0)
            # midpoint of the interval assigned to this category
            result[i] = (cum[idx] + cum[idx + 1]) / 2.0
        return result

    def quantile(self, u: np.ndarray) -> np.ndarray:
        """Inverse CDF: uniform u -> category label."""
        if self._posterior_alpha is None:
            raise RuntimeError("Call fit() before quantile().")
        probs = self._posterior_alpha / self._posterior_alpha.sum()
        cum = np.concatenate([[0.0], np.cumsum(probs)])
        result = np.empty(len(u), dtype=object)
        for i, ui in enumerate(u):
            ui = float(np.clip(ui, 0.0, 1.0 - 1e-12))
            idx = int(np.searchsorted(cum[1:], ui, side="right"))
            idx = min(idx, self.K - 1)
            result[i] = self.categories[idx]
        return result


# ---------------------------------------------------------------------------
# Ordinal (Bayesian Ordinal Probit, Albert & Chib 1993)
# ---------------------------------------------------------------------------

class OrdinalMarginal:
    """
    Bayesian Ordinal Probit via Gibbs sampler.

    Model:
        z_i ~ N(mu, 1)   (latent utility)
        y_i = k  iff  tau_{k-1} < z_i <= tau_k

    Priors:
        mu ~ N(0, 100)         (vague)
        tau_k: unconstrained except ordering; initialised at equispaced values.

    The Gibbs sampler iterates:
        1. Draw z_i | y_i, mu, tau  ~ TruncatedNormal(mu, 1, tau_{y_i-1}, tau_{y_i})
        2. Draw mu | z, tau          ~ N(posterior_mean, posterior_var)
        3. Draw tau_k | z, tau_{-k} from uniform on (max z in class k, min z in class k+1)
           (standard Albert & Chib threshold update)
    """

    def __init__(self, scale: Tuple[int, int], n_mcmc: int = 500) -> None:
        self.lo, self.hi = scale
        self.K = self.hi - self.lo + 1          # number of ordinal levels
        self.n_mcmc = n_mcmc
        self._mu_samples: Optional[np.ndarray] = None
        self._tau_samples: Optional[np.ndarray] = None
        # empirical CDF for copula
        self._emp_counts: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def fit(self, col: np.ndarray, rng: np.random.Generator) -> "OrdinalMarginal":
        """col: 1-D array of ints in [lo, hi]."""
        y = np.asarray(col, dtype=int) - self.lo   # recode to 0..K-1
        n = len(y)

        # Count empirical frequencies for CDF
        counts = np.bincount(y, minlength=self.K).astype(float)
        self._emp_counts = counts

        # --- Initial values ---
        # Thresholds: K+1 values; tau[0]=-inf, tau[K]=+inf; tau[1..K-1] free
        # Internal thresholds indexed 1..K-1 (K-1 free parameters)
        tau = np.empty(self.K + 1)
        tau[0] = -np.inf
        tau[self.K] = np.inf
        tau[1:self.K] = np.linspace(-1.5, 1.5, self.K - 1)

        mu = 0.0
        z = np.zeros(n, dtype=float)

        mu_prior_mean = 0.0
        mu_prior_var = 100.0

        mu_samples = np.empty(self.n_mcmc)
        tau_samples = np.empty((self.n_mcmc, self.K + 1))

        # --- Gibbs ---
        for it in range(self.n_mcmc):
            # 1. Draw z_i | y_i, mu, tau
            for k in range(self.K):
                idx = (y == k)
                if idx.sum() == 0:
                    continue
                a = tau[k]
                b = tau[k + 1]
                # truncated normal draw
                lo_p = stats.norm.cdf(a - mu) if np.isfinite(a) else 0.0
                hi_p = stats.norm.cdf(b - mu) if np.isfinite(b) else 1.0
                lo_p = np.clip(lo_p, 1e-15, 1 - 1e-15)
                hi_p = np.clip(hi_p, 1e-15, 1 - 1e-15)
                if hi_p <= lo_p:
                    hi_p = lo_p + 1e-10
                u = rng.uniform(lo_p, hi_p, size=idx.sum())
                z[idx] = stats.norm.ppf(u) + mu

            # 2. Draw mu | z
            post_var = 1.0 / (n / 1.0 + 1.0 / mu_prior_var)
            post_mean = post_var * (z.sum() / 1.0 + mu_prior_mean / mu_prior_var)
            mu = rng.normal(post_mean, math.sqrt(post_var))

            # 3. Draw tau_k (internal thresholds) | z, ordered
            for k in range(1, self.K):
                # tau[k] must satisfy: max(z[y==k-1]) < tau[k] < min(z[y==k])
                left_z = z[y == k - 1]
                right_z = z[y == k]
                lb = left_z.max() if len(left_z) else tau[k - 1]
                ub = right_z.min() if len(right_z) else tau[k + 1]
                lb = max(lb, tau[k - 1] if np.isfinite(tau[k - 1]) else -10.0)
                ub = min(ub, tau[k + 1] if np.isfinite(tau[k + 1]) else 10.0)
                if ub > lb:
                    tau[k] = rng.uniform(lb, ub)

            mu_samples[it] = mu
            tau_samples[it] = tau.copy()

        # Keep second half (burn-in discard)
        half = self.n_mcmc // 2
        self._mu_samples = mu_samples[half:]
        self._tau_samples = tau_samples[half:]
        return self

    # ------------------------------------------------------------------
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw n posterior-predictive ordinal integers in [lo, hi]."""
        if self._mu_samples is None:
            raise RuntimeError("Call fit() before sample().")
        out = np.empty(n, dtype=int)
        # Pick a random posterior draw for each sample
        idx = rng.integers(0, len(self._mu_samples), size=n)
        mus = self._mu_samples[idx]
        taus = self._tau_samples[idx]  # (n, K+1)
        z = rng.normal(mus, 1.0)
        for i in range(n):
            tau_i = taus[i]
            k = int(np.searchsorted(tau_i[1:self.K], z[i], side="right"))
            k = max(0, min(k, self.K - 1))
            out[i] = k + self.lo
        return out

    # ------------------------------------------------------------------
    def cdf(self, col: np.ndarray) -> np.ndarray:
        """Smoothed empirical CDF for copula (midpoint rule)."""
        counts = self._emp_counts if self._emp_counts is not None else np.ones(self.K)
        probs = (counts + 0.5) / (counts.sum() + 0.5 * self.K)  # add-half smoothing
        cum = np.concatenate([[0.0], np.cumsum(probs)])
        result = np.empty(len(col), dtype=float)
        for i, val in enumerate(col):
            k = int(val) - self.lo
            k = max(0, min(k, self.K - 1))
            result[i] = (cum[k] + cum[k + 1]) / 2.0
        return result

    def quantile(self, u: np.ndarray) -> np.ndarray:
        """Inverse CDF: uniform u -> integer in [lo, hi]."""
        counts = self._emp_counts if self._emp_counts is not None else np.ones(self.K)
        probs = (counts + 0.5) / (counts.sum() + 0.5 * self.K)
        cum = np.concatenate([[0.0], np.cumsum(probs)])
        result = np.empty(len(u), dtype=int)
        for i, ui in enumerate(u):
            ui = float(np.clip(ui, 0.0, 1.0 - 1e-12))
            k = int(np.searchsorted(cum[1:], ui, side="right"))
            k = max(0, min(k, self.K - 1))
            result[i] = k + self.lo
        return result


# ---------------------------------------------------------------------------
# Continuous (Normal-Inverse-Gamma conjugate)
# ---------------------------------------------------------------------------

class ContinuousMarginal:
    """
    Normal-Inverse-Gamma (NIG) conjugate model for a bounded continuous column.

    Prior (vague):
        mu_0 = sample_mean  (updated empirically)
        kappa_0 = 1         (weak prior on mean)
        alpha_0 = 1         (weak prior on variance)
        beta_0  = sample_var (weak prior on variance scale)

    Posterior-predictive: Student-t  (closed form, no MCMC needed).
    Samples are clipped to declared bounds.
    """

    def __init__(self, bounds: Tuple[float, float]) -> None:
        self.lo, self.hi = bounds
        self._post: Optional[dict] = None
        self._mean: float = 0.0
        self._std: float = 1.0

    def fit(self, col: np.ndarray) -> "ContinuousMarginal":
        """col: 1-D array of floats."""
        col = np.asarray(col, dtype=float)
        n = len(col)
        x_bar = col.mean()
        s2 = col.var(ddof=1) if n > 1 else 1.0

        # Vague NIG prior
        mu_0 = x_bar
        kappa_0 = 1.0
        alpha_0 = 1.0
        beta_0 = s2

        # NIG update
        kappa_n = kappa_0 + n
        mu_n = (kappa_0 * mu_0 + n * x_bar) / kappa_n
        alpha_n = alpha_0 + n / 2.0
        beta_n = (
            beta_0
            + 0.5 * (n - 1) * s2
            + (kappa_0 * n * (x_bar - mu_0) ** 2) / (2.0 * kappa_n)
        )

        self._post = dict(mu_n=mu_n, kappa_n=kappa_n, alpha_n=alpha_n, beta_n=beta_n)
        self._mean = x_bar
        self._std = math.sqrt(s2) if s2 > 0 else 1.0
        return self

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw n posterior-predictive samples, clipped to bounds."""
        if self._post is None:
            raise RuntimeError("Call fit() before sample().")
        p = self._post
        # Posterior predictive: Student-t with df=2*alpha_n
        df = 2.0 * p["alpha_n"]
        scale = math.sqrt(p["beta_n"] * (p["kappa_n"] + 1) / (p["alpha_n"] * p["kappa_n"]))
        samples = rng.standard_t(df, size=n) * scale + p["mu_n"]
        return np.clip(samples, self.lo, self.hi)

    def cdf(self, col: np.ndarray) -> np.ndarray:
        """Empirical CDF (rank-based) for copula, with small correction for ties."""
        col = np.asarray(col, dtype=float)
        n = len(col)
        # Blom-style rank-based CDF: (rank - 3/8) / (n + 1/4)
        ranks = stats.rankdata(col, method="average")
        return (ranks - 0.375) / (n + 0.25)

    def quantile(self, u: np.ndarray) -> np.ndarray:
        """Inverse CDF via posterior-predictive Student-t, clipped to bounds."""
        if self._post is None:
            raise RuntimeError("Call fit() before quantile().")
        p = self._post
        df = 2.0 * p["alpha_n"]
        scale = math.sqrt(p["beta_n"] * (p["kappa_n"] + 1) / (p["alpha_n"] * p["kappa_n"]))
        u = np.clip(u, 1e-8, 1 - 1e-8)
        vals = stats.t.ppf(u, df=df) * scale + p["mu_n"]
        return np.clip(vals, self.lo, self.hi)
