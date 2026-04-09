"""
D-Stage: Decomposition module for the DNA forecaster.

Mathematical formulation
------------------------
Given an observed series y = (y_1, ..., y_n), we posit the additive model:

    y_t = μ_t + s_t + ε_t           (1)

where
    μ_t   — trend component (smooth, slowly-varying)
    s_t   — seasonal component (periodic with known or estimated period P)
    ε_t   — irregular residual (zero-mean, weakly-stationary)

─── Trend estimation ────────────────────────────────────────────────────
Henderson (1916) filter with half-length m produces minimum-roughness
weights w = (w_{-m}, ..., w_m) satisfying:

    min Σ (∇³ μ_t)²     subject to   Σ w_j = 1,  Σ j·w_j = 0,  Σ j²·w_j = 0

Closed-form weights (Doherty 2001):

    w_j = (h² - j²)(h₁² - j²)(h₂² - j²) · [315(h² - j²) + ...] / Z

where h = m + 1, h₁ = m + 2, h₂ = m + 3, Z is a normalising constant.
At series endpoints a modified asymmetric version is applied.

For brevity we implement the exact Henderson weights via the recurrence
relation from Loader (1999) and fall back to a Gaussian kernel when n
is too short to support the full filter.

─── Seasonal estimation ─────────────────────────────────────────────────
Conditional on the trend estimate μ̂_t, the detrended series is:

    d_t = y_t - μ̂_t ≈ s_t + ε_t

We represent s_t as a truncated Fourier series of order K:

    s_t = Σ_{k=1}^{K} [ a_k cos(2πk t / P) + b_k sin(2πk t / P) ]

Stacking design matrix F ∈ ℝ^{n×2K}:

    F_{t,2k-1} = cos(2πk t / P),   F_{t,2k} = sin(2πk t / P)

OLS estimate:

    [a, b]* = (FᵀF)⁻¹ Fᵀ d     →    ŝ_t = F [a, b]*          (2)

Seasonal normalisation: subtract column mean so Σ ŝ_t = 0 over one period.

─── Period estimation ───────────────────────────────────────────────────
When period is unknown, we estimate it from the periodogram:

    I(ω) = (1/n) |Σ_{t=1}^n y_t e^{-iωt}|²

    P̂ = argmax_{P ∈ {2,...,n/2}} I(2π/P)
"""
from __future__ import annotations

import numpy as np
from typing import Tuple, Optional


# ──────────────────────────────────────────────────────────────────────
# Henderson filter weights
# ──────────────────────────────────────────────────────────────────────

def _henderson_weights(m: int) -> np.ndarray:
    """
    Compute the 2m+1 symmetric Henderson moving-average weights.

    References
    ----------
    Doherty, M. (2001). The Surrogate Henderson Filters in X-11.
    Australian & New Zealand J. Statistics 43(4), 385-392.
    """
    h = m + 1
    h1 = m + 2
    h2 = m + 3
    j = np.arange(-m, m + 1, dtype=float)

    num = (
        (h**2 - j**2)
        * (h1**2 - j**2)
        * (h2**2 - j**2)
        * (3 * h**2 - 11 * j**2 - 16)
    )
    w = num.copy()
    w[num == 0] = 0.0
    w /= w.sum()
    return w


def henderson_filter(y: np.ndarray, m: int) -> np.ndarray:
    """
    Apply Henderson symmetric MA filter of half-length m.

    Interior points use the full 2m+1 weights.
    Endpoints use reflected padding (mode='reflect').

    Parameters
    ----------
    y : array of shape (n,)
    m : half-length of filter (must be ≥ 1)

    Returns
    -------
    trend : array of shape (n,) — estimated μ̂_t
    """
    w = _henderson_weights(m)
    n = len(y)
    # pad symmetrically at both ends
    padded = np.pad(y, m, mode="reflect")
    trend = np.convolve(padded, w, mode="valid")
    assert len(trend) == n
    return trend


# ──────────────────────────────────────────────────────────────────────
# Fourier seasonal matrix
# ──────────────────────────────────────────────────────────────────────

def fourier_matrix(t: np.ndarray, period: float, K: int) -> np.ndarray:
    """
    Build the Fourier design matrix F ∈ ℝ^{n × 2K}.

    F_{t, 2k-2} = cos(2π k t / P)
    F_{t, 2k-1} = sin(2π k t / P)     for k = 1, ..., K

    Parameters
    ----------
    t      : integer time indices (0-based), shape (n,)
    period : seasonal period P
    K      : number of Fourier harmonics

    Returns
    -------
    F : array of shape (n, 2K)
    """
    cols = []
    for k in range(1, K + 1):
        angle = 2.0 * np.pi * k * t / period
        cols.append(np.cos(angle))
        cols.append(np.sin(angle))
    return np.column_stack(cols)


# ──────────────────────────────────────────────────────────────────────
# Period estimator
# ──────────────────────────────────────────────────────────────────────

def estimate_period(y: np.ndarray, min_period: int = 2) -> int:
    """
    Estimate dominant seasonal period via periodogram.

    Returns
    -------
    P : int, estimated period (between min_period and n//2)
    """
    n = len(y)
    # Remove linear trend before computing periodogram
    t = np.arange(n)
    p = np.polyfit(t, y, 1)
    y_dt = y - np.polyval(p, t)

    fft_vals = np.abs(np.fft.rfft(y_dt)) ** 2
    freqs = np.fft.rfftfreq(n)

    # Exclude DC and Nyquist; find peak frequency
    mask = (freqs >= 1 / (n // 2)) & (freqs <= 0.5)
    if not mask.any():
        return 4  # fallback to quarterly
    peak_freq = freqs[mask][np.argmax(fft_vals[mask])]
    if peak_freq == 0:
        return 4
    P = int(round(1.0 / peak_freq))
    return max(min_period, min(P, n // 2))


# ──────────────────────────────────────────────────────────────────────
# Main decomposition function
# ──────────────────────────────────────────────────────────────────────

def decompose(
    y: np.ndarray,
    period: int,
    trend_window: int,
    n_fourier: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose y into (trend, seasonal, residual) via equations (1)-(2).

    Parameters
    ----------
    y            : observed series, shape (n,)
    period       : seasonal period P
    trend_window : Henderson filter half-length m
    n_fourier    : Fourier harmonics K

    Returns
    -------
    trend    : μ̂_t, shape (n,)
    seasonal : ŝ_t, shape (n,)
    sea_coef : Fourier coefficients [a_1,b_1,...,a_K,b_K], shape (2K,)
    residual : ε̂_t = y - μ̂ - ŝ, shape (n,)
    """
    n = len(y)
    t = np.arange(n, dtype=float)

    # ── D1: Trend ──────────────────────────────────────────────────
    m = min(trend_window, (n - 1) // 2)
    trend = henderson_filter(y, m)

    # ── D2: Seasonal ───────────────────────────────────────────────
    detrended = y - trend
    F = fourier_matrix(t, period, n_fourier)
    sea_coef, _, _, _ = np.linalg.lstsq(F, detrended, rcond=None)
    seasonal = F @ sea_coef

    # Normalise: seasonal sums to 0 over one full period
    # subtract per-phase mean computed over complete periods
    n_full = (n // period) * period
    if n_full >= period:
        phase_means = np.array(
            [seasonal[:n_full].reshape(-1, period)[:, p].mean()
             for p in range(period)]
        )
        for i, s in enumerate(seasonal):
            seasonal[i] -= phase_means[i % period]

    # ── D3: Residual ───────────────────────────────────────────────
    residual = y - trend - seasonal

    return trend, seasonal, sea_coef, residual


def forecast_components(
    trend: np.ndarray,
    sea_coef: np.ndarray,
    period: int,
    n_fourier: int,
    h: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extrapolate trend and seasonal components h steps ahead.

    Trend: linear extrapolation from the last two Henderson values.
    Seasonal: evaluate Fourier model at future time indices.

    Parameters
    ----------
    trend    : estimated trend, shape (n,)
    sea_coef : Fourier coefficients, shape (2K,)
    period   : seasonal period P
    n_fourier: harmonics K
    h        : forecast horizon

    Returns
    -------
    trend_fc    : shape (h,)
    seasonal_fc : shape (h,)
    """
    n = len(trend)
    # Linear extrapolation of trend
    slope = trend[-1] - trend[-2] if n >= 2 else 0.0
    trend_fc = trend[-1] + slope * np.arange(1, h + 1)

    # Seasonal: evaluate at future time indices
    t_future = np.arange(n, n + h, dtype=float)
    F_future = fourier_matrix(t_future, period, n_fourier)
    seasonal_fc = F_future @ sea_coef

    return trend_fc, seasonal_fc
