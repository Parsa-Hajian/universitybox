# Changelog

All notable changes to this project will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — 2026-04-09

### Added
- `DNA` — Dynamic Nonlinear Adaptive forecaster (D + N + A stages)
- `_decomposition` — Henderson moving average trend + Fourier seasonal OLS
- `_nonlinear` — k-means++ RBF feature map + Ridge regression
- `_adaptive` — Local Linear Trend Kalman filter with optional MLE noise estimation
- `_metrics` — MAE, RMSE, MAPE, sMAPE, MASE, CRPS (Gaussian closed-form)
- `BaseForecaster` — abstract sklearn-compatible interface
- `Club` — interest-based audience segmentation with configurable CTA threshold
- Ensemble combination: inverse-variance, OLS stacking, equal weights
- Prediction intervals: analytical Gaussian + bootstrap (empirical quantiles)
- Automatic period estimation via periodogram
- Full type annotations + `py.typed` marker
- 22 unit tests covering all stages, metrics, and edge cases
- `MATH.md` — complete mathematical derivations (PhD-level)
