# Changelog

All notable changes to this project will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.5] — 2026-04-08

### Added
- `universitybox.survey` — new Survey Response Synthesizer submodule
  - `SurveySchema` — define mixed-type survey questions (categorical, ordinal, continuous) with per-question scale specification
  - `SurveySynthesizer` — three-stage pipeline: Bayesian marginals (Dirichlet-Multinomial, Ordinal Probit Gibbs, NIG) + Gaussian Copula + NHOP oversampling
  - Bayesian Ordinal Probit sampler (Albert & Chib 1993) for Likert-scale questions
  - Dirichlet-Multinomial with Jeffreys prior for categorical questions
  - Normal-Inverse-Gamma conjugate model for continuous questions
  - Gaussian Copula with Ledoit-Wolf shrinkage for cross-question correlation preservation
  - k-means++ seed selection for representative prototype selection
  - NHOP (Neighbourhood-based Outlier Pruning) rejection to keep synthesised responses in-distribution
  - Density-proportional oversampling to reach target N
  - Optional Tkinter GUI (`launch_gui()`) — Schema Builder, Data Input, Synthesize & Export tabs
  - No new dependencies (pandas optional, tkinter is stdlib)

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
