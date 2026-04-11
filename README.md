# universitybox

**DNA — Dynamic Nonlinear Adaptive Time Series Forecaster**

[![PyPI](https://img.shields.io/pypi/v/universitybox)](https://pypi.org/project/universitybox/)
[![Python](https://img.shields.io/pypi/pyversions/universitybox)](https://pypi.org/project/universitybox/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/Parsa-Hajian/universitybox/blob/main/LICENSE)
[![Tests](https://img.shields.io/badge/tests-30%20passed-brightgreen)]()

A pure-NumPy/SciPy time series forecasting library built around the **DNA** model — a three-stage hierarchical forecaster combining classical decomposition, nonlinear basis expansion, and adaptive Kalman filtering.

No TensorFlow. No PyTorch. No black boxes. Every equation is documented.

**Full mathematical derivations:**
[MATH.md on GitHub](https://github.com/Parsa-Hajian/universitybox/blob/main/MATH.md)

---

## Install

```bash
pip install universitybox
```

With optional extras:

```bash
pip install "universitybox[full]"   # + pandas + matplotlib
pip install "universitybox[viz]"    # + matplotlib only
pip install "universitybox[data]"   # + pandas only
```

---

## Quick start

```python
import numpy as np
from universitybox import DNA

# Any 1-D time series
y = np.array([112, 118, 132, 129, 121, 135, 148, 148, 136, 119,
              104, 118, 115, 126, 141, 135, 125, 149, 170, 170,
              158, 133, 114, 140, 145, 150, 178, 163, 172, 178,
              199, 199, 184, 162, 146, 166, 171, 180, 193, 181])

model = DNA(period=12)
model.fit(y)

point_forecast      = model.forecast(h=12)
lower, upper        = model.predict_interval(h=12, level=0.95)
test_metrics        = model.evaluate(y[-6:])

model.summary()
```

---

## The DNA Model

DNA decomposes the time series into three progressively finer layers:

```
y(t) = mu(t)          Stage D — trend,    Henderson moving average
     + s(t)           Stage D — seasonal, Fourier OLS
     + f(Phi(x_t))    Stage N — nonlinear correction, Ridge + RBF features
     + l(t)           Stage A — adaptive correction, Kalman LLT filter
     + noise
```

Each stage is fitted on the residual of the previous stage.
The final forecast is an inverse-variance weighted combination of all four components.

### Stage D — Decomposition

- **Trend:** Henderson symmetric moving average of half-length `m`.
  Weights minimise the third-difference roughness of the trend while
  exactly reproducing polynomials up to degree 2 (closed-form, Doherty 2001).
- **Seasonal:** Fourier regression of order `K` on the detrended series,
  estimated by OLS. Normalised to zero mean over each complete period.
- **Period:** Auto-estimated from the periodogram when `period='auto'`.

### Stage N — Nonlinear Basis Expansion

Feature map composed of three dictionaries:

```
Phi(t) = [ polynomial(t)  |  AR lags of residual  |  RBF kernels ]
```

- **Polynomial:** time index normalised to [0,1], up to degree `p`.
- **AR lags:** standardised lagged residuals (lags 1 to L).
- **RBF:** squared-exponential kernels centred by **k-means++ seeding**,
  bandwidth set by the **median heuristic**.

Ridge regression (L2 regularised least squares, closed-form Cholesky solve)
fits the feature map to the D-stage residual.

### Stage A — Adaptive Kalman Filter

Local Linear Trend (LLT) state-space model on the N-stage residual:

```
State:       x(t) = [level, slope]
Transition:  x(t) = F x(t-1) + noise_process
Observation: r(t) = H x(t)  + noise_obs
```

Kalman filter recursion (prediction → innovation → gain → update).
Optional MLE estimation of noise parameters (`kalman_mle=True`).
h-step forecast: level(n) + h * slope(n).

### Ensemble Combination

Component forecasts combined as:

```
y_hat(n+h) = alpha * trend(n+h)
           + beta  * seasonal(n+h)
           + gamma * nonlinear(n+h)
           + delta * adaptive(n+h)
```

Weights computed by:
- `'iv'`    — inverse-variance (Bates & Granger 1969), default
- `'ols'`   — non-negative least-squares stacking
- `'equal'` — 0.25 each

### Prediction Intervals

- **Analytical:** `forecast ± z * sigma * sqrt(h)`, where sigma is the in-sample RMSE.
- **Bootstrap:** resample in-sample residuals B times, propagate as cumulative shocks, take empirical quantiles.

---

## All parameters

```python
DNA(
    period         = "auto",  # int or 'auto' — seasonal period (4, 12, 7, ...)
    trend_window   = "auto",  # Henderson filter half-length m
    n_fourier      = 3,       # Fourier harmonics K
    poly_degree    = 2,       # polynomial degree in N-stage feature map
    n_lags         = 4,       # AR lag count in N-stage feature map
    n_rbf          = 10,      # RBF centres (k-means++ selected)
    rbf_gamma      = "auto",  # RBF bandwidth ('auto' = median heuristic)
    ridge_alpha    = 1e-3,    # L2 regularisation lambda
    kalman_q_level = 1e-4,    # Kalman level process noise
    kalman_q_slope = 1e-6,    # Kalman slope process noise
    kalman_obs_var = 1e-2,    # Kalman observation noise
    kalman_mle     = False,   # estimate Kalman noise by MLE
    ensemble       = "iv",    # 'iv' | 'equal' | 'ols'
    ci_method      = "analytical",  # 'analytical' | 'bootstrap'
    ci_bootstrap_n = 500,     # bootstrap replications
    random_state   = None,    # reproducibility seed
)
```

---

## API reference

| Method / Property | Description |
|-------------------|-------------|
| `fit(y)` | Fit the model. y must be 1-D, finite, length >= 4. |
| `forecast(h)` | Point forecasts for horizons 1 to h. Returns array of shape (h,). |
| `predict_interval(h, level=0.95)` | Returns (lower, upper) arrays of shape (h,). |
| `evaluate(y_test)` | MAE, RMSE, MAPE, sMAPE, MASE vs held-out test set. |
| `fitted_values` | In-sample fitted values, shape (n,). |
| `residuals` | In-sample residuals y - y_hat, shape (n,). |
| `components` | Dict: trend, seasonal, nonlinear, adaptive — each shape (n,). |
| `weights` | Dict: alpha, beta, gamma, delta ensemble weights. |
| `summary()` | Print model card. |

---

## Metrics

```python
from universitybox import metrics

metrics.mae(y_true, y_pred)
metrics.rmse(y_true, y_pred)
metrics.mape(y_true, y_pred)
metrics.smape(y_true, y_pred)
metrics.mase(y_true, y_pred, y_train=y_train, period=4)
metrics.crps_gaussian(y_true, mu=fc, sigma=sigma_h)
metrics.summary(y_true, y_pred)          # dict of all metrics
```

---

## Survey Response Synthesizer

Have a small set of real survey responses (30–200)? Generate a large synthetic population that preserves the same distributions and cross-question correlations.

```python
import pandas as pd
from universitybox.survey import SurveySchema, SurveySynthesizer

# Define the survey structure — specify scale per question
schema = SurveySchema()
schema.add_categorical("Preferred_Brand", categories=["Lenovo", "HP", "Dell"])
schema.add_ordinal("Overall_Satisfaction", scale=(1, 5))
schema.add_ordinal("Likelihood_to_Recommend", scale=(1, 7))
schema.add_continuous("Age", bounds=(18, 65))

# Fit on real responses, generate synthetic population
real_df = pd.read_csv("my_survey_responses.csv")   # e.g. 50 real rows

synth = SurveySynthesizer(n_mcmc=500, n_seeds=10, random_state=42)
synth.fit(real_df, schema)
population = synth.synthesize(N=2000)   # returns DataFrame of 2000 rows
```

### How it works

**Stage 1 — Bayesian per-question estimation** (handles small N without overfitting):
- Categorical: Dirichlet-Multinomial with Jeffreys prior
- Ordinal/Likert: Bayesian Ordinal Probit via Gibbs sampler (Albert & Chib 1993)
- Continuous: Normal-Inverse-Gamma conjugate model

**Stage 2 — Gaussian Copula** preserves cross-question correlations:
rank-based CDF transform → probit scores → Ledoit-Wolf regularised correlation → MVN draw → back-map via quantile functions.

**Stage 3 — NHOP oversampling**:
- k-means++ seed selection to anchor coverage
- NHOP rejection: discard synthesised points too far from the real sample (avoids hallucinated response patterns)
- Density-proportional resampling to reach target N

### Optional no-code GUI

```python
from universitybox.survey import launch_gui
launch_gui()   # opens Tkinter window — no external dependencies
```

Three tabs: Schema Builder / Data Input / Synthesize & Export.

---

## Audience segmentation (Club)

```python
from universitybox.segments import Club

category_map = {
    "lenovo":   "Technology",
    "hp store": "Technology",
    "samsung":  "Technology",
    "zara":     "Fashion",
}

club = Club(category_map=category_map, min_cta=6)
club.fit(events_df)          # DataFrame: user_id, brand, cta_count

club.size("Technology")      # int — number of members
club.share("Technology")     # float — fraction of classified users
club.members("Technology")   # list of user_ids
club.summary()               # dict: all clubs with size + share
```

---

## Extending DNA — adding a custom forecaster

All forecasters extend `BaseForecaster`:

```python
from universitybox.forecast._base import BaseForecaster
import numpy as np

class MyForecaster(BaseForecaster):
    def __init__(self, alpha=0.3):
        self.alpha = alpha

    def fit(self, y: np.ndarray, **kwargs) -> "MyForecaster":
        y = self._validate_y(y)
        # ... fit logic ...
        self._insample_rmse = float(np.std(y))
        return self

    def forecast(self, h: int) -> np.ndarray:
        # ... return array of shape (h,) ...
        ...
```

`predict_interval()` and `score()` are provided by the base class automatically.

---

## Mathematical documentation

Full derivations for every formula in the package:
[MATH.md](https://github.com/Parsa-Hajian/universitybox/blob/main/MATH.md), covering:

1. Notation and problem statement
2. The DNA decomposition equation
3. Stage D — Henderson filter weights (closed-form), Fourier OLS, periodogram period estimation
4. Stage N — polynomial / AR-lag / RBF feature map, k-means++ seed selection, median heuristic bandwidth, Ridge regression (primal & dual), RKHS interpretation
5. Stage A — LLT state-space model, Kalman filter recursion (all equations), h-step forecast, MLE log-likelihood
6. Ensemble combination — inverse-variance weighting, OLS stacking
7. Prediction intervals — analytical Gaussian and bootstrap
8. Evaluation metrics — MAE, RMSE, MAPE, sMAPE, MASE, CRPS (Gaussian closed-form)
9. Identifiability and consistency proofs
10. Computational complexity table
11. Full bibliography

---

## Design principles

- **Pure NumPy/SciPy** — no heavy ML framework required
- **Minimal core dependencies** — only `numpy` and `scipy`
- **sklearn-compatible** — `fit` / `forecast` / `score` interface
- **Fully typed** — `py.typed` marker, complete type annotations
- **22 unit tests** — all components, edge cases, and metrics covered

---

## Contributing

```bash
git clone https://github.com/Parsa-Hajian/universitybox
cd universitybox
pip install -e ".[dev]"
pytest tests/ -v
```

See [CONTRIBUTING.md](https://github.com/Parsa-Hajian/universitybox/blob/main/CONTRIBUTING.md).

---

## Citation

```bibtex
@software{universitybox2026,
  author  = {UniversityBox Data Team},
  title   = {universitybox: DNA Dynamic Nonlinear Adaptive Forecaster},
  year    = {2026},
  url     = {https://github.com/Parsa-Hajian/universitybox},
  version = {0.1.2}
}
```

---

## License

MIT — see [LICENSE](https://github.com/Parsa-Hajian/universitybox/blob/main/LICENSE).
