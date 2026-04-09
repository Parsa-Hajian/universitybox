"""
Tests for the DNA forecaster.

Run with:  pytest tests/ -v
"""
import numpy as np
import pytest
from universitybox import DNA
from universitybox.forecast import metrics as m


# ── fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def quarterly_series():
    """Synthetic quarterly series: linear trend + seasonal + noise."""
    rng = np.random.default_rng(0)
    n = 40
    t = np.arange(n)
    trend = 100 + 2.5 * t
    seasonal = 20 * np.sin(2 * np.pi * t / 4)
    noise = rng.normal(0, 5, n)
    return trend + seasonal + noise


@pytest.fixture
def monthly_series():
    """Synthetic monthly series."""
    rng = np.random.default_rng(42)
    n = 60
    t = np.arange(n)
    trend = 500 + 1.5 * t
    seasonal = 80 * np.sin(2 * np.pi * t / 12)
    noise = rng.normal(0, 15, n)
    return trend + seasonal + noise


# ── smoke tests ───────────────────────────────────────────────────────

def test_fit_returns_self(quarterly_series):
    model = DNA(period=4, random_state=0)
    result = model.fit(quarterly_series)
    assert result is model


def test_forecast_shape(quarterly_series):
    model = DNA(period=4, random_state=0).fit(quarterly_series)
    fc = model.forecast(h=8)
    assert fc.shape == (8,)
    assert np.all(np.isfinite(fc))


def test_predict_interval_shape(quarterly_series):
    model = DNA(period=4, random_state=0).fit(quarterly_series)
    lo, hi = model.predict_interval(h=4, level=0.95)
    assert lo.shape == (4,)
    assert hi.shape == (4,)
    assert np.all(lo <= hi)


def test_fitted_values_shape(quarterly_series):
    model = DNA(period=4, random_state=0).fit(quarterly_series)
    fv = model.fitted_values
    assert fv.shape == quarterly_series.shape


def test_residuals_shape(quarterly_series):
    model = DNA(period=4, random_state=0).fit(quarterly_series)
    assert model.residuals.shape == quarterly_series.shape


def test_components_keys(quarterly_series):
    model = DNA(period=4, random_state=0).fit(quarterly_series)
    c = model.components
    for key in ("trend", "seasonal", "nonlinear", "adaptive"):
        assert key in c
        assert c[key].shape == quarterly_series.shape


def test_weights_sum_to_one(quarterly_series):
    model = DNA(period=4, random_state=0).fit(quarterly_series)
    w = model.weights
    assert abs(sum(w.values()) - 1.0) < 1e-6


def test_auto_period(quarterly_series):
    model = DNA(period="auto", random_state=0).fit(quarterly_series)
    assert model._period >= 2


def test_ensemble_equal(quarterly_series):
    model = DNA(period=4, ensemble="equal", random_state=0).fit(quarterly_series)
    for v in model.weights.values():
        assert abs(v - 0.25) < 1e-9


def test_ensemble_ols(quarterly_series):
    model = DNA(period=4, ensemble="ols", random_state=0).fit(quarterly_series)
    fc = model.forecast(4)
    assert np.all(np.isfinite(fc))


def test_bootstrap_interval(quarterly_series):
    model = DNA(period=4, ci_method="bootstrap",
                ci_bootstrap_n=100, random_state=0).fit(quarterly_series)
    lo, hi = model.predict_interval(h=4, level=0.90)
    assert np.all(lo <= hi)


def test_kalman_mle(quarterly_series):
    model = DNA(period=4, kalman_mle=True, random_state=0).fit(quarterly_series)
    fc = model.forecast(4)
    assert np.all(np.isfinite(fc))


def test_evaluate_returns_dict(quarterly_series):
    n = len(quarterly_series)
    train, test = quarterly_series[:n-4], quarterly_series[n-4:]
    model = DNA(period=4, random_state=0).fit(train)
    result = model.evaluate(test)
    for key in ("MAE", "RMSE", "MAPE", "sMAPE", "MASE"):
        assert key in result
        assert np.isfinite(result[key])


# ── input validation ──────────────────────────────────────────────────

def test_rejects_2d_input():
    with pytest.raises(ValueError, match="1-D"):
        DNA(period=4).fit(np.ones((10, 2)))


def test_rejects_nan_input():
    y = np.ones(20)
    y[5] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        DNA(period=4).fit(y)


def test_rejects_too_short():
    with pytest.raises(ValueError, match="at least 4"):
        DNA(period=4).fit(np.ones(3))


def test_forecast_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        DNA(period=4).forecast(4)


# ── metrics ───────────────────────────────────────────────────────────

def test_mae():
    assert m.mae([1, 2, 3], [1, 2, 3]) == 0.0
    assert m.mae([0, 0], [1, 1]) == 1.0


def test_rmse():
    assert m.rmse([0, 0], [1, 1]) == 1.0


def test_smape_symmetry():
    a = m.smape([100], [200])
    b = m.smape([200], [100])
    assert abs(a - b) < 1e-6


def test_mase_naïve():
    # Perfect forecast → MASE = 0
    y = np.arange(10, dtype=float)
    assert m.mase(y[1:], y[1:], y_train=y) == 0.0


def test_crps_perfect():
    from universitybox.forecast._metrics import crps_gaussian
    y = np.array([1.0, 2.0, 3.0])
    # Very tight Gaussian centred on truth
    score = crps_gaussian(y, mu=y, sigma=np.full(3, 1e-6))
    assert score < 0.01
