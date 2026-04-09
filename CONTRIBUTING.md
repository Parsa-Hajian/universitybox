# Contributing to universitybox

Thank you for your interest in contributing!

## Setup

```bash
git clone https://github.com/universitybox/universitybox-pkg
cd universitybox-pkg
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/ -v
```

## Adding a new forecaster

All forecasters must extend `BaseForecaster`:

```python
from universitybox.forecast._base import BaseForecaster
import numpy as np

class MyForecaster(BaseForecaster):
    def __init__(self, my_param=1.0):
        self.my_param = my_param

    def fit(self, y: np.ndarray, **kwargs) -> "MyForecaster":
        y = self._validate_y(y)
        # ... fit logic ...
        self._insample_rmse = float(np.std(y - self.fitted_values))
        return self

    def forecast(self, h: int) -> np.ndarray:
        # ... return shape (h,) array ...
        ...
```

Implement `get_params()` if your constructor does not use default arguments.
`predict_interval()`, `score()` are provided by the base class automatically.

## Code style

- No external dependencies beyond `numpy` and `scipy` in the core package
- All mathematical formulae must have a corresponding equation in a docstring comment and, for significant new methods, in `MATH.md`
- Every public function must have a type-annotated signature
- Every new feature needs at least one test in `tests/`

## Publishing to PyPI

```bash
pip install build twine
python -m build
twine upload dist/*
```

## Pull requests

- One feature or fix per PR
- Include tests
- Update `CHANGELOG.md`
