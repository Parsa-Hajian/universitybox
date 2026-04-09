"""
universitybox
=============

UniversityBox analytics toolkit.

Modules
-------
forecast    — Time series forecasting (DNA model and utilities)
segments    — Audience club segmentation

Quick start
-----------
    # Forecasting
    from universitybox import DNA
    model = DNA(period=4).fit(y)
    fc = model.forecast(h=4)

    # Or via submodule
    from universitybox.forecast import DNA

    # Segmentation
    from universitybox.segments import Club
"""

from .forecast import DNA
from .forecast import metrics
from .segments import Club

from .forecast._base import BaseForecaster

__version__ = "0.1.1"
__author__  = "UniversityBox Data Team"

__all__ = [
    "DNA",
    "Club",
    "metrics",
    "BaseForecaster",
]
