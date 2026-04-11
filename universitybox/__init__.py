"""
universitybox
=============

UniversityBox analytics toolkit.

Modules
-------
forecast    — Time series forecasting (DNA model and utilities)
segments    — Audience club segmentation
survey      — Survey response synthesizer

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

    # Survey synthesis
    from universitybox.survey import SurveySchema, SurveySynthesizer
    schema = SurveySchema()
    schema.add_ordinal("Satisfaction", scale=(1, 5))
    synth = SurveySynthesizer().fit(real_df, schema)
    population = synth.synthesize(N=1000)
"""

from .forecast import DNA
from .forecast import metrics
from .segments import Club
from .survey import SurveySchema, SurveySynthesizer

from .forecast._base import BaseForecaster

__version__ = "0.1.7"
__author__  = "UniversityBox Data Team"

__all__ = [
    "DNA",
    "Club",
    "metrics",
    "BaseForecaster",
    "SurveySchema",
    "SurveySynthesizer",
]
