"""
universitybox.survey
====================
Survey response synthesizer — generate realistic survey populations
from small real samples using Bayesian marginals, Gaussian Copula,
and NHOP-based oversampling.

Also includes a SurveyMonkey export parser that converts any SurveyMonkey
CSV or Excel export directly into a clean DataFrame + SurveySchema.

Quick start — from scratch
--------------------------
    import pandas as pd
    from universitybox.survey import SurveySchema, SurveySynthesizer

    schema = SurveySchema()
    schema.add_categorical("Preferred_Brand", categories=["Lenovo", "HP", "Dell"])
    schema.add_ordinal("Satisfaction", scale=(1, 5))
    schema.add_continuous("Age", bounds=(18, 65))

    synth = SurveySynthesizer(n_mcmc=500, random_state=42)
    synth.fit(real_responses_df, schema)
    population = synth.synthesize(N=2000)

Quick start — from SurveyMonkey export
----------------------------------------
    from universitybox.survey import SurveyMonkeyReader, SurveySynthesizer

    reader = SurveyMonkeyReader()
    reader.read_csv("my_survey_export.csv")   # or .read_excel(...)
    df, schema = reader.to_synthesizer_ready()

    synth = SurveySynthesizer(n_mcmc=500, random_state=42)
    synth.fit(df, schema)
    population = synth.synthesize(N=2000)

    # Optional no-code GUI (requires tkinter)
    from universitybox.survey import launch_gui
    launch_gui()
"""

from ._schema import SurveySchema, Question
from ._synthesizer import SurveySynthesizer
from ._surveymonkey import SurveyMonkeyReader
from ._viz import compare_plot


def launch_gui() -> None:
    """Open the Tkinter GUI for the Survey Synthesizer (no external deps)."""
    from ._gui import launch_gui as _launch
    _launch()


__all__ = [
    "SurveySchema",
    "Question",
    "SurveySynthesizer",
    "SurveyMonkeyReader",
    "compare_plot",
    "launch_gui",
]
