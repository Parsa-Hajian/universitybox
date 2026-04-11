"""
Tests for universitybox.survey
================================
Covers:
  - SurveySchema construction and validation
  - CategoricalMarginal, OrdinalMarginal, ContinuousMarginal
  - GaussianCopula
  - SurveySynthesizer end-to-end
"""
import numpy as np
import pandas as pd
import pytest

from universitybox.survey import SurveySchema, SurveySynthesizer
from universitybox.survey._marginals import (
    CategoricalMarginal,
    OrdinalMarginal,
    ContinuousMarginal,
)
from universitybox.survey._copula import GaussianCopula


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def simple_schema():
    schema = SurveySchema()
    schema.add_categorical("Brand", categories=["A", "B", "C"])
    schema.add_ordinal("Score", scale=(1, 5))
    schema.add_continuous("Age", bounds=(18.0, 65.0))
    return schema


@pytest.fixture
def small_real_df(simple_schema):
    rng = np.random.default_rng(0)
    n = 40
    brands = rng.choice(["A", "B", "C"], size=n)
    scores = rng.integers(1, 6, size=n)
    ages = rng.uniform(18, 65, size=n)
    return pd.DataFrame({"Brand": brands, "Score": scores, "Age": ages})


# ---------------------------------------------------------------------------
# SurveySchema
# ---------------------------------------------------------------------------

class TestSurveySchema:
    def test_add_categorical(self):
        schema = SurveySchema()
        schema.add_categorical("Q1", categories=["X", "Y"])
        assert len(schema) == 1
        assert schema.questions[0].qtype == "categorical"

    def test_add_ordinal(self):
        schema = SurveySchema()
        schema.add_ordinal("Q1", scale=(1, 7))
        q = schema.questions[0]
        assert q.scale == (1, 7)
        assert q.qtype == "ordinal"

    def test_add_continuous(self):
        schema = SurveySchema()
        schema.add_continuous("Age", bounds=(0.0, 100.0))
        q = schema.questions[0]
        assert q.bounds == (0.0, 100.0)

    def test_chaining(self):
        schema = SurveySchema()
        result = schema.add_categorical("A", ["x", "y"]).add_ordinal("B", (1, 5))
        assert result is schema
        assert len(schema) == 2

    def test_categorical_needs_two_categories(self):
        with pytest.raises(ValueError):
            SurveySchema().add_categorical("Q", categories=["only_one"])

    def test_ordinal_needs_scale(self):
        with pytest.raises(TypeError):
            SurveySchema().add_ordinal("Q")  # missing scale

    def test_ordinal_bad_scale(self):
        with pytest.raises(ValueError):
            SurveySchema().add_ordinal("Q", scale=(5, 1))  # max < min

    def test_continuous_bad_bounds(self):
        with pytest.raises(ValueError):
            SurveySchema().add_continuous("Q", bounds=(100.0, 0.0))

    def test_validate_dataframe_missing_col(self, simple_schema):
        df = pd.DataFrame({"Brand": ["A"], "Score": [3]})  # missing Age
        with pytest.raises(ValueError, match="missing"):
            simple_schema.validate_dataframe(df)

    def test_validate_dataframe_bad_category(self, simple_schema):
        df = pd.DataFrame({"Brand": ["Z"], "Score": [3], "Age": [25.0]})
        with pytest.raises(ValueError, match="unknown categories"):
            simple_schema.validate_dataframe(df)

    def test_validate_dataframe_ordinal_out_of_range(self, simple_schema):
        df = pd.DataFrame({"Brand": ["A"], "Score": [9], "Age": [25.0]})
        with pytest.raises(ValueError, match="outside scale"):
            simple_schema.validate_dataframe(df)

    def test_validate_dataframe_continuous_out_of_bounds(self, simple_schema):
        df = pd.DataFrame({"Brand": ["A"], "Score": [3], "Age": [200.0]})
        with pytest.raises(ValueError, match="outside bounds"):
            simple_schema.validate_dataframe(df)

    def test_repr(self, simple_schema):
        r = repr(simple_schema)
        assert "categorical" in r
        assert "ordinal" in r
        assert "continuous" in r


# ---------------------------------------------------------------------------
# CategoricalMarginal
# ---------------------------------------------------------------------------

class TestCategoricalMarginal:
    def test_sample_returns_valid_categories(self, rng):
        cats = ["Rome", "Milan", "Naples"]
        m = CategoricalMarginal(cats)
        col = np.array(["Rome", "Rome", "Milan", "Naples", "Rome"])
        m.fit(col)
        samples = m.sample(200, rng)
        assert set(samples).issubset(set(cats))

    def test_cdf_in_unit_interval(self, rng):
        cats = ["A", "B", "C"]
        m = CategoricalMarginal(cats).fit(np.array(["A", "A", "B", "C", "B"]))
        u = m.cdf(np.array(["A", "B", "C"]))
        assert np.all(u >= 0) and np.all(u <= 1)

    def test_quantile_inverse_of_cdf(self, rng):
        cats = ["X", "Y"]
        m = CategoricalMarginal(cats).fit(np.array(["X", "Y", "X"]))
        u = m.cdf(np.array(["X", "Y"]))
        recovered = m.quantile(u)
        assert list(recovered) == ["X", "Y"]


# ---------------------------------------------------------------------------
# OrdinalMarginal
# ---------------------------------------------------------------------------

class TestOrdinalMarginal:
    def test_sample_in_scale(self, rng):
        col = np.array([1, 2, 3, 4, 5, 3, 3, 2, 4, 1])
        m = OrdinalMarginal(scale=(1, 5), n_mcmc=200).fit(col, rng)
        samples = m.sample(300, rng)
        assert samples.dtype == int
        assert samples.min() >= 1
        assert samples.max() <= 5

    def test_cdf_monotone(self, rng):
        col = np.array([1, 2, 2, 3, 3, 3, 4, 4, 5])
        m = OrdinalMarginal(scale=(1, 5), n_mcmc=100).fit(col, rng)
        u1 = m.cdf(np.array([1]))
        u5 = m.cdf(np.array([5]))
        assert u1[0] < u5[0]

    def test_quantile_in_scale(self, rng):
        col = np.array([1, 2, 3, 4, 5, 3])
        m = OrdinalMarginal(scale=(1, 5), n_mcmc=100).fit(col, rng)
        q = m.quantile(np.array([0.1, 0.5, 0.9]))
        assert np.all(q >= 1) and np.all(q <= 5)

    def test_wider_scale(self, rng):
        col = np.array([1, 3, 5, 7, 7, 4, 2, 6])
        m = OrdinalMarginal(scale=(1, 7), n_mcmc=100).fit(col, rng)
        samples = m.sample(100, rng)
        assert samples.min() >= 1
        assert samples.max() <= 7


# ---------------------------------------------------------------------------
# ContinuousMarginal
# ---------------------------------------------------------------------------

class TestContinuousMarginal:
    def test_sample_in_bounds(self):
        col = np.random.default_rng(1).uniform(18, 65, 50)
        m = ContinuousMarginal(bounds=(18.0, 65.0)).fit(col)
        samples = m.sample(500, np.random.default_rng(2))
        assert samples.min() >= 18.0
        assert samples.max() <= 65.0

    def test_cdf_in_unit_interval(self):
        col = np.linspace(20, 60, 20)
        m = ContinuousMarginal(bounds=(18.0, 65.0)).fit(col)
        u = m.cdf(col)
        assert np.all(u >= 0) and np.all(u <= 1)

    def test_quantile_monotone(self):
        col = np.linspace(20, 60, 30)
        m = ContinuousMarginal(bounds=(18.0, 65.0)).fit(col)
        u = np.array([0.1, 0.5, 0.9])
        q = m.quantile(u)
        assert q[0] < q[1] < q[2]


# ---------------------------------------------------------------------------
# GaussianCopula
# ---------------------------------------------------------------------------

class TestGaussianCopula:
    def test_copula_output_shape(self, small_real_df, simple_schema, rng):
        marginals = []
        for q in simple_schema.questions:
            col = small_real_df[q.name].values
            if q.qtype == "categorical":
                m = CategoricalMarginal(q.categories).fit(col)
            elif q.qtype == "ordinal":
                m = OrdinalMarginal(q.scale, n_mcmc=100).fit(col, rng)
            else:
                m = ContinuousMarginal(q.bounds).fit(col)
            marginals.append(m)

        copula = GaussianCopula().fit(small_real_df[simple_schema.names], marginals)
        out = copula.synthesise(50, marginals, rng)
        assert out.shape == (50, 3)

    def test_correlation_matrix_is_positive_definite(self, small_real_df, simple_schema, rng):
        marginals = []
        for q in simple_schema.questions:
            col = small_real_df[q.name].values
            if q.qtype == "categorical":
                m = CategoricalMarginal(q.categories).fit(col)
            elif q.qtype == "ordinal":
                m = OrdinalMarginal(q.scale, n_mcmc=100).fit(col, rng)
            else:
                m = ContinuousMarginal(q.bounds).fit(col)
            marginals.append(m)

        copula = GaussianCopula().fit(small_real_df[simple_schema.names], marginals)
        eigvals = np.linalg.eigvalsh(copula._R)
        assert np.all(eigvals > 0)


# ---------------------------------------------------------------------------
# SurveySynthesizer — end-to-end
# ---------------------------------------------------------------------------

class TestSurveySynthesizer:
    def test_output_shape(self, small_real_df, simple_schema):
        synth = SurveySynthesizer(n_mcmc=100, random_state=0)
        synth.fit(small_real_df, simple_schema)
        pop = synth.synthesize(N=200)
        assert pop.shape == (200, 3)

    def test_column_names_preserved(self, small_real_df, simple_schema):
        synth = SurveySynthesizer(n_mcmc=100, random_state=0)
        synth.fit(small_real_df, simple_schema)
        pop = synth.synthesize(N=50)
        assert list(pop.columns) == ["Brand", "Score", "Age"]

    def test_ordinal_values_in_range(self, small_real_df, simple_schema):
        synth = SurveySynthesizer(n_mcmc=100, random_state=1)
        synth.fit(small_real_df, simple_schema)
        pop = synth.synthesize(N=300)
        assert pop["Score"].min() >= 1
        assert pop["Score"].max() <= 5

    def test_continuous_values_in_bounds(self, small_real_df, simple_schema):
        synth = SurveySynthesizer(n_mcmc=100, random_state=2)
        synth.fit(small_real_df, simple_schema)
        pop = synth.synthesize(N=300)
        assert pop["Age"].min() >= 18.0
        assert pop["Age"].max() <= 65.0

    def test_categorical_values_valid(self, small_real_df, simple_schema):
        synth = SurveySynthesizer(n_mcmc=100, random_state=3)
        synth.fit(small_real_df, simple_schema)
        pop = synth.synthesize(N=200)
        assert set(pop["Brand"]).issubset({"A", "B", "C"})

    def test_requires_fit_before_synthesize(self, simple_schema):
        synth = SurveySynthesizer()
        with pytest.raises(RuntimeError, match="fit"):
            synth.synthesize(N=100)

    def test_requires_at_least_4_rows(self, simple_schema):
        df = pd.DataFrame({"Brand": ["A", "B"], "Score": [1, 2], "Age": [20.0, 30.0]})
        synth = SurveySynthesizer(n_mcmc=50)
        with pytest.raises(ValueError, match="4"):
            synth.fit(df, simple_schema)

    def test_only_ordinal_schema(self):
        schema = SurveySchema()
        schema.add_ordinal("Q1", scale=(1, 5))
        schema.add_ordinal("Q2", scale=(1, 7))
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "Q1": rng.integers(1, 6, 30),
            "Q2": rng.integers(1, 8, 30),
        })
        synth = SurveySynthesizer(n_mcmc=100, random_state=0)
        synth.fit(df, schema)
        pop = synth.synthesize(N=100)
        assert pop["Q1"].between(1, 5).all()
        assert pop["Q2"].between(1, 7).all()

    def test_only_categorical_schema(self):
        schema = SurveySchema()
        schema.add_categorical("City", categories=["Rome", "Milan", "Naples"])
        schema.add_categorical("Gender", categories=["M", "F", "Other"])
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "City": rng.choice(["Rome", "Milan", "Naples"], 30),
            "Gender": rng.choice(["M", "F", "Other"], 30),
        })
        synth = SurveySynthesizer(n_mcmc=100, random_state=0)
        synth.fit(df, schema)
        pop = synth.synthesize(N=100)
        assert set(pop["City"]).issubset({"Rome", "Milan", "Naples"})
        assert set(pop["Gender"]).issubset({"M", "F", "Other"})

    def test_summary_runs(self, small_real_df, simple_schema, capsys):
        synth = SurveySynthesizer(n_mcmc=100, random_state=0)
        synth.fit(small_real_df, simple_schema)
        synth.summary()
        out = capsys.readouterr().out
        assert "SurveySynthesizer" in out

    def test_top_level_import(self):
        from universitybox import SurveySchema, SurveySynthesizer
        assert SurveySchema is not None
        assert SurveySynthesizer is not None
