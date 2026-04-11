"""
Tests for universitybox.survey.SurveyMonkeyReader
===================================================
Simulates all real SurveyMonkey export scenarios in-memory
without needing actual CSV files on disk.
"""
import io
import textwrap

import numpy as np
import pandas as pd
import pytest

from universitybox.survey import SurveyMonkeyReader, SurveySynthesizer


# ---------------------------------------------------------------------------
# Helpers: build raw CSV strings and parse them
# ---------------------------------------------------------------------------

def _csv_to_reader(csv_text: str, **kwargs) -> SurveyMonkeyReader:
    """Parse a CSV string as if it were a SurveyMonkey export file."""
    raw = pd.read_csv(
        io.StringIO(textwrap.dedent(csv_text)),
        header=None, dtype=str, keep_default_na=False,
        on_bad_lines="skip", engine="python",
    )
    return SurveyMonkeyReader(**kwargs).parse(raw)


# ---------------------------------------------------------------------------
# Scenario A — 2-row header, Condensed, numeric ordinal values
# ---------------------------------------------------------------------------

CONDENSED_CSV = """\
Respondent ID,Collector ID,Start Date,End Date,IP Address,How satisfied are you?,How likely to recommend?,Age
,,,,,,,
1001,C1,2024-01-01,2024-01-01,1.2.3.4,4,8,28
1002,C1,2024-01-01,2024-01-01,1.2.3.4,3,7,35
1003,C1,2024-01-01,2024-01-01,1.2.3.4,5,10,22
1004,C1,2024-01-01,2024-01-01,1.2.3.4,2,6,40
1005,C1,2024-01-01,2024-01-01,1.2.3.4,4,9,31
1006,C1,2024-01-01,2024-01-01,1.2.3.4,5,10,29
1007,C1,2024-01-01,2024-01-01,1.2.3.4,3,8,44
1008,C1,2024-01-01,2024-01-01,1.2.3.4,1,5,55
1009,C1,2024-01-01,2024-01-01,1.2.3.4,4,7,38
1010,C1,2024-01-01,2024-01-01,1.2.3.4,5,9,27
"""


class TestScenarioA_Condensed:
    def test_parses_without_error(self):
        reader = _csv_to_reader(CONDENSED_CSV)
        assert reader.clean_df is not None

    def test_metadata_stripped(self):
        reader = _csv_to_reader(CONDENSED_CSV)
        cols = reader.clean_df.columns.tolist()
        for meta in ["Respondent ID", "Collector ID", "Start Date", "End Date", "IP Address"]:
            assert meta not in cols

    def test_ordinal_detected(self):
        reader = _csv_to_reader(CONDENSED_CSV)
        schema = reader.schema
        names = schema.names
        assert "How satisfied are you?" in names
        q_sat = next(q for q in schema.questions if "satisfied" in q.name)
        assert q_sat.qtype == "ordinal"
        assert q_sat.scale == (1, 5)

    def test_nps_detected_as_ordinal(self):
        reader = _csv_to_reader(CONDENSED_CSV)
        q_nps = next(q for q in reader.schema.questions if "recommend" in q.name)
        assert q_nps.qtype == "ordinal"
        assert q_nps.scale == (5, 10)

    def test_continuous_detected(self):
        reader = _csv_to_reader(CONDENSED_CSV)
        q_age = next(q for q in reader.schema.questions if "Age" in q.name)
        assert q_age.qtype == "continuous"
        assert q_age.bounds[0] >= 18

    def test_row_count(self):
        reader = _csv_to_reader(CONDENSED_CSV)
        assert len(reader.clean_df) == 10


# ---------------------------------------------------------------------------
# Scenario B — Likert text labels (5-point agreement scale)
# ---------------------------------------------------------------------------

LIKERT_CSV = """\
Respondent ID,Start Date,The product meets my needs,I would buy again,Overall impression
,,,,
1001,2024-01-01,Strongly Agree,Agree,Excellent
1002,2024-01-01,Agree,Strongly Agree,Good
1003,2024-01-01,Neutral,Neutral,Average
1004,2024-01-01,Disagree,Disagree,Poor
1005,2024-01-01,Strongly Disagree,Strongly Disagree,Very Poor
1006,2024-01-01,Agree,Agree,Good
1007,2024-01-01,Strongly Agree,Neutral,Excellent
1008,2024-01-01,Neutral,Agree,Average
1009,2024-01-01,Disagree,Neutral,Poor
1010,2024-01-01,Strongly Agree,Strongly Agree,Excellent
"""


class TestScenarioB_LikertText:
    def test_likert_mapped_to_ordinal(self):
        reader = _csv_to_reader(LIKERT_CSV)
        q = next(q for q in reader.schema.questions if "meets" in q.name)
        assert q.qtype == "ordinal"
        assert q.scale == (1, 5)

    def test_likert_values_are_integers(self):
        reader = _csv_to_reader(LIKERT_CSV)
        col = reader.clean_df["The product meets my needs"]
        assert col.dtype in (int, np.int64, "int64", object)
        numeric = pd.to_numeric(col, errors="coerce").dropna()
        assert numeric.between(1, 5).all()

    def test_quality_scale_detected(self):
        # "Excellent/Good/Average/Poor/Very Poor" should map to ordinal
        reader = _csv_to_reader(LIKERT_CSV)
        q = next((q for q in reader.schema.questions if "impression" in q.name), None)
        assert q is not None
        assert q.qtype == "ordinal"

    def test_schema_has_three_columns(self):
        reader = _csv_to_reader(LIKERT_CSV)
        assert len(reader.schema) == 3


# ---------------------------------------------------------------------------
# Scenario C — Expanded multi-select ("select all that apply")
# ---------------------------------------------------------------------------

MULTISELECT_CSV = """\
Respondent ID,Start Date,Which brands do you use?,Which brands do you use?,Which brands do you use?,Satisfaction
,,Lenovo,HP,Dell,
1001,2024-01-01,Lenovo,,,4
1002,2024-01-01,,HP,,4
1003,2024-01-01,Lenovo,HP,,3
1004,2024-01-01,,,Dell,5
1005,2024-01-01,Lenovo,,Dell,4
1006,2024-01-01,,HP,,5
1007,2024-01-01,Lenovo,HP,Dell,3
1008,2024-01-01,Lenovo,,,4
1009,2024-01-01,,HP,,5
1010,2024-01-01,Lenovo,,Dell,3
"""


class TestScenarioC_MultiSelect:
    def test_binary_columns_created(self):
        reader = _csv_to_reader(MULTISELECT_CSV, multiselect_as_binary=True)
        cols = reader.clean_df.columns.tolist()
        # Should have binary Yes/No columns for each option
        brand_cols = [c for c in cols if "brands" in c.lower()]
        assert len(brand_cols) == 3

    def test_binary_values_yes_no(self):
        reader = _csv_to_reader(MULTISELECT_CSV, multiselect_as_binary=True)
        for col in reader.clean_df.columns:
            if "brands" in col.lower():
                vals = reader.clean_df[col].dropna().unique()
                assert set(vals).issubset({"Yes", "No"})

    def test_binary_as_categorical(self):
        reader = _csv_to_reader(MULTISELECT_CSV, multiselect_as_binary=True)
        for q in reader.schema.questions:
            if "brands" in q.name.lower():
                assert q.qtype == "categorical"
                assert set(q.categories).issubset({"Yes", "No"})

    def test_satisfaction_also_parsed(self):
        reader = _csv_to_reader(MULTISELECT_CSV, multiselect_as_binary=True)
        q_names = reader.schema.names
        assert any("Satisfaction" in n for n in q_names)


# ---------------------------------------------------------------------------
# Scenario D — Matrix question (rating grid)
# ---------------------------------------------------------------------------

MATRIX_CSV = """\
Respondent ID,Start Date,Please rate each aspect:,Please rate each aspect:,Please rate each aspect:,Overall
,,Speed,Quality,Support,
1001,2024-01-01,5,4,3,5
1002,2024-01-01,3,5,4,4
1003,2024-01-01,4,3,5,3
1004,2024-01-01,2,4,3,2
1005,2024-01-01,5,5,5,5
1006,2024-01-01,1,2,3,2
1007,2024-01-01,4,4,4,4
1008,2024-01-01,3,3,2,3
1009,2024-01-01,5,4,3,4
1010,2024-01-01,2,5,4,3
"""


class TestScenarioD_Matrix:
    def test_matrix_rows_as_separate_columns(self):
        reader = _csv_to_reader(MATRIX_CSV)
        col_names = reader.clean_df.columns.tolist()
        # Each matrix row → separate column
        assert any("Speed" in c for c in col_names)
        assert any("Quality" in c for c in col_names)
        assert any("Support" in c for c in col_names)

    def test_matrix_rows_ordinal(self):
        reader = _csv_to_reader(MATRIX_CSV)
        for q in reader.schema.questions:
            if any(label in q.name for label in ["Speed", "Quality", "Support"]):
                assert q.qtype == "ordinal"
                assert q.scale[0] >= 1
                assert q.scale[1] <= 5

    def test_overall_column_present(self):
        reader = _csv_to_reader(MATRIX_CSV)
        assert any("Overall" in n for n in reader.schema.names)


# ---------------------------------------------------------------------------
# Scenario E — Open-ended text columns are skipped
# ---------------------------------------------------------------------------

OPENENDED_CSV = """\
Respondent ID,Start Date,Satisfaction,Please describe your experience in detail,Age
,,,,
1001,2024-01-01,4,I really enjoyed using this product it was very helpful for my workflow,30
1002,2024-01-01,3,It was okay but could use some improvements in the user interface design,25
1003,2024-01-01,5,Absolutely fantastic product I would highly recommend it to all my colleagues,40
1004,2024-01-01,2,Not great the product kept crashing and customer support was unhelpful overall,35
1005,2024-01-01,4,Good product overall with minor issues that were quickly resolved by support team,28
1006,2024-01-01,5,Loved every aspect especially the speed and reliability of the platform,45
1007,2024-01-01,3,Average experience nothing particularly stood out as exceptional or problematic,32
1008,2024-01-01,4,Very solid product with great documentation and responsive support team,29
1009,2024-01-01,1,Terrible experience the product did not work as advertised at all for my use,37
1010,2024-01-01,5,One of the best tools I have used in recent years definitely worth the price,42
"""


class TestScenarioE_OpenEnded:
    def test_open_ended_skipped(self):
        reader = _csv_to_reader(OPENENDED_CSV, skip_open_ended=True)
        for q in reader.schema.questions:
            assert "experience" not in q.name.lower()

    def test_open_ended_in_skipped_list(self):
        reader = _csv_to_reader(OPENENDED_CSV, skip_open_ended=True)
        assert any("experience" in c.lower() for c in reader.skipped_columns)

    def test_numeric_cols_kept(self):
        reader = _csv_to_reader(OPENENDED_CSV, skip_open_ended=True)
        assert any("Satisfaction" in n for n in reader.schema.names)

    def test_age_kept(self):
        reader = _csv_to_reader(OPENENDED_CSV, skip_open_ended=True)
        assert any("Age" in n for n in reader.schema.names)


# ---------------------------------------------------------------------------
# Scenario F — 1-row header (pre-cleaned / manually exported)
# ---------------------------------------------------------------------------

SINGLE_HEADER_CSV = """\
Satisfaction,Brand,Age,Likelihood
4,Lenovo,28,8
3,HP,35,7
5,Lenovo,22,10
2,Dell,40,6
4,Lenovo,31,9
5,HP,29,10
3,Dell,44,8
1,Lenovo,55,5
4,HP,38,7
5,Lenovo,27,9
"""


class TestScenarioF_SingleHeader:
    def test_single_header_parsed(self):
        reader = _csv_to_reader(SINGLE_HEADER_CSV)
        assert len(reader.schema) >= 2

    def test_values_correct(self):
        reader = _csv_to_reader(SINGLE_HEADER_CSV)
        df = reader.clean_df
        assert len(df) == 10

    def test_categorical_brand(self):
        reader = _csv_to_reader(SINGLE_HEADER_CSV)
        # Column may be named "Brand" directly or "Brand__<first_value>" if misdetected
        brand_q = next((q for q in reader.schema.questions if "Brand" in q.name), None)
        assert brand_q is not None
        assert brand_q.qtype == "categorical"
        assert set(brand_q.categories) == {"Lenovo", "HP", "Dell"}


# ---------------------------------------------------------------------------
# Scenario G — Yes/No binary questions
# ---------------------------------------------------------------------------

YESNO_CSV = """\
Respondent ID,Start Date,Would you buy again?,Did you recommend us?,Satisfaction
,,,,
1001,2024-01-01,Yes,Yes,4
1002,2024-01-01,No,No,2
1003,2024-01-01,Yes,Yes,5
1004,2024-01-01,Yes,No,3
1005,2024-01-01,No,No,2
1006,2024-01-01,Yes,Yes,4
1007,2024-01-01,Yes,Yes,5
1008,2024-01-01,No,Yes,3
1009,2024-01-01,Yes,No,4
1010,2024-01-01,Yes,Yes,5
"""


class TestScenarioG_YesNo:
    def test_yesno_as_ordinal(self):
        reader = _csv_to_reader(YESNO_CSV)
        q = next(q for q in reader.schema.questions if "buy" in q.name)
        # Yes/No is in our Likert map → ordinal(1,2) or categorical
        assert q.qtype in ("ordinal", "categorical")

    def test_satisfaction_ordinal(self):
        reader = _csv_to_reader(YESNO_CSV)
        q = next(q for q in reader.schema.questions if "Satisfaction" in q.name)
        assert q.qtype == "ordinal"


# ---------------------------------------------------------------------------
# Scenario H — NPS (Net Promoter Score, 0-10)
# ---------------------------------------------------------------------------

NPS_CSV = """\
Respondent ID,Start Date,How likely are you to recommend? (0-10),Satisfaction
,,,
1001,2024-01-01,9,5
1002,2024-01-01,7,4
1003,2024-01-01,10,5
1004,2024-01-01,6,3
1005,2024-01-01,8,4
1006,2024-01-01,10,5
1007,2024-01-01,3,2
1008,2024-01-01,9,4
1009,2024-01-01,7,3
1010,2024-01-01,10,5
"""


class TestScenarioH_NPS:
    def test_nps_detected_as_ordinal(self):
        reader = _csv_to_reader(NPS_CSV)
        q = next(q for q in reader.schema.questions if "recommend" in q.name.lower())
        assert q.qtype == "ordinal"
        assert q.scale[0] >= 0
        assert q.scale[1] == 10


# ---------------------------------------------------------------------------
# End-to-end: read → synthesize
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_condensed_to_synthesize(self):
        reader = _csv_to_reader(CONDENSED_CSV)
        df, schema = reader.to_synthesizer_ready()
        synth = SurveySynthesizer(n_mcmc=100, random_state=42)
        synth.fit(df, schema)
        pop = synth.synthesize(N=100)
        assert len(pop) == 100
        assert list(pop.columns) == schema.names

    def test_likert_to_synthesize(self):
        reader = _csv_to_reader(LIKERT_CSV)
        df, schema = reader.to_synthesizer_ready()
        synth = SurveySynthesizer(n_mcmc=100, random_state=0)
        synth.fit(df, schema)
        pop = synth.synthesize(N=100)
        assert len(pop) == 100

    def test_matrix_to_synthesize(self):
        reader = _csv_to_reader(MATRIX_CSV)
        df, schema = reader.to_synthesizer_ready()
        synth = SurveySynthesizer(n_mcmc=100, random_state=1)
        synth.fit(df, schema)
        pop = synth.synthesize(N=100)
        assert len(pop) == 100
        for q in schema.questions:
            col = pop[q.name]
            if q.qtype == "ordinal":
                assert col.between(*q.scale).all()

    def test_summary_runs(self, capsys):
        reader = _csv_to_reader(CONDENSED_CSV)
        reader.summary()
        out = capsys.readouterr().out
        assert "SurveyMonkeyReader" in out

    def test_top_level_import(self):
        from universitybox.survey import SurveyMonkeyReader
        assert SurveyMonkeyReader is not None
