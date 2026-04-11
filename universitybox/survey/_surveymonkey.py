"""
universitybox.survey._surveymonkey
====================================
SurveyMonkeyReader — parse SurveyMonkey CSV / Excel exports into a clean
DataFrame + SurveySchema ready for SurveySynthesizer.

Handles all standard SurveyMonkey export scenarios:

  Format A — 2-row header, Condensed
    Row 0: question text (+ metadata column names)
    Row 1: sub-label / answer option (blank for condensed single-answer)
    Values: selected answer text (or blank)

  Format B — 2-row header, Expanded
    Row 0: question text repeated for each answer option
    Row 1: each answer option text
    Values: answer text if selected, blank if not
    (Multi-select and matrix questions always come as expanded columns)

  Format C — 2-row header, Numerical
    Like Expanded but with numeric codes instead of text

  Format D — 1-row header (pre-cleaned / manual export)
    Single header row, values may be text or numeric

  Format E — Excel (.xlsx)
    Same structure as CSV but in Excel format; may have extra formatting rows

Question-type auto-detection
-----------------------------
  Integer values in a bounded range      → ordinal (scale = min..max)
  Known Likert text labels               → ordinal (mapped to 1..K)
  Small finite set of string values      → categorical
  Continuous / wide numeric range        → continuous
  Binary 0/1 expanded columns (grouped)  → multi-select → one binary per option
  High-cardinality / long free text      → SKIPPED (open-ended)

Metadata columns stripped automatically
----------------------------------------
  Respondent ID, Survey ID, Collector ID, Start Date, End Date,
  Last Updated, IP Address, Email Address, First Name, Last Name,
  Custom Data 1/2/3, Score (NPS internal), …
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._schema import SurveySchema


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_META_COLS: frozenset = frozenset({
    "respondent id", "survey id", "collector id",
    "start date", "end date", "last updated", "last modified date",
    "ip address", "email address", "email",
    "first name", "last name",
    "custom data 1", "custom data 2", "custom data 3",
    "custom data", "score",
})

# Well-known ordered Likert label sets → mapped to integers 1..K
_LIKERT_MAPS: List[Tuple[List[str], int]] = [
    # 5-point agreement
    (["strongly disagree", "disagree", "neutral", "agree", "strongly agree"], 5),
    (["strongly disagree", "disagree", "neither agree nor disagree", "agree", "strongly agree"], 5),
    (["strongly disagree", "disagree", "neither", "agree", "strongly agree"], 5),
    # 5-point satisfaction
    (["very dissatisfied", "dissatisfied", "neutral", "satisfied", "very satisfied"], 5),
    (["very dissatisfied", "dissatisfied", "neither", "satisfied", "very satisfied"], 5),
    # 5-point frequency
    (["never", "rarely", "sometimes", "often", "always"], 5),
    # 5-point likelihood
    (["very unlikely", "unlikely", "neutral", "likely", "very likely"], 5),
    (["very unlikely", "unlikely", "neither", "likely", "very likely"], 5),
    # 5-point quality
    (["very poor", "poor", "average", "good", "excellent"], 5),
    (["very poor", "poor", "fair", "good", "excellent"], 5),
    # 4-point
    (["never", "sometimes", "often", "always"], 4),
    (["strongly disagree", "disagree", "agree", "strongly agree"], 4),
    # 7-point
    (["strongly disagree", "disagree", "somewhat disagree", "neutral",
      "somewhat agree", "agree", "strongly agree"], 7),
    # Yes/No (binary ordinal)
    (["no", "yes"], 2),
    (["yes", "no"], 2),
]

_LIKERT_LOOKUP: Dict[frozenset, Tuple[Dict[str, int], int]] = {}
for _labels, _k in _LIKERT_MAPS:
    _key = frozenset(l.lower() for l in _labels)
    _mapping = {l.lower(): i + 1 for i, l in enumerate(_labels)}
    _LIKERT_LOOKUP[_key] = (_mapping, _k)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_meta(col_name: str) -> bool:
    return col_name.strip().lower() in _META_COLS


def _normalise_col(name: str) -> str:
    """Strip whitespace / newlines from column names."""
    return str(name).strip().replace("\n", " ").replace("\r", " ")


def _detect_sm_header_rows(raw: pd.DataFrame) -> int:
    """
    Return 1 or 2: how many header rows this SurveyMonkey export has.

    Heuristic: if row index 1 (the candidate second header) is mostly
    non-numeric short strings or blanks (not response data), it's a 2-row
    header file.  If it looks like actual response values (lots of numbers),
    it's already a 1-row header file.
    """
    if len(raw) < 2:
        return 1
    # Check the SECOND row (index 1) to decide whether it is a sub-label row
    # (SurveyMonkey 2-row header) or actual response data (1-row header).
    row1_vals = raw.iloc[1].fillna("").astype(str).tolist()
    # Count values that look like actual survey responses (numbers or longer text)
    numeric_count = sum(
        1 for v in row1_vals
        if v.strip() and re.match(r"^-?\d+(\.\d+)?$", v.strip())
    )
    long_text_count = sum(1 for v in row1_vals if len(v.strip()) > 40)
    data_like = numeric_count + long_text_count
    if data_like / max(len(row1_vals), 1) > 0.4:
        return 1  # second row looks like response data → 1-row header
    return 2


def _detect_question_type(col: pd.Series, col_name: str) -> Optional[dict]:
    """
    Analyse a single column and return a dict describing how to treat it:
      {"type": "ordinal",    "scale": (lo, hi), "map": {label: int} or None}
      {"type": "categorical","categories": [...]}
      {"type": "continuous", "bounds": (lo, hi)}
      {"type": "skip",       "reason": "..."}    — open-ended or unrecognised
    Returns None to indicate the column should be skipped.
    """
    non_null = col.dropna()
    non_null = non_null[non_null.astype(str).str.strip() != ""]
    n = len(non_null)
    if n == 0:
        return {"type": "skip", "reason": "all null"}

    # --- Try numeric ---
    try:
        numeric = pd.to_numeric(non_null, errors="raise")
        lo = int(numeric.min()) if numeric.dtype.kind in "iu" else float(numeric.min())
        hi = int(numeric.max()) if numeric.dtype.kind in "iu" else float(numeric.max())
        n_unique = numeric.nunique()

        # Integer range with few unique values → ordinal
        if numeric.dtype.kind in "iu" or (numeric == numeric.round()).all():
            lo_i, hi_i = int(lo), int(hi)
            n_levels = hi_i - lo_i + 1
            if 2 <= n_levels <= 11:  # covers 1-5, 1-7, 0-10 (NPS), etc.
                return {"type": "ordinal", "scale": (lo_i, hi_i), "map": None}

        # Float or wide integer range → continuous
        return {"type": "continuous", "bounds": (float(lo), float(hi))}

    except (ValueError, TypeError):
        pass

    # --- String values ---
    str_vals = non_null.astype(str).str.strip()
    unique_vals = str_vals.unique().tolist()
    n_unique = len(unique_vals)

    # Check for known Likert scale
    lower_set = frozenset(v.lower() for v in unique_vals)
    if lower_set in _LIKERT_LOOKUP:
        mapping, k = _LIKERT_LOOKUP[lower_set]
        return {"type": "ordinal", "scale": (1, k), "map": mapping}

    # High cardinality or long text → open-ended, skip
    avg_len = str_vals.str.len().mean()
    uniqueness = n_unique / n
    if avg_len > 50 or uniqueness > 0.7:
        return {"type": "skip", "reason": "open-ended / high cardinality"}

    # Small finite set → categorical
    if n_unique <= 30:
        return {"type": "categorical", "categories": sorted(unique_vals)}

    return {"type": "skip", "reason": "too many unique values"}


# ---------------------------------------------------------------------------
# Main reader class
# ---------------------------------------------------------------------------

class SurveyMonkeyReader:
    """
    Parse a SurveyMonkey CSV or Excel export into a clean DataFrame
    and an auto-detected SurveySchema.

    Supported scenarios
    -------------------
    - 2-row header CSV (condensed, expanded, numerical)
    - 1-row header CSV (pre-cleaned or manually exported)
    - Excel (.xlsx / .xls) with same structures
    - Matrix questions → one ordinal column per matrix row
    - Multi-select "select all that apply" expanded → binary categorical per option
    - Likert text labels → auto-mapped to integer ordinal
    - NPS (0-10) → ordinal
    - Open-ended / free text → skipped automatically
    - Metadata columns (Respondent ID, dates, IP, …) → stripped

    Parameters
    ----------
    skip_open_ended : bool
        If True (default), automatically skip open-ended text columns.
    min_responses : int
        Columns with fewer than this many non-null responses are skipped.
    multiselect_as_binary : bool
        If True (default), each option of a "select all that apply" question
        becomes its own binary "Yes"/"No" categorical column.
        If False, the options are joined into a single comma-separated string
        column (treated as categorical).

    Usage
    -----
    >>> reader = SurveyMonkeyReader()
    >>> reader.read_csv("survey_export.csv")
    >>> df, schema = reader.to_synthesizer_ready()
    >>> synth = SurveySynthesizer().fit(df, schema)
    >>> population = synth.synthesize(N=2000)
    """

    def __init__(
        self,
        skip_open_ended: bool = True,
        min_responses: int = 3,
        multiselect_as_binary: bool = True,
    ) -> None:
        self.skip_open_ended = skip_open_ended
        self.min_responses = min_responses
        self.multiselect_as_binary = multiselect_as_binary

        self._clean_df: Optional[pd.DataFrame] = None
        self._schema: Optional[SurveySchema] = None
        self._skipped_cols: List[str] = []
        self._col_type_map: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Public read methods
    # ------------------------------------------------------------------

    def read_csv(
        self,
        path: str,
        encoding: str = "utf-8-sig",
        delimiter: str = ",",
    ) -> "SurveyMonkeyReader":
        """
        Load a SurveyMonkey CSV export.

        Parameters
        ----------
        path     : path to the .csv file
        encoding : file encoding (default utf-8-sig handles BOM from Windows exports)
        delimiter: column delimiter (default ','; some locales use ';')
        """
        # Read raw without interpreting headers
        raw = pd.read_csv(path, header=None, encoding=encoding, sep=delimiter,
                          dtype=str, keep_default_na=False,
                          on_bad_lines="warn", engine="python")
        return self._parse_raw(raw)

    def read_excel(self, path: str, sheet_name: int = 0) -> "SurveyMonkeyReader":
        """
        Load a SurveyMonkey Excel (.xlsx / .xls) export.

        Parameters
        ----------
        path       : path to the .xlsx or .xls file
        sheet_name : sheet index or name (default: first sheet)
        """
        raw = pd.read_excel(path, header=None, sheet_name=sheet_name, dtype=str)
        raw = raw.fillna("")
        return self._parse_raw(raw)

    def parse(self, df_raw: pd.DataFrame) -> "SurveyMonkeyReader":
        """
        Parse an already-loaded raw DataFrame (e.g. from your own pd.read_csv call).
        The DataFrame must not yet have proper headers set — pass the raw output
        of pd.read_csv(..., header=None).
        """
        return self._parse_raw(df_raw.astype(str).fillna(""))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def clean_df(self) -> pd.DataFrame:
        if self._clean_df is None:
            raise RuntimeError("Call read_csv(), read_excel(), or parse() first.")
        return self._clean_df

    @property
    def schema(self) -> SurveySchema:
        if self._schema is None:
            raise RuntimeError("Call read_csv(), read_excel(), or parse() first.")
        return self._schema

    @property
    def skipped_columns(self) -> List[str]:
        """Columns that were dropped (open-ended, metadata, all-null, etc.)."""
        return list(self._skipped_cols)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def to_synthesizer_ready(self) -> Tuple[pd.DataFrame, SurveySchema]:
        """Return (clean_df, schema) for direct use with SurveySynthesizer."""
        return self.clean_df, self.schema

    def summary(self) -> None:
        """Print a diagnostic summary of parsed columns."""
        print(f"SurveyMonkeyReader — {len(self.clean_df)} responses")
        print(f"{'='*50}")
        print(f"  Included columns : {len(self.schema)}")
        for q in self.schema.questions:
            if q.qtype == "categorical":
                detail = f"categories={q.categories}"
            elif q.qtype == "ordinal":
                detail = f"scale={q.scale}"
            else:
                detail = f"bounds={q.bounds}"
            print(f"    [{q.qtype:11s}] {q.name} — {detail}")
        if self._skipped_cols:
            print(f"  Skipped columns  : {len(self._skipped_cols)}")
            for c in self._skipped_cols:
                reason = self._col_type_map.get(c, {}).get("reason", "")
                print(f"    SKIP  {c}  ({reason})")

    # ------------------------------------------------------------------
    # Core parsing logic
    # ------------------------------------------------------------------

    def _parse_raw(self, raw: pd.DataFrame) -> "SurveyMonkeyReader":
        raw = raw.copy()
        raw = raw.fillna("").astype(str)

        # --- Step 1: determine number of header rows ---
        n_header = _detect_sm_header_rows(raw)

        if n_header == 2:
            header_row0 = raw.iloc[0].tolist()   # question texts
            header_row1 = raw.iloc[1].tolist()   # sub-labels / answer options
            data = raw.iloc[2:].reset_index(drop=True)
        else:
            header_row0 = raw.iloc[0].tolist()
            header_row1 = [""] * len(header_row0)
            data = raw.iloc[1:].reset_index(drop=True)

        # Normalise header strings
        header_row0 = [_normalise_col(h) for h in header_row0]
        header_row1 = [_normalise_col(h) for h in header_row1]

        # --- Step 2: assign column names and build DataFrame ---
        col_names = self._build_col_names(header_row0, header_row1)
        data.columns = col_names

        # Replace empty strings with NaN
        data = data.replace({"": np.nan, "nan": np.nan})

        # --- Step 3: strip metadata columns ---
        keep_cols = [c for c in data.columns if not _is_meta(c.split("__")[0])]
        data = data[keep_cols]

        # --- Step 4: detect multi-select groups (expanded binary columns) ---
        data, multiselect_groups, binary_cols = self._collapse_multiselect(
            data, header_row0, header_row1, col_names
        )

        # --- Step 5: per-column type detection ---
        clean_cols = {}
        schema = SurveySchema()
        self._skipped_cols = []
        self._col_type_map = {}

        for col in data.columns:
            series = data[col].copy()
            # Skip if too few responses
            n_valid = series.dropna().shape[0]
            if n_valid < self.min_responses:
                self._skipped_cols.append(col)
                self._col_type_map[col] = {"type": "skip", "reason": "too few responses"}
                continue

            # Binary multi-select columns always have ["No", "Yes"] regardless of observed values
            if col in binary_cols:
                info = {"type": "categorical", "categories": ["No", "Yes"]}
            else:
                info = _detect_question_type(series, col)
            self._col_type_map[col] = info

            if info["type"] == "skip":
                if self.skip_open_ended:
                    self._skipped_cols.append(col)
                    continue
                else:
                    # Treat as categorical anyway
                    vals = series.dropna().astype(str).unique().tolist()
                    if len(vals) <= 30:
                        info = {"type": "categorical", "categories": sorted(vals)}
                    else:
                        self._skipped_cols.append(col)
                        continue

            if info["type"] == "ordinal":
                # Apply label mapping if present
                lmap = info.get("map")
                if lmap:
                    series = series.apply(
                        lambda v: lmap.get(str(v).strip().lower()) if pd.notna(v) else np.nan
                    )
                else:
                    series = pd.to_numeric(series, errors="coerce")
                series = series.dropna().astype(int)
                lo, hi = info["scale"]
                # Clip to declared scale (some exports have out-of-range artefacts)
                series = series.clip(lo, hi)
                # Rebuild with original index structure for alignment
                full_series = pd.to_numeric(
                    data[col].apply(lambda v: lmap.get(str(v).strip().lower()) if (lmap and pd.notna(v)) else v),
                    errors="coerce"
                ).clip(lo, hi) if lmap else pd.to_numeric(data[col], errors="coerce").clip(lo, hi)
                clean_cols[col] = full_series
                schema.add_ordinal(col, scale=(lo, hi))

            elif info["type"] == "categorical":
                cats = info["categories"]
                clean_series = data[col].astype(str).where(data[col].notna(), np.nan)
                clean_series = clean_series.where(clean_series != "nan", np.nan)
                clean_cols[col] = clean_series
                schema.add_categorical(col, categories=cats)

            elif info["type"] == "continuous":
                lo, hi = info["bounds"]
                numeric_series = pd.to_numeric(data[col], errors="coerce")
                clean_cols[col] = numeric_series.clip(lo, hi)
                schema.add_continuous(col, bounds=(lo, hi))

        # --- Step 6: build final DataFrame, drop rows with all NaN ---
        if not clean_cols:
            raise ValueError(
                "No usable columns found after parsing. "
                "Check that the file is a valid SurveyMonkey export and that "
                "question columns contain non-empty, non-text-only responses."
            )

        clean_df = pd.DataFrame(clean_cols)
        # Drop rows where ALL schema columns are null
        clean_df = clean_df.dropna(how="all").reset_index(drop=True)

        # For synthesis, also drop rows with any null in ordinal/continuous columns
        # (categorical nulls are OK — they're replaced by synthesised values)
        self._clean_df = clean_df
        self._schema = schema
        return self

    # ------------------------------------------------------------------
    # Helper: build unique column names from 2-row header
    # ------------------------------------------------------------------

    def _build_col_names(
        self, row0: List[str], row1: List[str]
    ) -> List[str]:
        """
        Combine question text (row0) and sub-label (row1) into unique column names.

        For metadata / single-answer questions: use row0 only.
        For matrix / multi-select: use "QuestionText__SubLabel".
        Deduplicate by appending __2, __3, etc.
        """
        result = []
        seen: Dict[str, int] = {}
        prev_q = ""

        for q, sub in zip(row0, row1):
            q = q.strip()
            sub = sub.strip()

            # Carry forward empty question text (happens in expanded exports)
            if not q and prev_q:
                q = prev_q
            else:
                prev_q = q

            if sub and sub.lower() not in ("response", ""):
                name = f"{q}__{sub}"
            else:
                name = q

            # Deduplicate
            if name in seen:
                seen[name] += 1
                name = f"{name}__{seen[name]}"
            else:
                seen[name] = 1

            result.append(name)

        return result

    # ------------------------------------------------------------------
    # Helper: detect and collapse multi-select (expanded binary) groups
    # ------------------------------------------------------------------

    def _collapse_multiselect(
        self,
        data: pd.DataFrame,
        header_row0: List[str],
        header_row1: List[str],
        col_names: List[str],
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Detect expanded multi-select groups (same question text, binary values,
        each column = one answer option).

        If multiselect_as_binary=True: keep one column per option as "Yes"/"No" categorical.
        If multiselect_as_binary=False: collapse all options into a single
            comma-separated string column.

        Returns the modified DataFrame and list of group question names.
        """
        # Find columns that are part of expanded binary groups:
        # same question prefix (before "__"), binary-ish values
        question_to_cols: Dict[str, List[str]] = {}
        for col in data.columns:
            if "__" in col:
                q_part = col.split("__")[0]
                question_to_cols.setdefault(q_part, []).append(col)

        groups_found = []
        drop_cols = []
        new_cols: Dict[str, pd.Series] = {}

        for q_text, cols in question_to_cols.items():
            if len(cols) < 2:
                continue  # only single sub-column — likely matrix, keep as-is

            # Check if all these columns look binary (0/1 or text/blank)
            is_binary_group = True
            for c in cols:
                non_null = data[c].dropna()
                non_null_str = non_null.astype(str).str.strip()
                unique_vals = set(non_null_str.unique()) - {"", "nan"}
                # Binary: at most 2 non-empty unique values, one of which should be
                # the option text or "1" / "True"
                if len(unique_vals) > 2:
                    is_binary_group = False
                    break

            if not is_binary_group:
                continue  # matrix or multi-label — leave as individual columns

            groups_found.append(q_text)

            if self.multiselect_as_binary:
                # Convert each option column → "Yes"/"No"
                for c in cols:
                    option_label = c.split("__", 1)[-1]
                    col_key = f"{q_text}__{option_label}"
                    series = data[c].copy()
                    non_null_vals = series.dropna().astype(str).str.strip()
                    unique_vals = set(non_null_vals.unique()) - {"", "nan"}

                    # Determine what "selected" looks like
                    # Could be: "1", option text itself, "True", "Yes"
                    def _to_binary(v):
                        if pd.isna(v):
                            return np.nan
                        s = str(v).strip()
                        if s in ("", "nan"):
                            return "No"
                        if s in ("0", "False", "false"):
                            return "No"
                        return "Yes"

                    new_cols[col_key] = series.apply(_to_binary)
                    drop_cols.append(c)
            else:
                # Collapse to comma-separated string
                def _to_csv_string(row):
                    selected = []
                    for c in cols:
                        v = row[c]
                        if pd.isna(v):
                            continue
                        s = str(v).strip()
                        if s and s not in ("0", "False", "false", "nan", ""):
                            option = c.split("__", 1)[-1]
                            selected.append(option)
                    return ",".join(selected) if selected else np.nan

                new_cols[q_text] = data[cols].apply(_to_csv_string, axis=1)
                drop_cols.extend(cols)

        if drop_cols:
            data = data.drop(columns=drop_cols)
        for col_name, series in new_cols.items():
            data[col_name] = series

        return data, groups_found, set(new_cols.keys())
