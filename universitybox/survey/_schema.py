"""
universitybox.survey._schema
============================
SurveySchema — describes the structure of a survey.

Each question has a name and a type:
  - categorical : unordered categories (nominal)
  - ordinal     : ordered integer scale  (e.g. Likert 1-5)
  - continuous  : real-valued with bounds (e.g. Age 18-80)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple


QuestionType = Literal["categorical", "ordinal", "continuous"]


@dataclass
class Question:
    name: str
    qtype: QuestionType
    # categorical
    categories: Optional[List[str]] = None
    # ordinal
    scale: Optional[Tuple[int, int]] = None   # (min, max) inclusive
    # continuous
    bounds: Optional[Tuple[float, float]] = None

    def __post_init__(self) -> None:
        if self.qtype == "categorical":
            if not self.categories or len(self.categories) < 2:
                raise ValueError(
                    f"Question '{self.name}': categorical questions need at least 2 categories."
                )
        elif self.qtype == "ordinal":
            if self.scale is None:
                raise ValueError(
                    f"Question '{self.name}': ordinal questions require a scale, e.g. scale=(1, 5)."
                )
            lo, hi = self.scale
            if not isinstance(lo, int) or not isinstance(hi, int) or hi <= lo:
                raise ValueError(
                    f"Question '{self.name}': scale must be (int_min, int_max) with max > min."
                )
        elif self.qtype == "continuous":
            if self.bounds is None:
                raise ValueError(
                    f"Question '{self.name}': continuous questions require bounds, e.g. bounds=(0.0, 100.0)."
                )
            lo, hi = self.bounds
            if hi <= lo:
                raise ValueError(
                    f"Question '{self.name}': bounds must satisfy max > min."
                )
        else:
            raise ValueError(f"Unknown question type: '{self.qtype}'")


@dataclass
class SurveySchema:
    """
    Defines the structure of a survey.

    Usage
    -----
    schema = SurveySchema()
    schema.add_categorical("Preferred_Brand", categories=["Lenovo", "HP", "Dell"])
    schema.add_ordinal("Satisfaction", scale=(1, 5))
    schema.add_continuous("Age", bounds=(18, 65))
    """

    _questions: List[Question] = field(default_factory=list, init=False, repr=False)

    def add_categorical(self, name: str, categories: List[str]) -> "SurveySchema":
        """Add a nominal (unordered) categorical question."""
        self._questions.append(
            Question(name=name, qtype="categorical", categories=list(categories))
        )
        return self

    def add_ordinal(self, name: str, scale: Tuple[int, int]) -> "SurveySchema":
        """Add an ordinal / Likert question. scale=(min, max) inclusive integers."""
        self._questions.append(
            Question(name=name, qtype="ordinal", scale=tuple(scale))  # type: ignore[arg-type]
        )
        return self

    def add_continuous(self, name: str, bounds: Tuple[float, float]) -> "SurveySchema":
        """Add a continuous numeric question. bounds=(min, max)."""
        self._questions.append(
            Question(name=name, qtype="continuous", bounds=tuple(bounds))  # type: ignore[arg-type]
        )
        return self

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def questions(self) -> List[Question]:
        return list(self._questions)

    @property
    def names(self) -> List[str]:
        return [q.name for q in self._questions]

    def __len__(self) -> int:
        return len(self._questions)

    def __repr__(self) -> str:
        lines = [f"SurveySchema ({len(self)} questions):"]
        for q in self._questions:
            if q.qtype == "categorical":
                detail = f"categories={q.categories}"
            elif q.qtype == "ordinal":
                detail = f"scale={q.scale}"
            else:
                detail = f"bounds={q.bounds}"
            lines.append(f"  [{q.qtype}] {q.name} — {detail}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def validate_dataframe(self, df: "import pandas as pd; pd.DataFrame") -> None:  # type: ignore[valid-type]
        """Raise ValueError if df is missing columns or has out-of-range values."""
        import pandas as pd  # noqa: F401 — only used for type hint context

        missing = [q.name for q in self._questions if q.name not in df.columns]
        if missing:
            raise ValueError(f"DataFrame is missing columns: {missing}")

        for q in self._questions:
            col = df[q.name].dropna()
            if q.qtype == "categorical":
                bad = set(col.astype(str)) - set(q.categories)  # type: ignore[arg-type]
                if bad:
                    raise ValueError(
                        f"Column '{q.name}' contains unknown categories: {bad}"
                    )
            elif q.qtype == "ordinal":
                lo, hi = q.scale  # type: ignore[misc]
                out = col[(col < lo) | (col > hi)]
                if len(out):
                    raise ValueError(
                        f"Column '{q.name}' has values outside scale {q.scale}: {out.unique().tolist()}"
                    )
            elif q.qtype == "continuous":
                lo, hi = q.bounds  # type: ignore[misc]
                out = col[(col < lo) | (col > hi)]
                if len(out):
                    raise ValueError(
                        f"Column '{q.name}' has values outside bounds {q.bounds}: {out.unique().tolist()}"
                    )
