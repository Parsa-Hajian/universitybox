"""
universitybox.survey._synthesizer
===================================
SurveySynthesizer — orchestrates the three-stage pipeline:

  Stage 1: Per-column Bayesian marginal estimation (_marginals)
  Stage 2: Gaussian Copula for joint dependency (_copula)
  Stage 3: Seed selection + NHOP rejection + density oversampling (_nhop)
"""
from __future__ import annotations

from typing import List, Optional, Union

import numpy as np

from ._schema import SurveySchema, Question
from ._marginals import CategoricalMarginal, OrdinalMarginal, ContinuousMarginal
from ._copula import GaussianCopula
from ._nhop import apply_stage3


class SurveySynthesizer:
    """
    Synthesise survey responses from a small real sample.

    Parameters
    ----------
    n_mcmc : int
        Number of MCMC iterations for ordinal Gibbs sampler (Stage 1).
        Default: 500. Increase for better mixing with fewer real responses.
    n_seeds : int
        Number of k-means++ seeds for Stage 3 seed selection.
        Default: 10.
    k_nhop : int
        Number of nearest neighbours for NHOP rejection (Stage 3).
        Default: 5.
    nhop_pct : float
        Percentile threshold for NHOP (0–100). Higher = more permissive.
        Default: 99.0 (keeps points within 99th percentile of real kNN distances).
    k_density : int
        Number of neighbours for local density estimation in oversampling.
        Default: 5.
    oversample_factor : int
        Intermediate oversampling multiplier: generate this many times the
        target N before NHOP pruning, so pruning has enough candidates.
        Default: 5.
    random_state : int or None
        Seed for reproducibility.

    Usage
    -----
    >>> schema = SurveySchema()
    >>> schema.add_ordinal("Satisfaction", scale=(1, 5))
    >>> schema.add_categorical("City", categories=["Rome", "Milan"])
    >>>
    >>> synth = SurveySynthesizer(n_mcmc=500, random_state=42)
    >>> synth.fit(real_df, schema)
    >>> population = synth.synthesize(N=1000)
    """

    def __init__(
        self,
        n_mcmc: int = 500,
        n_seeds: int = 10,
        k_nhop: int = 5,
        nhop_pct: float = 99.0,
        k_density: int = 5,
        oversample_factor: int = 5,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_mcmc = n_mcmc
        self.n_seeds = n_seeds
        self.k_nhop = k_nhop
        self.nhop_pct = nhop_pct
        self.k_density = k_density
        self.oversample_factor = oversample_factor
        self.random_state = random_state

        self._schema: Optional[SurveySchema] = None
        self._marginals: List = []
        self._copula: Optional[GaussianCopula] = None
        self._real_raw: Optional[np.ndarray] = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    def fit(self, df, schema: SurveySchema) -> "SurveySynthesizer":
        """
        Learn the distribution from real survey responses.

        Parameters
        ----------
        df     : pandas DataFrame with one column per question in schema.
        schema : SurveySchema describing question types and constraints.

        Returns self (for chaining).
        """
        import pandas as pd

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if len(df) < 4:
            raise ValueError(
                f"Need at least 4 real responses to fit; got {len(df)}."
            )
        schema.validate_dataframe(df)
        if len(schema) == 0:
            raise ValueError("Schema must have at least one question.")

        self._schema = schema
        rng = np.random.default_rng(self.random_state)

        # Subset to schema columns in order
        df = df[schema.names].copy()

        # Stage 1: fit per-column marginals
        self._marginals = []
        for q in schema.questions:
            col = df[q.name].values
            if q.qtype == "categorical":
                m = CategoricalMarginal(categories=q.categories).fit(col)  # type: ignore[arg-type]
            elif q.qtype == "ordinal":
                m = OrdinalMarginal(scale=q.scale, n_mcmc=self.n_mcmc).fit(col, rng)  # type: ignore[arg-type]
            else:
                m = ContinuousMarginal(bounds=q.bounds).fit(col)  # type: ignore[arg-type]
            self._marginals.append(m)

        # Stage 2: fit copula
        self._copula = GaussianCopula().fit(df, self._marginals)

        # Store real data as object array for NHOP
        self._real_raw = df.values.astype(object)

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    def synthesize(self, N: int) -> "import pandas as pd; pd.DataFrame":  # type: ignore[valid-type]
        """
        Generate N synthetic survey responses.

        Parameters
        ----------
        N : int — number of synthetic responses to produce.

        Returns
        -------
        pandas DataFrame of shape (N, p) with correct dtypes per column.
        """
        import pandas as pd

        if not self._is_fitted:
            raise RuntimeError("Call fit() before synthesize().")

        rng = np.random.default_rng(self.random_state)
        schema = self._schema
        questions = schema.questions  # type: ignore[union-attr]

        # Step 1: draw a large intermediate batch from the copula
        n_intermediate = max(N * self.oversample_factor, N + 500)
        raw = self._copula.synthesise(n_intermediate, self._marginals, rng)  # (n_int, p)

        # Step 2: Stage 3 — seed selection + NHOP + density oversampling
        final_raw = apply_stage3(
            synthetic_raw=raw,
            real_raw=self._real_raw,  # type: ignore[arg-type]
            questions=questions,
            target_n=N,
            n_seeds=self.n_seeds,
            k_nhop=self.k_nhop,
            nhop_pct=self.nhop_pct,
            k_density=self.k_density,
            rng=rng,
        )

        # Step 3: build DataFrame with correct dtypes
        data = {}
        for j, q in enumerate(questions):
            col_vals = final_raw[:, j]
            if q.qtype == "categorical":
                data[q.name] = pd.Categorical(
                    col_vals.astype(str),
                    categories=q.categories,
                )
            elif q.qtype == "ordinal":
                data[q.name] = col_vals.astype(int)
            else:
                data[q.name] = col_vals.astype(float)

        return pd.DataFrame(data)

    # ------------------------------------------------------------------
    def summary(self) -> None:
        """Print a summary of the fitted synthesizer."""
        if not self._is_fitted:
            print("SurveySynthesizer — not fitted yet.")
            return
        schema = self._schema
        print("SurveySynthesizer")
        print("=" * 40)
        print(f"  Real responses used : {len(self._real_raw)}")
        print(f"  Questions           : {len(schema)}")
        for q in schema.questions:  # type: ignore[union-attr]
            if q.qtype == "categorical":
                detail = f"categories={q.categories}"
            elif q.qtype == "ordinal":
                detail = f"scale={q.scale}"
            else:
                detail = f"bounds={q.bounds}"
            print(f"    [{q.qtype:11s}] {q.name} — {detail}")
        print(f"  MCMC iterations     : {self.n_mcmc}")
        print(f"  NHOP k / pct        : {self.k_nhop} / {self.nhop_pct}%")
        print(f"  Seeds               : {self.n_seeds}")
        print(f"  Oversample factor   : {self.oversample_factor}x")
