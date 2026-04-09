"""
Club — interest-based audience segmentation.

A Club groups users by their dominant brand interaction, subject to a
minimum engagement threshold (CTA events).

Usage
-----
    from universitybox.segments import Club

    club = Club(category_map={"lenovo": "Technology", "samsung": "Technology"})
    club.fit(events_df)          # events_df: [user_id, brand, cta_count]

    tech_users = club.members("Technology")
    size       = club.size("Technology")
    share      = club.share("Technology")
    summary    = club.summary()
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Any


class Club:
    """
    Interest-based audience club.

    Parameters
    ----------
    category_map : dict
        Maps brand name (lower-case) → macro category string.
        E.g. {"lenovo": "Technology", "zara": "Fashion"}.
    min_cta : int
        Minimum total CTA clicks for a user to be classified.
        Users below this threshold are labelled 'No Interest Data'.
        Default: 6.
    label_col : str
        Column name in the events DataFrame containing brand/label.
    user_col : str
        Column name for the user identifier.
    cta_col : str
        Column name for the CTA click count.
    """

    def __init__(
        self,
        category_map: Dict[str, str],
        min_cta: int = 6,
        label_col: str = "brand",
        user_col: str = "user_id",
        cta_col: str = "cta_count",
    ):
        self.category_map = {k.strip().lower(): v for k, v in category_map.items()}
        self.min_cta = min_cta
        self.label_col = label_col
        self.user_col = user_col
        self.cta_col = cta_col

        self._user_labels: Optional[Dict[Any, str]] = None
        self._category_counts: Optional[Dict[str, int]] = None

    def fit(self, events) -> "Club":
        """
        Classify users from a tabular events structure.

        Parameters
        ----------
        events : pandas DataFrame or dict-like with columns:
                 [user_col, label_col, cta_col]

        Returns
        -------
        self
        """
        # Accept both pandas DataFrame and plain dicts
        try:
            import pandas as pd
            if not isinstance(events, pd.DataFrame):
                events = pd.DataFrame(events)

            agg = (
                events
                .groupby([self.user_col, self.label_col])[self.cta_col]
                .sum()
                .reset_index()
            )

            # Total CTA per user
            totals = (
                agg.groupby(self.user_col)[self.cta_col].sum().rename("total")
            )

            # Top brand per user
            idx_max = agg.groupby(self.user_col)[self.cta_col].idxmax()
            top_brand = agg.loc[idx_max].set_index(self.user_col)[self.label_col]

            combined = pd.concat([totals, top_brand], axis=1)
            combined.columns = ["total", "top_brand"]

            def classify(row):
                if row["total"] < self.min_cta:
                    return "No Interest Data"
                key = str(row["top_brand"]).strip().lower()
                if key in self.category_map:
                    return self.category_map[key]
                # Fuzzy fallback
                for map_key, cat in self.category_map.items():
                    if map_key in key or key in map_key:
                        return cat
                return "Other"

            combined["category"] = combined.apply(classify, axis=1)
            self._user_labels = combined["category"].to_dict()
            counts = combined["category"].value_counts().to_dict()
            self._category_counts = counts

        except ImportError:
            raise ImportError(
                "pandas is required for Club.fit(). "
                "Install with: pip install universitybox[full]"
            )

        return self

    def members(self, category: str) -> list:
        """Return list of user IDs in the given category Club."""
        self._check_fitted()
        return [u for u, c in self._user_labels.items() if c == category]

    def size(self, category: str) -> int:
        """Number of users in the given Club."""
        self._check_fitted()
        return self._category_counts.get(category, 0)

    def share(self, category: str, exclude_no_data: bool = True) -> float:
        """
        Fraction of (classified) users in the given Club.

        Parameters
        ----------
        exclude_no_data : if True, denominator excludes 'No Interest Data'.
        """
        self._check_fitted()
        denom = sum(
            v for k, v in self._category_counts.items()
            if not exclude_no_data or k != "No Interest Data"
        )
        if denom == 0:
            return 0.0
        return self._category_counts.get(category, 0) / denom

    def summary(self) -> dict:
        """
        Return a summary dict of all clubs with size and share.
        """
        self._check_fitted()
        result = {}
        total_classified = sum(
            v for k, v in self._category_counts.items()
            if k != "No Interest Data"
        )
        total_all = sum(self._category_counts.values())
        for cat, cnt in sorted(self._category_counts.items(),
                               key=lambda x: -x[1]):
            result[cat] = {
                "size": cnt,
                "share_of_classified": (
                    cnt / total_classified if total_classified > 0 else 0.0
                ),
                "share_of_all": cnt / total_all if total_all > 0 else 0.0,
            }
        return result

    def categories(self) -> List[str]:
        """List all categories found in the fitted data."""
        self._check_fitted()
        return list(self._category_counts.keys())

    def _check_fitted(self):
        if self._user_labels is None:
            raise RuntimeError("Call fit() before using this method.")
