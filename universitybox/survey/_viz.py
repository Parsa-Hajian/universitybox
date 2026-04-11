"""
universitybox.survey._viz
==========================
Visualisation utilities for comparing real vs synthetic survey responses.

compare_plot(real_df, synthetic_df, schema)
  • PCA scatter  — both populations projected onto 2 principal components
  • Marginal plots — per-column distributions side-by-side:
      ordinal / continuous → histogram + KDE
      categorical          → grouped bar chart

Requires matplotlib (optional dep: pip install universitybox[viz]).
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from ._schema import SurveySchema


# ---------------------------------------------------------------------------
# Numeric encoding (mirrors _nhop._encode but standalone)
# ---------------------------------------------------------------------------

def _encode_df(df: "pd.DataFrame", schema: "SurveySchema") -> np.ndarray:
    """Encode a mixed-type DataFrame to a float matrix using the schema."""
    import pandas as pd
    cols = []
    for q in schema.questions:
        if q.name not in df.columns:
            continue
        col = df[q.name]
        if q.qtype == "categorical":
            cat_map = {c: i for i, c in enumerate(q.categories)}  # type: ignore[arg-type]
            encoded = col.map(lambda v: cat_map.get(str(v), 0) if pd.notna(v) else 0)
        else:
            encoded = pd.to_numeric(col, errors="coerce").fillna(0)
        cols.append(encoded.values.astype(float))
    return np.column_stack(cols) if cols else np.empty((len(df), 0))


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def compare_plot(
    real_df: "pd.DataFrame",
    synthetic_df: "pd.DataFrame",
    schema: "SurveySchema",
    method: str = "pca",
    n_components: int = 2,
    figsize: Optional[tuple] = None,
    alpha_real: float = 0.8,
    alpha_syn: float = 0.35,
    color_real: str = "#2563EB",      # blue
    color_syn: str = "#F97316",       # orange
    bins: int = 20,
    title: str = "Real vs Synthetic Survey Responses",
) -> "import matplotlib.figure.Figure":  # type: ignore[valid-type]
    """
    Plot real vs synthetic survey distributions side-by-side.

    Parameters
    ----------
    real_df       : original real responses (DataFrame)
    synthetic_df  : synthesised responses from SurveySynthesizer.synthesize()
    schema        : SurveySchema used to fit the synthesizer
    method        : dimensionality reduction for the overview plot.
                    'pca' (default) or 'umap' (requires umap-learn).
    n_components  : number of PCA/UMAP components (2 recommended for scatter)
    figsize       : (width, height) in inches; auto-calculated if None
    alpha_real    : opacity for real points / bars
    alpha_syn     : opacity for synthetic points / bars
    color_real    : colour for real data
    color_syn     : colour for synthetic data
    bins          : histogram bins for continuous / ordinal columns
    title         : overall figure title

    Returns
    -------
    matplotlib.figure.Figure — call .savefig("out.png") or plt.show() on it.

    Example
    -------
    >>> fig = compare_plot(real_df, population, schema)
    >>> fig.savefig("survey_comparison.png", dpi=150, bbox_inches="tight")
    """
    try:
        import matplotlib
        matplotlib.use("Agg")          # non-interactive backend (safe default)
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.lines import Line2D
    except ImportError:
        raise ImportError(
            "matplotlib is required for compare_plot(). "
            "Install it with: pip install 'universitybox[viz]'"
        )

    questions = [q for q in schema.questions if q.name in real_df.columns
                 and q.name in synthetic_df.columns]
    n_q = len(questions)

    # ------------------------------------------------------------------
    # Layout: 1 overview panel (PCA) + n_q marginal panels
    # ------------------------------------------------------------------
    n_cols = min(3, n_q)
    n_rows_marginal = (n_q + n_cols - 1) // n_cols
    total_rows = 2 + n_rows_marginal          # top: title row; row 1: PCA; rest: marginals

    if figsize is None:
        figsize = (6 * n_cols, 4 + 3.5 * n_rows_marginal)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)

    gs = gridspec.GridSpec(
        1 + n_rows_marginal, n_cols, figure=fig,
        hspace=0.45, wspace=0.35,
    )

    # ------------------------------------------------------------------
    # Panel 1: PCA / UMAP overview (spans all columns in first row)
    # ------------------------------------------------------------------
    ax_overview = fig.add_subplot(gs[0, :])

    X_real = _encode_df(real_df, schema)
    X_syn  = _encode_df(synthetic_df, schema)

    # Standardise before PCA
    mean_ = X_real.mean(axis=0)
    std_  = X_real.std(axis=0) + 1e-9
    X_real_s = (X_real - mean_) / std_
    X_syn_s  = (X_syn  - mean_) / std_

    if method == "umap":
        try:
            from umap import UMAP
            reducer = UMAP(n_components=n_components, random_state=42)
            X_all = np.vstack([X_real_s, X_syn_s])
            Z_all = reducer.fit_transform(X_all)
            Z_real = Z_all[:len(X_real)]
            Z_syn  = Z_all[len(X_real):]
            method_label = "UMAP"
        except ImportError:
            method = "pca"   # fallback

    if method == "pca":
        from scipy.linalg import svd
        # PCA via SVD on the combined matrix (fit on real, project synthetic)
        n_comp = min(n_components, X_real_s.shape[1], X_real_s.shape[0] - 1)
        U, S, Vt = svd(X_real_s, full_matrices=False)
        V = Vt[:n_comp].T                    # (p, n_comp)
        Z_real = X_real_s @ V                # (n_real, n_comp)
        Z_syn  = X_syn_s  @ V                # (n_syn,  n_comp)
        var_explained = (S[:n_comp] ** 2) / (S ** 2).sum() * 100
        method_label = "PCA"
        pc1_label = f"PC1 ({var_explained[0]:.1f}% var)"
        pc2_label = f"PC2 ({var_explained[min(1, n_comp-1)]:.1f}% var)" if n_comp > 1 else "PC2"
    else:
        pc1_label, pc2_label = "Dim 1", "Dim 2"

    ax_overview.scatter(Z_syn[:, 0],  Z_syn[:, 1]  if Z_syn.shape[1] > 1 else np.zeros(len(Z_syn)),
                        c=color_syn, alpha=alpha_syn, s=12, label=f"Synthetic (n={len(synthetic_df):,})")
    ax_overview.scatter(Z_real[:, 0], Z_real[:, 1] if Z_real.shape[1] > 1 else np.zeros(len(Z_real)),
                        c=color_real, alpha=alpha_real, s=40, marker="D",
                        label=f"Real (n={len(real_df)})", zorder=5)

    ax_overview.set_xlabel(pc1_label, fontsize=10)
    ax_overview.set_ylabel(pc2_label if Z_real.shape[1] > 1 else "", fontsize=10)
    ax_overview.set_title(f"{method_label} — Joint Distribution Overview", fontsize=11)
    ax_overview.legend(fontsize=9, framealpha=0.85)
    ax_overview.grid(True, alpha=0.3, linewidth=0.5)
    ax_overview.spines[["top", "right"]].set_visible(False)

    # ------------------------------------------------------------------
    # Panels 2+: marginal distributions per question
    # ------------------------------------------------------------------
    import pandas as pd

    for i, q in enumerate(questions):
        row = 1 + i // n_cols
        col = i % n_cols
        ax = fig.add_subplot(gs[row, col])

        col_real = real_df[q.name].dropna()
        col_syn  = synthetic_df[q.name].dropna()

        short_name = q.name if len(q.name) <= 32 else q.name[:29] + "…"

        if q.qtype == "categorical":
            # Grouped bar chart: normalised frequencies
            cats = q.categories or sorted(set(col_real.astype(str)) | set(col_syn.astype(str)))
            r_counts = col_real.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0)
            s_counts = col_syn.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0)

            x = np.arange(len(cats))
            w = 0.38
            ax.bar(x - w/2, r_counts.values, width=w, color=color_real, alpha=alpha_real, label="Real")
            ax.bar(x + w/2, s_counts.values, width=w, color=color_syn,  alpha=0.8,       label="Synthetic")
            ax.set_xticks(x)
            ax.set_xticklabels([str(c)[:10] for c in cats], rotation=30, ha="right", fontsize=7)
            ax.set_ylabel("Proportion", fontsize=8)

        elif q.qtype == "ordinal":
            lo, hi = q.scale  # type: ignore[misc]
            levels = list(range(lo, hi + 1))
            r_counts = pd.to_numeric(col_real, errors="coerce").value_counts(normalize=True).reindex(levels, fill_value=0)
            s_counts = pd.to_numeric(col_syn,  errors="coerce").value_counts(normalize=True).reindex(levels, fill_value=0)

            x = np.arange(len(levels))
            w = 0.38
            ax.bar(x - w/2, r_counts.values, width=w, color=color_real, alpha=alpha_real, label="Real")
            ax.bar(x + w/2, s_counts.values, width=w, color=color_syn,  alpha=0.8,       label="Synthetic")
            ax.set_xticks(x)
            ax.set_xticklabels([str(l) for l in levels], fontsize=8)
            ax.set_ylabel("Proportion", fontsize=8)

        else:  # continuous
            lo, hi = q.bounds  # type: ignore[misc]
            bin_edges = np.linspace(lo, hi, bins + 1)
            r_num = pd.to_numeric(col_real, errors="coerce").dropna().values
            s_num = pd.to_numeric(col_syn,  errors="coerce").dropna().values
            ax.hist(r_num, bins=bin_edges, density=True, color=color_real,
                    alpha=alpha_real, label="Real", edgecolor="white", linewidth=0.3)
            ax.hist(s_num, bins=bin_edges, density=True, color=color_syn,
                    alpha=0.6, label="Synthetic", edgecolor="white", linewidth=0.3)
            ax.set_ylabel("Density", fontsize=8)

        ax.set_title(short_name, fontsize=8, pad=4)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, alpha=0.25, linewidth=0.4, axis="y")

    # Shared legend for marginal panels
    legend_elements = [
        Line2D([0], [0], color=color_real, lw=0, marker="s", markersize=9,
               alpha=alpha_real, label=f"Real (n={len(real_df)})"),
        Line2D([0], [0], color=color_syn,  lw=0, marker="s", markersize=9,
               alpha=0.8, label=f"Synthetic (n={len(synthetic_df):,})"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=2, fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02))

    return fig
