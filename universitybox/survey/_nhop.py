"""
universitybox.survey._nhop
============================
Stage 3: Seed selection, NHOP rejection, and density-proportional oversampling.

Three sequential operations applied to the copula-synthesised batch:

1. k-means++ Seed Selection
   -------------------------
   Select n_seeds representative "prototype" points from the synthesised batch
   using the k-means++ distance-proportional seeding protocol.  Seeds span the
   space well without clustering near the mode.

   Algorithm (same as used in DNA's RBF stage):
     - Seed[0] = random point.
     - For each subsequent seed: compute squared distance of every candidate to
       its nearest existing seed; sample proportional to that distance^2.

2. NHOP — Neighbourhood-based Outlier Pruning
   ---------------------------------------------
   For each synthesised point x, compute the distance to its k-th nearest
   neighbour in the *real* sample.  If this distance exceeds a threshold
   (default: nhop_pct-th percentile of the same kNN distance distribution
   computed on the real sample itself), the point is rejected as out-of-distribution.

   This prevents the copula from generating response patterns that never appear
   in the real data (e.g., an ordinal variable simultaneously at its floor while
   a correlated variable is at its ceiling).

3. Density-Proportional Oversampling
   ------------------------------------
   Resample the surviving synthesised points (with replacement) to reach the
   target N, weighting each point by its estimated local density.

   Local density proxy: inverse of average distance to k nearest neighbours
   within the surviving set.  Points in dense regions are favoured — the final
   sample better reflects the actual distribution of real responses.

All operations work on a numeric embedding of the mixed-type data:
  - Categorical → integer code (0, 1, 2, ...)
  - Ordinal     → integer value
  - Continuous  → float
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ._schema import Question


# ---------------------------------------------------------------------------
# Helper: encode mixed-type DataFrame rows to a float matrix
# ---------------------------------------------------------------------------

def _encode(rows: np.ndarray, questions: List[Question]) -> np.ndarray:
    """
    Convert (n, p) object array of mixed survey values to (n, p) float matrix.
    Categorical → integer code mapped from declared categories.
    Ordinal / continuous → float as-is.
    """
    n, p = rows.shape
    out = np.empty((n, p), dtype=float)
    for j, q in enumerate(questions):
        col = rows[:, j]
        if q.qtype == "categorical":
            cat_map = {c: i for i, c in enumerate(q.categories)}  # type: ignore[arg-type]
            out[:, j] = np.array([cat_map.get(str(v), 0) for v in col], dtype=float)
        else:
            out[:, j] = col.astype(float)
    return out


# ---------------------------------------------------------------------------
# 1. k-means++ seed selection
# ---------------------------------------------------------------------------

def kmeans_plus_seeds(
    X: np.ndarray,
    n_seeds: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Select n_seeds representative seeds from X using k-means++ seeding.

    Parameters
    ----------
    X       : (n, p) float array
    n_seeds : number of seeds to select
    rng     : numpy Generator

    Returns
    -------
    seeds : (n_seeds, p) float array — rows of X
    """
    n = len(X)
    n_seeds = min(n_seeds, n)
    chosen = [int(rng.integers(0, n))]
    dists = np.full(n, np.inf)

    for _ in range(1, n_seeds):
        last = X[chosen[-1]]
        d2 = np.sum((X - last) ** 2, axis=1)
        dists = np.minimum(dists, d2)
        total = dists.sum()
        if total == 0.0:
            # All remaining points are equidistant (e.g. all-categorical data) — sample uniformly
            probs = np.ones(n) / n
        else:
            probs = dists / total
        chosen.append(int(rng.choice(n, p=probs)))

    return X[chosen]


# ---------------------------------------------------------------------------
# 2. NHOP — Neighbourhood-based Outlier Pruning
# ---------------------------------------------------------------------------

def _knn_distances(query: np.ndarray, reference: np.ndarray, k: int) -> np.ndarray:
    """
    For each point in query, return the distance to its k-th nearest neighbour
    in reference.  Uses a simple O(n*m) loop; adequate for survey-scale data.
    """
    k = min(k, len(reference))
    dists = np.empty(len(query))
    for i, q in enumerate(query):
        d = np.sqrt(np.sum((reference - q) ** 2, axis=1))
        d.sort()
        dists[i] = d[k - 1]
    return dists


def nhop_filter(
    synthetic_enc: np.ndarray,
    real_enc: np.ndarray,
    k: int = 5,
    nhop_pct: float = 99.0,
) -> np.ndarray:
    """
    NHOP rejection: keep synthetic points whose kNN distance to the real sample
    is within the nhop_pct-th percentile of the real sample's own kNN distances.

    Parameters
    ----------
    synthetic_enc : (n_syn, p) float array — encoded synthesised points
    real_enc      : (n_real, p) float array — encoded real responses
    k             : number of nearest neighbours
    nhop_pct      : percentile threshold (0-100); higher = more permissive

    Returns
    -------
    Boolean mask of length n_syn — True means keep.
    """
    if len(real_enc) == 0 or len(synthetic_enc) == 0:
        return np.ones(len(synthetic_enc), dtype=bool)

    # Compute threshold from real sample's self-kNN distances
    real_knn = _knn_distances(real_enc, real_enc, k=min(k + 1, len(real_enc)))
    threshold = float(np.percentile(real_knn, nhop_pct))

    # Compute kNN distances from synthetic to real
    syn_knn = _knn_distances(synthetic_enc, real_enc, k=min(k, len(real_enc)))

    return syn_knn <= threshold


# ---------------------------------------------------------------------------
# 3. Density-proportional oversampling
# ---------------------------------------------------------------------------

def density_oversample(
    candidates: np.ndarray,
    target_n: int,
    rng: np.random.Generator,
    k: int = 5,
) -> np.ndarray:
    """
    Resample `candidates` (with replacement) to reach target_n rows, weighting
    by estimated local density (inverse of average kNN distance within set).

    Parameters
    ----------
    candidates : (m, p) float array of surviving synthesised encoded rows
    target_n   : desired output size
    rng        : numpy Generator
    k          : neighbourhood size for density estimation

    Returns
    -------
    Indices (length target_n) into candidates — for use with the original
    mixed-type array.
    """
    m = len(candidates)
    if m == 0:
        raise ValueError("No candidates survived NHOP filtering. "
                         "Try increasing nhop_pct or reducing k.")
    if m >= target_n:
        # No oversampling needed — just subsample
        return rng.choice(m, size=target_n, replace=False)

    # Estimate local density via average kNN distance
    k_eff = min(k, m - 1) if m > 1 else 1
    avg_dists = np.empty(m)
    for i in range(m):
        d = np.sqrt(np.sum((candidates - candidates[i]) ** 2, axis=1))
        d.sort()
        avg_dists[i] = d[1 : k_eff + 1].mean() if k_eff > 0 else 1.0

    # Density = 1 / avg_dist; normalise to probability
    density = 1.0 / (avg_dists + 1e-9)
    probs = density / density.sum()
    return rng.choice(m, size=target_n, replace=True, p=probs)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def apply_stage3(
    synthetic_raw: np.ndarray,       # (n_syn, p) object array from copula
    real_raw: np.ndarray,             # (n_real, p) object array of real data
    questions: List[Question],
    target_n: int,
    n_seeds: int,
    k_nhop: int,
    nhop_pct: float,
    k_density: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Full Stage 3 pipeline.

    Returns (target_n, p) object array of final synthesised responses.
    """
    # Encode to float for distance computations
    syn_enc = _encode(synthetic_raw, questions)
    real_enc = _encode(real_raw, questions)

    # 1. Seed selection (informational — seeds are used internally for density seeding)
    if n_seeds > 0 and len(syn_enc) >= n_seeds:
        # Seeds anchor the density estimate — subsample candidates near seeds first
        seeds = kmeans_plus_seeds(syn_enc, n_seeds=n_seeds, rng=rng)
        # Expand: each seed attracts its closest synthetic points
        dists_to_seeds = np.min(
            np.stack([
                np.sum((syn_enc - s) ** 2, axis=1) for s in seeds
            ], axis=1),
            axis=1,
        )
        # Keep top 80% closest to any seed (pre-filter before NHOP)
        threshold_80 = np.percentile(dists_to_seeds, 80.0)
        preseed_mask = dists_to_seeds <= threshold_80
        syn_enc_f = syn_enc[preseed_mask]
        synthetic_raw_f = synthetic_raw[preseed_mask]
    else:
        syn_enc_f = syn_enc
        synthetic_raw_f = synthetic_raw

    # 2. NHOP filtering
    keep_mask = nhop_filter(syn_enc_f, real_enc, k=k_nhop, nhop_pct=nhop_pct)
    syn_enc_kept = syn_enc_f[keep_mask]
    synthetic_kept = synthetic_raw_f[keep_mask]

    # Fallback if too many were pruned
    if len(synthetic_kept) == 0:
        synthetic_kept = synthetic_raw
        syn_enc_kept = syn_enc

    # 3. Density-proportional oversampling → target_n rows
    idx = density_oversample(syn_enc_kept, target_n=target_n, rng=rng, k=k_density)
    return synthetic_kept[idx]
