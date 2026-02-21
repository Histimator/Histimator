"""Uncertainty estimation: Poisson, sumw2, and bootstrap.

Three variance estimators for binned data and their equivalence:

1. Poisson: Var(n_j) = n_j
   Valid for unweighted Poisson-distributed bin counts.

2. Barlow-Beeston (sumw2): Var(sumw_j) = sumw2_j = sum(w_i^2)
   Valid for weighted events; reduces to Poisson for unit weights.

3. Bootstrap: Var*(n_j) = sample variance across B resamples
   Nonparametric; converges to the other two in appropriate limits.

The convergence proof is empirical: for large samples, the three
estimators agree per-bin within the statistical uncertainty of the
bootstrap estimate itself.  This validates that any representation
producing bootstrap-estimated variances is consistent with the
histogram-based Barlow-Beeston treatment used in the likelihood.
"""

from __future__ import annotations

import numpy as np

from histimator.data import Dataset
from histimator.histograms import Histogram


def poisson_variance(histogram: Histogram) -> np.ndarray:
    """Poisson variance: Var(n_j) = n_j.

    This is exact for unweighted Poisson-distributed bin counts and
    serves as the baseline estimator.
    """
    return histogram.values.copy()


def sumw2_variance(histogram: Histogram) -> np.ndarray:
    """Barlow-Beeston variance: Var(sumw_j) = sumw2_j.

    For unweighted data, sumw2 = counts (Poisson).
    For weighted data, sumw2 = sum(w_i^2) per bin, which is the correct
    variance of the weighted sum under resampling.
    """
    return histogram.sumw2.copy()


def bootstrap_histogram_variance(
    dataset: Dataset,
    edges: np.ndarray,
    n_bootstrap: int = 1000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Bootstrap variance of bin counts.

    Generates n_bootstrap resamples of the dataset, bins each one,
    and returns the per-bin variance across resamples.

    For large n_bootstrap, this converges to:
      - Poisson variance (n_j) for unweighted data
      - sumw2 for weighted data

    Parameters
    ----------
    dataset : Dataset
        Raw event data.
    edges : array-like
        Bin edges for histogramming.
    n_bootstrap : int
        Number of bootstrap resamples.
    rng : Generator or None
        Random number generator.

    Returns
    -------
    np.ndarray
        Per-bin variance across bootstrap resamples.
    """
    if rng is None:
        rng = np.random.default_rng()

    edges = np.asarray(edges, dtype=np.float64)
    nbins = len(edges) - 1
    counts_matrix = np.empty((n_bootstrap, nbins), dtype=np.float64)

    for b in range(n_bootstrap):
        resample = dataset.bootstrap_resample(rng)
        h = resample.to_histogram(edges)
        counts_matrix[b] = h.values

    return np.var(counts_matrix, axis=0, ddof=1)


def compare_uncertainties(
    dataset: Dataset,
    edges: np.ndarray,
    n_bootstrap: int = 1000,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Compare all three variance estimators side by side.

    Returns a dict with keys 'poisson', 'sumw2', 'bootstrap', each
    mapping to a per-bin variance array.
    """
    if rng is None:
        rng = np.random.default_rng()

    h = dataset.to_histogram(edges)
    return {
        "poisson": poisson_variance(h),
        "sumw2": sumw2_variance(h),
        "bootstrap": bootstrap_histogram_variance(dataset, edges, n_bootstrap, rng),
    }
