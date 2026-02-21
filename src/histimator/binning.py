"""Binning strategies for density estimation.

Implements principled bin width selection methods from nonparametric statistics.

The cross-validation approach follows Breheny STA 621 (Lecture 10-16):
the leave-one-out criterion for histograms has a closed form,

    J_hat(h) = 2 / (h*(n-1)) - (n+1) / (h*(n-1)) * sum(p_hat_j^2)

where p_hat_j = n_j / n is the relative frequency in bin j and h is
the common bin width.  The sum runs over bins within the data range.
The optimal h minimises J_hat(h) over a grid of candidate widths.

The bias-variance decomposition of the integrated squared error (ISE)
shows that J_hat is an unbiased estimator of ISE minus an irreducible
term, so minimising J_hat minimises the expected ISE.

Also included are standard textbook rules:
  - Normal reference (Scott's rule): h = 3.49 * sigma * n^(-1/3)
  - Sturges:  m = 1 + ceil(log2(n))
  - Freedman-Diaconis:  h = 2 * IQR * n^(-1/3)
"""

from __future__ import annotations

import numpy as np

from histimator.data import Dataset


def _data_range(dataset: Dataset) -> tuple[float, float]:
    """Return (min, max) of dataset values."""
    if len(dataset) == 0:
        raise ValueError("Cannot compute binning for empty dataset")
    return float(dataset.values.min()), float(dataset.values.max())


def cross_validation_binwidth(
    dataset: Dataset,
    h_range: tuple[float, float] | None = None,
    n_grid: int = 200,
) -> float:
    """Find optimal bin width by leave-one-out cross-validation.

    Evaluates the Breheny LOO criterion J_hat(h) on a grid and returns
    the h that minimises it.

    Parameters
    ----------
    dataset : Dataset
        Raw event data (weights are ignored for CV; uses event counts).
    h_range : (float, float) or None
        Range of candidate bin widths.  If None, uses a heuristic range
        from data_range/(4*n) to data_range/2.
    n_grid : int
        Number of grid points to evaluate.

    Returns
    -------
    float
        Optimal bin width.
    """
    vals = dataset.values
    n = len(vals)
    if n < 2:
        raise ValueError("Cross-validation requires at least 2 events")

    lo, hi = _data_range(dataset)
    data_range = hi - lo
    if data_range == 0:
        return 1.0  # all events at one point

    if h_range is None:
        h_min = max(data_range / (4 * n), data_range / 500)
        h_max = data_range / 2
        h_range = (h_min, h_max)

    h_candidates = np.linspace(h_range[0], h_range[1], n_grid)
    j_scores = np.empty(n_grid)

    for i, h in enumerate(h_candidates):
        edges = np.arange(lo, hi + h, h)
        if len(edges) < 2:
            edges = np.array([lo, hi + h])
        counts, _ = np.histogram(vals, bins=edges)
        m = len(counts)
        p_hat = counts / n
        sum_p2 = np.sum(p_hat ** 2)

        # J_hat(h) = 2/(h*(n-1)) - (n+1)/(h*(n-1)) * sum(p_j^2)
        j_scores[i] = 2.0 / (h * (n - 1)) - (n + 1) / (h * (n - 1)) * sum_p2

    return float(h_candidates[np.argmin(j_scores)])


def normal_reference_binwidth(dataset: Dataset) -> float:
    """Normal reference rule (Scott's rule): h = 3.49 * sigma * n^(-1/3).

    This is optimal when the underlying density is Gaussian.  It tends
    to oversmooth for multimodal distributions but provides a fast,
    parameter-free default.
    """
    n = len(dataset)
    if n < 2:
        raise ValueError("Normal reference rule requires at least 2 events")
    sigma = float(np.std(dataset.values, ddof=1))
    if sigma == 0:
        return 1.0
    return 3.49 * sigma * n ** (-1.0 / 3.0)


def sturges_edges(dataset: Dataset) -> np.ndarray:
    """Sturges' rule: m = 1 + ceil(log2(n)) bins."""
    n = len(dataset)
    if n < 1:
        raise ValueError("Sturges' rule requires at least 1 event")
    lo, hi = _data_range(dataset)
    m = max(1, 1 + int(np.ceil(np.log2(n))))
    return np.linspace(lo, hi, m + 1)


def scott_edges(dataset: Dataset) -> np.ndarray:
    """Scott's rule: h = 3.49 * sigma * n^(-1/3)."""
    h = normal_reference_binwidth(dataset)
    lo, hi = _data_range(dataset)
    # Extend range slightly to ensure all data is captured
    m = max(1, int(np.ceil((hi - lo) / h)))
    return np.linspace(lo, hi, m + 1)


def freedman_diaconis_edges(dataset: Dataset) -> np.ndarray:
    """Freedman-Diaconis rule: h = 2 * IQR * n^(-1/3)."""
    n = len(dataset)
    if n < 2:
        raise ValueError("Freedman-Diaconis requires at least 2 events")
    vals = dataset.values
    iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
    if iqr == 0:
        return sturges_edges(dataset)
    h = 2.0 * iqr * n ** (-1.0 / 3.0)
    lo, hi = _data_range(dataset)
    m = max(1, int(np.ceil((hi - lo) / h)))
    return np.linspace(lo, hi, m + 1)


def auto_edges(dataset: Dataset, method: str = "cv") -> np.ndarray:
    """Dispatcher: compute bin edges using the named strategy.

    Parameters
    ----------
    dataset : Dataset
        Raw event data.
    method : str
        One of "cv" (cross-validation), "normal_reference", "scott",
        "sturges", "freedman_diaconis" (or "fd").

    Returns
    -------
    np.ndarray
        Bin edges.
    """
    method = method.lower().strip()
    lo, hi = _data_range(dataset)

    if method == "cv":
        h = cross_validation_binwidth(dataset)
        m = max(1, int(np.ceil((hi - lo) / h)))
        return np.linspace(lo, hi, m + 1)
    elif method in ("normal_reference", "scott"):
        return scott_edges(dataset)
    elif method == "sturges":
        return sturges_edges(dataset)
    elif method in ("freedman_diaconis", "fd"):
        return freedman_diaconis_edges(dataset)
    else:
        raise ValueError(
            f"Unknown binning method '{method}'. "
            f"Choose from: cv, normal_reference, scott, sturges, freedman_diaconis"
        )
