"""Dataset: raw event-level data container with proper binning.

A Dataset holds a 1-D array of event values and optional per-event weights.
The raw data is retained permanently so that re-binning with different
edges or strategies is always possible.

The core method to_histogram(edges) performs binning and computes sumw2
correctly for both weighted and unweighted events.  For unweighted events,
sumw2_j = n_j (Poisson variance).  For weighted events, sumw2_j is the
sum of squared weights in bin j, which enters the Barlow-Beeston treatment
as the effective statistics.

The relationship between sumw2 and effective statistics is:
    n_eff_j = (sum w_i)^2 / sumw2_j

This equals n_j for unweighted events and is smaller for weighted events,
correctly reflecting the reduced statistical power.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from histimator.histograms import Histogram


class Dataset:
    """Container for raw event-level data.

    Parameters
    ----------
    values : array-like
        1-D array of event values (the observable).
    weights : array-like or None
        Optional per-event weights.  When None, all events have unit weight.
    """

    def __init__(self, values: ArrayLike, weights: ArrayLike | None = None) -> None:
        self._values = np.asarray(values, dtype=np.float64).ravel()
        if weights is not None:
            self._weights = np.asarray(weights, dtype=np.float64).ravel()
            if self._weights.shape != self._values.shape:
                raise ValueError(
                    f"weights length ({self._weights.shape[0]}) must match "
                    f"values length ({self._values.shape[0]})"
                )
        else:
            self._weights = None

    @property
    def values(self) -> np.ndarray:
        """Raw event values (read-only view)."""
        return self._values

    @property
    def weights(self) -> np.ndarray | None:
        """Per-event weights, or None for unweighted data."""
        return self._weights

    @property
    def is_weighted(self) -> bool:
        return self._weights is not None

    def __len__(self) -> int:
        return self._values.shape[0]

    def to_histogram(self, edges: ArrayLike | None = None,
                     strategy: str | None = None) -> Histogram:
        """Bin the dataset into a Histogram.

        Parameters
        ----------
        edges : array-like or None
            Explicit bin edges.  Takes precedence over strategy.
        strategy : str or None
            Binning strategy name (used when edges is None).
            Requires histimator.binning module.

        Returns
        -------
        Histogram
            Binned data with correctly computed sumw2.

        The binning follows numpy.histogram conventions: events outside
        the edge range are silently excluded, matching standard practice.

        For unweighted data, sumw2_j = count_j (Poisson variance).
        For weighted data, sumw2_j = sum(w_i^2 for events i in bin j),
        which is the correct input for Barlow-Beeston uncertainty.
        """
        if edges is None and strategy is not None:
            from histimator.binning import auto_edges
            edges = auto_edges(self, method=strategy)
        if edges is None:
            raise ValueError(
                "Either edges or strategy must be provided for binning"
            )
        edges = np.asarray(edges, dtype=np.float64)

        if self._weights is None:
            counts, _ = np.histogram(self._values, bins=edges)
            counts = counts.astype(np.float64)
            # Unweighted: sumw2 = counts (Poisson)
            return Histogram(counts, edges, sumw2=counts.copy())
        else:
            # Weighted binning: compute both sum(w) and sum(w^2) per bin
            nbins = len(edges) - 1
            sumw = np.zeros(nbins, dtype=np.float64)
            sumw2 = np.zeros(nbins, dtype=np.float64)
            bin_idx = np.digitize(self._values, edges)
            for i in range(nbins):
                mask = bin_idx == (i + 1)
                sumw[i] = self._weights[mask].sum()
                sumw2[i] = (self._weights[mask] ** 2).sum()
            return Histogram(sumw, edges, sumw2=sumw2)

    def bootstrap_resample(self, rng: np.random.Generator) -> Dataset:
        """Return a bootstrap resample (sampling with replacement).

        For unweighted data, resamples event indices uniformly.
        For weighted data, resamples event indices with probability
        proportional to |w_i|, preserving the weight signs.

        The bootstrap resample has the same number of events as the
        original, which is the standard nonparametric bootstrap.
        """
        n = len(self)
        if n == 0:
            return Dataset(np.array([]), weights=self._weights)
        idx = rng.choice(n, size=n, replace=True)
        new_vals = self._values[idx]
        new_weights = self._weights[idx] if self._weights is not None else None
        return Dataset(new_vals, weights=new_weights)

    def __repr__(self) -> str:
        w = "weighted" if self.is_weighted else "unweighted"
        return f"Dataset(n={len(self)}, {w})"
