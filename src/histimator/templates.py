"""Template abstraction: separates representation from evaluation.

A Template represents a probability density f(x). When evaluated on bins,
it returns expected counts via integration of f over each bin. The modifier
system operates on these expected counts regardless of underlying
representation, enabling future non-parametric methods (KDE, GP) while
maintaining backward compatibility with histogram-based workflows.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from histimator.histograms import Histogram


@runtime_checkable
class Template(Protocol):
    """Protocol for all template types.

    A Template must provide:
    - counts(): baseline expected counts before modifiers
    - edges: bin edges array
    - nbins: number of bins
    - density(): probability density at bin centres (integrates to ~1)
    """

    @property
    def edges(self) -> np.ndarray: ...

    @property
    def nbins(self) -> int: ...

    def counts(self) -> np.ndarray:
        """Return baseline expected counts (copy). Modifiers chain on top."""
        ...

    def density(self) -> np.ndarray:
        """Return probability density at bin centres: f_hat(x_j).

        For a proper density estimator, sum(density * widths) ~ 1.
        """
        ...


class BinnedTemplate:
    """Wraps a Histogram as a Template.

    This is the simplest Template: the density is piecewise constant
    within each bin, and counts() returns the histogram values directly.
    The density interpretation is f_hat(x) = n_j / (N * h_j) where
    n_j is the bin count, N is total, and h_j is bin width.
    """

    __slots__ = ("_histogram",)

    def __init__(self, histogram: Histogram) -> None:
        if not isinstance(histogram, Histogram):
            raise TypeError(
                f"BinnedTemplate requires a Histogram, got {type(histogram).__name__}"
            )
        self._histogram = histogram

    @property
    def histogram(self) -> Histogram:
        return self._histogram

    @property
    def edges(self) -> np.ndarray:
        return self._histogram.edges

    @property
    def nbins(self) -> int:
        return self._histogram.nbins

    @property
    def sumw2(self) -> np.ndarray:
        return self._histogram.sumw2

    def counts(self) -> np.ndarray:
        """Return bin values as expected counts (copy)."""
        return self._histogram.values.copy()

    def density(self) -> np.ndarray:
        """Piecewise constant density: n_j / (N * h_j)."""
        total = self._histogram.total
        if total == 0:
            return np.zeros(self.nbins)
        return self._histogram.values / (total * self._histogram.widths)

    def __repr__(self) -> str:
        return (
            f"BinnedTemplate(nbins={self.nbins}, "
            f"range=({self.edges[0]:.3g}, {self.edges[-1]:.3g}), "
            f"total={self._histogram.total:.6g})"
        )
