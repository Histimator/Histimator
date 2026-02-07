"""Histogram container with validation, arithmetic, and integration.

The ``Histogram`` wraps a 1-D histogram (bin contents + bin edges) and
provides the basic operations needed by the template-fitting machinery:
element-wise arithmetic, integration over sub-ranges, and bin-centre /
bin-width accessors.  All heavy lifting is done with vectorised numpy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


class Histogram:
    """Immutable 1-D histogram.

    Parameters
    ----------
    values : array_like
        Bin contents.  Length *n*.
    edges : array_like
        Bin edges.  Length *n + 1*, monotonically increasing.
    sumw2 : array_like or None
        Sum-of-weights-squared per bin (for weighted templates).
        If ``None``, Poisson errors are assumed (``sumw2 = values``).

    Examples
    --------
    >>> import numpy as np
    >>> h = Histogram([10, 20, 30], [0.0, 1.0, 2.0, 3.0])
    >>> h.nbins
    3
    >>> h.widths
    array([1., 1., 1.])
    """

    __slots__ = ("_values", "_edges", "_sumw2")

    def __init__(
        self,
        values: ArrayLike,
        edges: ArrayLike,
        sumw2: ArrayLike | None = None,
    ) -> None:
        v = np.asarray(values, dtype=np.float64)
        e = np.asarray(edges, dtype=np.float64)

        if v.ndim != 1:
            raise ValueError(f"values must be 1-D, got shape {v.shape}")
        if e.ndim != 1:
            raise ValueError(f"edges must be 1-D, got shape {e.shape}")
        if e.shape[0] != v.shape[0] + 1:
            raise ValueError(
                f"edges length ({e.shape[0]}) must be values length + 1 "
                f"({v.shape[0] + 1})"
            )
        if not np.all(np.diff(e) > 0):
            raise ValueError("edges must be strictly monotonically increasing")

        self._values = v
        self._values.flags.writeable = False
        self._edges = e
        self._edges.flags.writeable = False

        if sumw2 is not None:
            sw2 = np.asarray(sumw2, dtype=np.float64)
            if sw2.shape != v.shape:
                raise ValueError("sumw2 must have the same shape as values")
            self._sumw2 = sw2
            self._sumw2.flags.writeable = False
        else:
            self._sumw2 = None

    # --- properties -----------------------------------------------------------

    @property
    def values(self) -> np.ndarray:
        """Bin contents (read-only view)."""
        return self._values

    @property
    def edges(self) -> np.ndarray:
        """Bin edges (read-only view)."""
        return self._edges

    @property
    def sumw2(self) -> np.ndarray:
        """Sum-of-weights-squared per bin."""
        if self._sumw2 is not None:
            return self._sumw2
        return self._values  # Poisson assumption

    @property
    def nbins(self) -> int:
        return self._values.shape[0]

    @property
    def centres(self) -> np.ndarray:
        """Bin centres."""
        return 0.5 * (self._edges[:-1] + self._edges[1:])

    @property
    def widths(self) -> np.ndarray:
        """Bin widths."""
        return np.diff(self._edges)

    @property
    def density(self) -> np.ndarray:
        """Bin contents divided by bin widths (rate per unit)."""
        return self._values / self.widths

    @property
    def total(self) -> float:
        """Sum of all bin contents."""
        return float(self._values.sum())

    # --- integration ----------------------------------------------------------

    def integrate(
        self, lo: float | None = None, hi: float | None = None
    ) -> float:
        """Integrate bin contents over [lo, hi].

        Partial bins are included proportionally.  If *lo* or *hi* is
        ``None``, the full histogram range on that side is used.
        """
        if lo is None:
            lo = float(self._edges[0])
        if hi is None:
            hi = float(self._edges[-1])
        lo = max(lo, float(self._edges[0]))
        hi = min(hi, float(self._edges[-1]))
        if lo >= hi:
            return 0.0

        total = 0.0
        for i in range(self.nbins):
            blo = float(self._edges[i])
            bhi = float(self._edges[i + 1])
            overlap_lo = max(lo, blo)
            overlap_hi = min(hi, bhi)
            if overlap_lo < overlap_hi:
                frac = (overlap_hi - overlap_lo) / (bhi - blo)
                total += self._values[i] * frac
        return total

    # --- constructors ---------------------------------------------------------

    @classmethod
    def from_numpy(cls, hist_tuple: tuple) -> Histogram:
        """Create from a ``numpy.histogram`` return value ``(counts, edges)``."""
        if len(hist_tuple) != 2:
            raise ValueError("Expected a (counts, edges) tuple")
        return cls(hist_tuple[0], hist_tuple[1])

    # --- arithmetic (return new Histograms) -----------------------------------

    def _check_compatible(self, other: Histogram) -> None:
        if not np.array_equal(self._edges, other._edges):
            raise ValueError("Histograms must have identical bin edges")

    def __add__(self, other: Histogram | float | np.ndarray) -> Histogram:
        if isinstance(other, Histogram):
            self._check_compatible(other)
            new_vals = self._values + other._values
            sw2 = self.sumw2 + other.sumw2
            return Histogram(new_vals, self._edges.copy(), sumw2=sw2)
        scalar = np.asarray(other, dtype=np.float64)
        return Histogram(self._values + scalar, self._edges.copy())

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, scalar: float | np.ndarray) -> Histogram:
        s = np.asarray(scalar, dtype=np.float64)
        new_vals = self._values * s
        sw2 = self.sumw2 * s**2
        return Histogram(new_vals, self._edges.copy(), sumw2=sw2)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, other: Histogram | float) -> Histogram:
        if isinstance(other, Histogram):
            self._check_compatible(other)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(other._values != 0, self._values / other._values, 0.0)
            return Histogram(ratio, self._edges.copy())
        return self.__mul__(1.0 / other)

    # --- comparison / hashing -------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Histogram):
            return NotImplemented
        return (
            np.array_equal(self._values, other._values)
            and np.array_equal(self._edges, other._edges)
        )

    def __repr__(self) -> str:
        return (
            f"Histogram(nbins={self.nbins}, "
            f"range=({self._edges[0]:.3g}, {self._edges[-1]:.3g}), "
            f"total={self.total:.6g})"
        )
