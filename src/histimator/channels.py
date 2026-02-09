"""Channel: an analysis region combining samples with observed data.

A Channel represents a single histogram-based measurement region
(e.g. "signal region", "control region").  It holds one or more
Samples whose expected yields are summed to produce the total
prediction, and an observed-data histogram.
"""

from __future__ import annotations

import numpy as np

from histimator.histograms import Histogram
from histimator.parameters import Parameter
from histimator.samples import Sample


class Channel:
    """A single analysis region.

    Parameters
    ----------
    name : str
        Human-readable label (e.g. "SR", "CR_ttbar").
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._samples: list[Sample] = []
        self._data: Histogram | None = None

    # ---- sample management --------------------------------------------------

    def add_sample(self, sample: Sample) -> Channel:
        """Add a process to this channel. Returns self for chaining."""
        if self._samples:
            if sample.histogram.nbins != self._samples[0].histogram.nbins:
                raise ValueError(
                    f"Sample '{sample.name}' has {sample.histogram.nbins} bins "
                    f"but channel '{self.name}' expects "
                    f"{self._samples[0].histogram.nbins}"
                )
            if not np.array_equal(
                sample.histogram.edges, self._samples[0].histogram.edges
            ):
                raise ValueError(
                    f"Sample '{sample.name}' bin edges do not match "
                    f"channel '{self.name}'"
                )
        self._samples.append(sample)
        return self

    @property
    def samples(self) -> tuple[Sample, ...]:
        return tuple(self._samples)

    @property
    def nbins(self) -> int:
        if not self._samples:
            raise RuntimeError(f"Channel '{self.name}' has no samples")
        return self._samples[0].histogram.nbins

    @property
    def edges(self) -> np.ndarray:
        if not self._samples:
            raise RuntimeError(f"Channel '{self.name}' has no samples")
        return self._samples[0].histogram.edges

    # ---- data ---------------------------------------------------------------

    def set_data(self, data: Histogram | np.ndarray | list) -> Channel:
        """Set the observed data for this channel.

        Accepts a Histogram, a numpy array of bin counts, or a plain list.
        """
        if isinstance(data, Histogram):
            if data.nbins != self.nbins:
                raise ValueError(
                    f"Data has {data.nbins} bins, channel expects {self.nbins}"
                )
            self._data = data
        else:
            arr = np.asarray(data, dtype=np.float64)
            if arr.shape[0] != self.nbins:
                raise ValueError(
                    f"Data has {arr.shape[0]} bins, channel expects {self.nbins}"
                )
            self._data = Histogram(arr, self.edges)
        return self

    @property
    def data(self) -> Histogram | None:
        return self._data

    # ---- parameters ---------------------------------------------------------

    @property
    def parameters(self) -> list[Parameter]:
        """All unique parameters across all samples in this channel."""
        seen: set[str] = set()
        params: list[Parameter] = []
        for sample in self._samples:
            for p in sample.parameters:
                if p.name not in seen:
                    seen.add(p.name)
                    params.append(p)
        return params

    # ---- evaluation ---------------------------------------------------------

    def expected(self, params: dict[str, float]) -> np.ndarray:
        """Total expected bin yields (sum of all samples)."""
        total = np.zeros(self.nbins, dtype=np.float64)
        for sample in self._samples:
            total += sample.expected(params)
        return total

    def __repr__(self) -> str:
        n_data = "yes" if self._data is not None else "no"
        return (
            f"Channel({self.name!r}, samples={len(self._samples)}, "
            f"nbins={self.nbins if self._samples else '?'}, data={n_data})"
        )
