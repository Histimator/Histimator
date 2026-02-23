"""Channel: an analysis region combining samples with observed data."""
from __future__ import annotations

import numpy as np

from histimator.histograms import Histogram
from histimator.parameters import Parameter
from histimator.samples import Sample


class Channel:
    def __init__(self, name: str) -> None:
        self.name = name
        self._samples: list[Sample] = []
        self._data: Histogram | None = None

    def add_sample(self, sample: Sample) -> Channel:
        if self._samples:
            if sample.template.nbins != self._samples[0].template.nbins:
                raise ValueError(
                    f"Sample '{sample.name}' has {sample.template.nbins} bins "
                    f"but channel '{self.name}' expects {self._samples[0].template.nbins}"
                )
            if not np.array_equal(sample.template.edges, self._samples[0].template.edges):
                raise ValueError(
                    f"Sample '{sample.name}' bin edges do not match channel '{self.name}'"
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
        return self._samples[0].template.nbins

    @property
    def edges(self) -> np.ndarray:
        if not self._samples:
            raise RuntimeError(f"Channel '{self.name}' has no samples")
        return self._samples[0].template.edges

    def set_data(self, data) -> Channel:
        if isinstance(data, Histogram):
            if data.nbins != self.nbins:
                raise ValueError(f"Data has {data.nbins} bins, channel expects {self.nbins}")
            self._data = data
        else:
            arr = np.asarray(data, dtype=np.float64)
            if arr.shape[0] != self.nbins:
                raise ValueError(f"Data has {arr.shape[0]} bins, channel expects {self.nbins}")
            self._data = Histogram(arr, self.edges)
        return self

    @property
    def data(self) -> Histogram | None:
        return self._data

    @property
    def parameters(self) -> list[Parameter]:
        seen: set[str] = set()
        params: list[Parameter] = []
        for sample in self._samples:
            for p in sample.parameters:
                if p.name not in seen:
                    seen.add(p.name)
                    params.append(p)
        return params

    def expected(self, params: dict[str, float]) -> np.ndarray:
        total = np.zeros(self.nbins, dtype=np.float64)
        for sample in self._samples:
            total += sample.expected(params)
        return total

    def __repr__(self) -> str:
        n_data = "yes" if self._data is not None else "no"
        return f"Channel({self.name!r}, samples={len(self._samples)}, data={n_data})"

    # ------------------------------------------------------------------
    # Unbinned data
    # ------------------------------------------------------------------

    def set_unbinned_data(self, events) -> Channel:
        """Store raw event values for unbinned likelihood evaluation.

        Parameters
        ----------
        events : array-like or Dataset
            1-D array of observed event values, or a Dataset object.
            Events outside the template domain [edges[0], edges[-1]]
            are silently dropped to match standard practice.

        Returns
        -------
        Channel
            Self, for method chaining.
        """
        from histimator.data import Dataset as _Dataset

        if isinstance(events, _Dataset):
            raw = events.values
        else:
            raw = np.asarray(events, dtype=np.float64).ravel()

        # Clip to template domain
        lo, hi = float(self.edges[0]), float(self.edges[-1])
        mask = (raw >= lo) & (raw <= hi)
        self._unbinned_data = raw[mask]
        return self

    @property
    def unbinned_data(self) -> np.ndarray | None:
        """Raw event values for unbinned likelihood, or None."""
        return getattr(self, "_unbinned_data", None)

    def set_data_from_dataset(self, dataset) -> Channel:
        """Bin a Dataset using this channel's edges and set as observed data.

        The dataset is binned using the edges from the first sample,
        ensuring consistency between template and data binning.
        """
        from histimator.data import Dataset as _Dataset
        if not isinstance(dataset, _Dataset):
            raise TypeError(f"Expected Dataset, got {type(dataset).__name__}")
        histogram = dataset.to_histogram(edges=self.edges)
        return self.set_data(histogram)

    def add_data_and_samples_from_datasets(
        self, datasets_dict: dict, data_dataset=None, edges=None, strategy=None,
        template_type: str = "binned", **template_kwargs,
    ) -> Channel:
        """Build a complete channel from raw datasets.

        This is the "data loader" workflow: go from raw data to a
        ready-to-fit channel in one call.

        Parameters
        ----------
        datasets_dict : dict[str, Dataset]
            Mapping of sample name to Dataset.
        data_dataset : Dataset or None
            Observed data dataset.  If None, sums the sample datasets.
        edges : array-like or None
            Explicit bin edges.
        strategy : str or None
            Binning strategy for auto edge computation.
        template_type : str
            "binned" (default), "gp", or "bspline".  Applied to all
            samples in the channel.
        **template_kwargs
            Extra keyword arguments forwarded to the template constructor.

        Returns
        -------
        Channel
            Self, with samples and data configured.
        """
        from histimator.data import Dataset as _Dataset

        # Determine edges
        if edges is None and strategy is not None:
            all_values = np.concatenate(
                [ds.values for ds in datasets_dict.values()]
            )
            from histimator.binning import auto_edges
            edges = auto_edges(_Dataset(all_values), method=strategy)
        elif edges is None:
            raise ValueError("Either edges or strategy must be provided")

        # Create and add samples
        for name, ds in datasets_dict.items():
            sample = Sample.from_dataset(name, ds, edges=edges,
                                         template_type=template_type,
                                         **template_kwargs)
            self.add_sample(sample)

        # Set observed data
        if data_dataset is not None:
            self.set_data_from_dataset(data_dataset)
        else:
            # Use sum of sample expectations as pseudo-data
            total = self.expected({})
            self.set_data(total)

        return self
