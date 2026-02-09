"""Model: the top-level statistical model.

A Model is a collection of Channels. It exposes a single ``expected``
method that returns the concatenated predicted bin counts across all
channels for a given set of parameter values, and a ``data`` property
that returns the concatenated observed counts.

The parameter bookkeeping (merging parameters with the same name from
different channels, providing initial values for the minimiser) is
handled here.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from histimator.channels import Channel
from histimator.parameters import Parameter


class Model:
    """Top-level statistical model combining one or more channels.

    Parameters
    ----------
    name : str
        Model label (purely informational).
    """

    def __init__(self, name: str = "model") -> None:
        self.name = name
        self._channels: list[Channel] = []

    # ---- channel management ------------------------------------------------

    def add_channel(self, channel: Channel) -> Model:
        """Append a channel to the model. Returns self for chaining."""
        if channel.data is None:
            raise ValueError(
                f"Channel '{channel.name}' has no observed data. "
                "Call channel.set_data() before adding to model."
            )
        self._channels.append(channel)
        return self

    @property
    def channels(self) -> tuple[Channel, ...]:
        return tuple(self._channels)

    @property
    def n_channels(self) -> int:
        return len(self._channels)

    # ---- parameters --------------------------------------------------------

    @property
    def parameters(self) -> list[Parameter]:
        """Unique parameters across all channels (order-preserving)."""
        seen: OrderedDict[str, Parameter] = OrderedDict()
        for ch in self._channels:
            for p in ch.parameters:
                if p.name not in seen:
                    seen[p.name] = p
        return list(seen.values())

    @property
    def parameter_names(self) -> list[str]:
        return [p.name for p in self.parameters]

    def nominal_values(self) -> dict[str, float]:
        """Starting parameter values as a dict."""
        return {p.name: p.value for p in self.parameters}

    def to_minuit_init(self) -> dict:
        """Build a kwargs dict for ``iminuit.Minuit``."""
        init: dict = {"errordef": 1}  # likelihood fit
        for p in self.parameters:
            init.update(p.to_minuit_init())
        return init

    # ---- binning info ------------------------------------------------------

    @property
    def total_nbins(self) -> int:
        return sum(ch.nbins for ch in self._channels)

    @property
    def channel_slices(self) -> list[slice]:
        """Index slices into the concatenated bin vector for each channel."""
        slices = []
        offset = 0
        for ch in self._channels:
            slices.append(slice(offset, offset + ch.nbins))
            offset += ch.nbins
        return slices

    # ---- evaluation --------------------------------------------------------

    def expected(self, params: dict[str, float]) -> np.ndarray:
        """Concatenated expected yields across all channels.

        Parameters
        ----------
        params : dict[str, float]
            Parameter name -> value mapping.

        Returns
        -------
        numpy.ndarray
            Shape ``(total_nbins,)``.
        """
        parts = [ch.expected(params) for ch in self._channels]
        return np.concatenate(parts)

    @property
    def data(self) -> np.ndarray:
        """Concatenated observed data across all channels.

        Returns
        -------
        numpy.ndarray
            Shape ``(total_nbins,)``.
        """
        parts = [ch.data.values for ch in self._channels]
        return np.concatenate(parts)

    def __repr__(self) -> str:
        ch_names = [ch.name for ch in self._channels]
        return (
            f"Model({self.name!r}, channels={ch_names}, "
            f"parameters={self.parameter_names})"
        )
