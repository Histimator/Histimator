"""Sample: a single histogram template with modifiers.

A Sample represents one physics process (signal, background, etc.) in a
single analysis channel. It carries the nominal histogram template and
an ordered list of *modifiers* that transform the template as a function
of model parameters.

Modifier types
--------------
- **normfactor**: multiplicative scale factor (parameter of interest or
  free normalisation).
- **normsys**: overall normalisation uncertainty driven by a nuisance
  parameter, using an interpolation code.
- **histosys**: per-bin shape uncertainty driven by a nuisance parameter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from histimator.histograms import Histogram
from histimator.interpolation import InterpolationCode, interpolate
from histimator.parameters import Parameter

# ---- Modifier descriptors -------------------------------------------------

@dataclass(frozen=True)
class NormFactor:
    """Free multiplicative normalisation parameter."""
    parameter: Parameter


@dataclass(frozen=True)
class NormSys:
    """Overall normalisation systematic.

    *lo* and *hi* are the scale factors at alpha = -1 and +1 respectively
    (e.g. 0.9 and 1.1 for a +/-10% uncertainty).
    """
    parameter: Parameter
    lo: float
    hi: float
    interp_code: InterpolationCode = InterpolationCode.PIECEWISE_EXPONENTIAL


@dataclass(frozen=True)
class HistoSys:
    """Per-bin shape systematic.

    *lo_hist* and *hi_hist* are the full histogram templates at the
    -1 sigma and +1 sigma variations.
    """
    parameter: Parameter
    lo_hist: Histogram
    hi_hist: Histogram
    interp_code: InterpolationCode = InterpolationCode.PIECEWISE_EXPONENTIAL


Modifier = NormFactor | NormSys | HistoSys


# ---- Sample ---------------------------------------------------------------

class Sample:
    """A single process contribution to a channel.

    Parameters
    ----------
    name : str
        Human-readable label (e.g. "signal", "ttbar").
    histogram : Histogram
        Nominal template.
    """

    def __init__(self, name: str, histogram: Histogram) -> None:
        self.name = name
        self.histogram = histogram
        self._modifiers: list[Modifier] = []

    @property
    def modifiers(self) -> tuple[Modifier, ...]:
        return tuple(self._modifiers)

    @property
    def parameters(self) -> list[Parameter]:
        """All parameters introduced by this sample's modifiers."""
        return [m.parameter for m in self._modifiers]

    # ---- modifier attachment ------------------------------------------------

    def add_normfactor(
        self,
        name: str,
        nominal: float = 1.0,
        bounds: tuple[float, float] | None = (0.0, 10.0),
    ) -> Sample:
        """Attach a free normalisation factor."""
        p = Parameter(name, value=nominal, bounds=bounds)
        self._modifiers.append(NormFactor(parameter=p))
        return self

    def add_normsys(
        self,
        name: str,
        lo: float,
        hi: float,
        interp_code: InterpolationCode = InterpolationCode.PIECEWISE_EXPONENTIAL,
    ) -> Sample:
        """Attach an overall normalisation systematic."""
        p = Parameter(name, value=0.0, bounds=(-5.0, 5.0))
        self._modifiers.append(NormSys(parameter=p, lo=lo, hi=hi, interp_code=interp_code))
        return self

    def add_histosys(
        self,
        name: str,
        lo_hist: Histogram,
        hi_hist: Histogram,
        interp_code: InterpolationCode = InterpolationCode.PIECEWISE_EXPONENTIAL,
    ) -> Sample:
        """Attach a per-bin shape systematic."""
        if lo_hist.nbins != self.histogram.nbins:
            raise ValueError(
                f"lo_hist has {lo_hist.nbins} bins, expected {self.histogram.nbins}"
            )
        if hi_hist.nbins != self.histogram.nbins:
            raise ValueError(
                f"hi_hist has {hi_hist.nbins} bins, expected {self.histogram.nbins}"
            )
        p = Parameter(name, value=0.0, bounds=(-5.0, 5.0))
        self._modifiers.append(
            HistoSys(parameter=p, lo_hist=lo_hist, hi_hist=hi_hist, interp_code=interp_code)
        )
        return self

    # ---- evaluation ---------------------------------------------------------

    def expected(self, params: dict[str, float]) -> np.ndarray:
        """Compute expected bin yields given parameter values.

        Parameters
        ----------
        params : dict[str, float]
            Mapping of parameter name to value.

        Returns
        -------
        numpy.ndarray
            Expected bin counts, shape ``(nbins,)``.
        """
        result = self.histogram.values.copy()

        for mod in self._modifiers:
            alpha = params.get(mod.parameter.name, mod.parameter.value)

            if isinstance(mod, NormFactor):
                result = result * alpha

            elif isinstance(mod, NormSys):
                scale = interpolate(
                    alpha, 1.0, mod.lo, mod.hi, mod.interp_code
                )
                result = result * float(scale)

            elif isinstance(mod, HistoSys):
                result = interpolate(
                    alpha,
                    result,
                    mod.lo_hist.values,
                    mod.hi_hist.values,
                    mod.interp_code,
                )

        return result

    def __repr__(self) -> str:
        n = self.histogram.nbins
        m = len(self._modifiers)
        return f"Sample({self.name!r}, nbins={n}, modifiers={m})"
