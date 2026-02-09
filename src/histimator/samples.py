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


@dataclass(frozen=True)
class StatError:
    """Per-bin MC statistical uncertainty (Barlow-Beeston).

    Introduces one gamma parameter per bin with Gaussian constraints.
    gamma_i multiplies the sample yield in bin i, constrained at
    N(1, rel_uncertainty_i).
    """
    parameters: list[Parameter]
    rel_uncertainties: np.ndarray


@dataclass(frozen=True)
class ShapeSys:
    """Per-bin uncorrelated shape systematic with Poisson constraints.

    Like StatError but uses Poisson auxiliary data as the constraint,
    which is more appropriate when the uncertainty comes from a
    limited-statistics auxiliary measurement.
    """
    parameters: list[Parameter]
    rel_uncertainties: np.ndarray


@dataclass(frozen=True)
class ShapeFactor:
    """Free-floating per-bin normalisation (no constraint).

    Used for data-driven background estimation where the shape is
    determined entirely by a control region fit.
    """
    parameters: list[Parameter]


@dataclass(frozen=True)
class LumiSys:
    """Luminosity systematic: single Gaussian-constrained scale factor.

    Multiplies the entire sample yield.  Constrained at N(1, uncertainty).
    """
    parameter: Parameter
    uncertainty: float


Modifier = NormFactor | NormSys | HistoSys | StatError | ShapeSys | ShapeFactor | LumiSys


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
        params = []
        for m in self._modifiers:
            if isinstance(m, (NormFactor, NormSys, HistoSys, LumiSys)):
                params.append(m.parameter)
            elif isinstance(m, (StatError, ShapeSys, ShapeFactor)):
                params.extend(m.parameters)
        return params

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

    def add_staterror(
        self,
        name: str,
        rel_uncertainties: np.ndarray | list,
    ) -> Sample:
        """Attach per-bin MC statistical uncertainties (Barlow-Beeston).

        Each bin gets a gamma parameter constrained at N(1, delta_i).
        """
        rel = np.asarray(rel_uncertainties, dtype=np.float64)
        if rel.shape[0] != self.histogram.nbins:
            raise ValueError(
                f"rel_uncertainties has {rel.shape[0]} entries, "
                f"expected {self.histogram.nbins}"
            )
        params = [
            Parameter(
                f"gamma_{name}_bin{i}",
                value=1.0,
                bounds=(0.0, 10.0),
            )
            for i in range(self.histogram.nbins)
        ]
        self._modifiers.append(StatError(parameters=params, rel_uncertainties=rel))
        return self

    def add_shapesys(
        self,
        name: str,
        rel_uncertainties: np.ndarray | list,
    ) -> Sample:
        """Attach per-bin shape systematic with Poisson constraints."""
        rel = np.asarray(rel_uncertainties, dtype=np.float64)
        if rel.shape[0] != self.histogram.nbins:
            raise ValueError(
                f"rel_uncertainties has {rel.shape[0]} entries, "
                f"expected {self.histogram.nbins}"
            )
        params = [
            Parameter(
                f"gamma_{name}_bin{i}",
                value=1.0,
                bounds=(0.0, 10.0),
            )
            for i in range(self.histogram.nbins)
        ]
        self._modifiers.append(ShapeSys(parameters=params, rel_uncertainties=rel))
        return self

    def add_shapefactor(self, name: str) -> Sample:
        """Attach free-floating per-bin normalisation (no constraint)."""
        params = [
            Parameter(
                f"sf_{name}_bin{i}",
                value=1.0,
                bounds=(0.01, 100.0),
            )
            for i in range(self.histogram.nbins)
        ]
        self._modifiers.append(ShapeFactor(parameters=params))
        return self

    def add_lumisys(self, name: str, uncertainty: float) -> Sample:
        """Attach a luminosity systematic (Gaussian-constrained overall scale)."""
        p = Parameter(name, value=1.0, bounds=(0.5, 1.5))
        self._modifiers.append(LumiSys(parameter=p, uncertainty=uncertainty))
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
            if isinstance(mod, NormFactor):
                alpha = params.get(mod.parameter.name, mod.parameter.value)
                result = result * alpha

            elif isinstance(mod, NormSys):
                alpha = params.get(mod.parameter.name, mod.parameter.value)
                scale = interpolate(
                    alpha, 1.0, mod.lo, mod.hi, mod.interp_code
                )
                result = result * float(scale)

            elif isinstance(mod, HistoSys):
                alpha = params.get(mod.parameter.name, mod.parameter.value)
                result = interpolate(
                    alpha,
                    result,
                    mod.lo_hist.values,
                    mod.hi_hist.values,
                    mod.interp_code,
                )

            elif isinstance(mod, (StatError, ShapeSys, ShapeFactor)):
                gammas = np.array([
                    params.get(p.name, p.value) for p in mod.parameters
                ])
                result = result * gammas

            elif isinstance(mod, LumiSys):
                lumi = params.get(mod.parameter.name, mod.parameter.value)
                result = result * lumi

        return result

    def __repr__(self) -> str:
        n = self.histogram.nbins
        m = len(self._modifiers)
        return f"Sample({self.name!r}, nbins={n}, modifiers={m})"
