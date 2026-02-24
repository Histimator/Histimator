"""Sample: a single histogram template with modifiers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from histimator.histograms import Histogram
from histimator.interpolation import InterpolationCode, interpolate
from histimator.parameters import Parameter
from histimator.templates import BinnedTemplate, Template


@dataclass(frozen=True)
class NormFactor:
    parameter: Parameter

@dataclass(frozen=True)
class NormSys:
    parameter: Parameter
    lo: float
    hi: float
    interp_code: InterpolationCode = InterpolationCode.PIECEWISE_EXPONENTIAL

@dataclass(frozen=True)
class HistoSys:
    parameter: Parameter
    lo_hist: Histogram
    hi_hist: Histogram
    interp_code: InterpolationCode = InterpolationCode.PIECEWISE_EXPONENTIAL

@dataclass(frozen=True)
class StatError:
    parameters: list[Parameter]
    rel_uncertainties: np.ndarray

@dataclass(frozen=True)
class ShapeSys:
    parameters: list[Parameter]
    rel_uncertainties: np.ndarray

@dataclass(frozen=True)
class ShapeFactor:
    parameters: list[Parameter]

@dataclass(frozen=True)
class LumiSys:
    parameter: Parameter
    uncertainty: float

Modifier = NormFactor | NormSys | HistoSys | StatError | ShapeSys | ShapeFactor | LumiSys


class Sample:
    def __init__(self, name: str, template_or_histogram: Histogram | Template) -> None:
        self.name = name
        if isinstance(template_or_histogram, Histogram):
            self._template: Template = BinnedTemplate(template_or_histogram)
        elif isinstance(template_or_histogram, Template):
            self._template = template_or_histogram
        else:
            raise TypeError(
                f"Expected Histogram or Template, got {type(template_or_histogram).__name__}"
            )
        self._modifiers: list[Modifier] = []

    @property
    def template(self) -> Template:
        return self._template

    @property
    def histogram(self) -> Histogram:
        """Access to the underlying Histogram.

        Available for BinnedTemplate, GPTemplate, BSplineTemplate,
        or any template that exposes a .histogram property.
        """
        if isinstance(self._template, BinnedTemplate):
            return self._template.histogram
        if hasattr(self._template, 'histogram'):
            return self._template.histogram
        raise AttributeError(
            f"Sample '{self.name}' uses a {type(self._template).__name__}, "
            "which does not expose a raw Histogram. Use sample.template instead."
        )

    @property
    def modifiers(self) -> tuple[Modifier, ...]:
        return tuple(self._modifiers)

    @property
    def parameters(self) -> list[Parameter]:
        params = []
        for m in self._modifiers:
            if isinstance(m, (NormFactor, NormSys, HistoSys, LumiSys)):
                params.append(m.parameter)
            elif isinstance(m, (StatError, ShapeSys, ShapeFactor)):
                params.extend(m.parameters)
        return params

    def add_normfactor(self, name, nominal=1.0, bounds=(0.0, 10.0)):
        p = Parameter(name, value=nominal, bounds=bounds)
        self._modifiers.append(NormFactor(parameter=p))
        return self

    def add_normsys(self, name, lo, hi,
                    interp_code=InterpolationCode.PIECEWISE_EXPONENTIAL):
        p = Parameter(name, value=0.0, bounds=(-5.0, 5.0))
        self._modifiers.append(NormSys(parameter=p, lo=lo, hi=hi, interp_code=interp_code))
        return self

    def add_histosys(self, name, lo_hist, hi_hist,
                     interp_code=InterpolationCode.PIECEWISE_EXPONENTIAL):
        if lo_hist.nbins != self._template.nbins:
            raise ValueError(f"lo_hist has {lo_hist.nbins} bins, expected {self._template.nbins}")
        if hi_hist.nbins != self._template.nbins:
            raise ValueError(f"hi_hist has {hi_hist.nbins} bins, expected {self._template.nbins}")
        p = Parameter(name, value=0.0, bounds=(-5.0, 5.0))
        self._modifiers.append(
            HistoSys(parameter=p, lo_hist=lo_hist, hi_hist=hi_hist, interp_code=interp_code)
        )
        return self

    def add_staterror(self, name, rel_uncertainties):
        rel = np.asarray(rel_uncertainties, dtype=np.float64)
        if rel.shape[0] != self._template.nbins:
            raise ValueError(
                f"rel_uncertainties has {rel.shape[0]} entries, expected {self._template.nbins}"
            )
        params = [
            Parameter(f"gamma_{name}_bin{i}", value=1.0, bounds=(0.0, 10.0))
            for i in range(self._template.nbins)
        ]
        self._modifiers.append(StatError(parameters=params, rel_uncertainties=rel))
        return self

    def add_shapesys(self, name, rel_uncertainties):
        rel = np.asarray(rel_uncertainties, dtype=np.float64)
        if rel.shape[0] != self._template.nbins:
            raise ValueError(
                f"rel_uncertainties has {rel.shape[0]} entries, expected {self._template.nbins}"
            )
        params = [
            Parameter(f"gamma_{name}_bin{i}", value=1.0, bounds=(0.0, 10.0))
            for i in range(self._template.nbins)
        ]
        self._modifiers.append(ShapeSys(parameters=params, rel_uncertainties=rel))
        return self

    def add_shapefactor(self, name):
        params = [
            Parameter(f"sf_{name}_bin{i}", value=1.0, bounds=(0.01, 100.0))
            for i in range(self._template.nbins)
        ]
        self._modifiers.append(ShapeFactor(parameters=params))
        return self

    def add_lumisys(self, name, uncertainty):
        p = Parameter(name, value=1.0, bounds=(0.5, 1.5))
        self._modifiers.append(LumiSys(parameter=p, uncertainty=uncertainty))
        return self

    def expected(self, params: dict[str, float]) -> np.ndarray:
        result = self._template.counts()
        for mod in self._modifiers:
            if isinstance(mod, NormFactor):
                alpha = params.get(mod.parameter.name, mod.parameter.value)
                result = result * alpha
            elif isinstance(mod, NormSys):
                alpha = params.get(mod.parameter.name, mod.parameter.value)
                scale = interpolate(alpha, 1.0, mod.lo, mod.hi, mod.interp_code)
                result = result * float(scale)
            elif isinstance(mod, HistoSys):
                alpha = params.get(mod.parameter.name, mod.parameter.value)
                result = interpolate(alpha, result, mod.lo_hist.values,
                                     mod.hi_hist.values, mod.interp_code)
            elif isinstance(mod, (StatError, ShapeSys, ShapeFactor)):
                gammas = np.array([params.get(p.name, p.value) for p in mod.parameters])
                result = result * gammas
            elif isinstance(mod, LumiSys):
                lumi = params.get(mod.parameter.name, mod.parameter.value)
                result = result * lumi
        return result

    def __repr__(self) -> str:
        return (f"Sample({self.name!r}, nbins={self._template.nbins},"
                f"modifiers={len(self._modifiers)})"
                )

    @classmethod
    def from_histogram(cls, name: str, histogram, template_type: str = "binned",
                       **template_kwargs) -> Sample:
        """Construct a Sample from a Histogram with a chosen template type.

        Parameters
        ----------
        name : str
            Sample name.
        histogram : Histogram
            Binned data.
        template_type : str
            "binned" (default) for BinnedTemplate (raw histogram).
            "gp" for GPTemplate (GP-smoothed with eigenmode uncertainty).
            "bspline" for BSplineTemplate (P-spline smoothed).
        **template_kwargs
            Extra keyword arguments forwarded to the template constructor.
        """
        from histimator.histograms import Histogram as _Histogram
        if not isinstance(histogram, _Histogram):
            raise TypeError(f"Expected Histogram, got {type(histogram).__name__}")

        if template_type == "binned":
            template = BinnedTemplate(histogram)
        elif template_type == "gp":
            from histimator.gp_template import GPTemplate
            template = GPTemplate(histogram, **template_kwargs)
        elif template_type == "bspline":
            from histimator.bspline_template import BSplineTemplate
            template = BSplineTemplate(histogram, **template_kwargs)
        else:
            raise ValueError(
                f"Unknown template_type {template_type!r}. "
                f"Choose from 'binned', 'gp', 'bspline'."
            )
        return cls(name, template)

    @classmethod
    def from_dataset(cls, name: str, dataset, edges=None, strategy=None,
                     template_type: str = "binned", **template_kwargs) -> Sample:
        """Construct a Sample by binning a Dataset.

        Parameters
        ----------
        name : str
            Sample name.
        dataset : Dataset
            Raw event-level data.
        edges : array-like or None
            Explicit bin edges.
        strategy : str or None
            Binning strategy (used when edges is None).
        template_type : str
            "binned" (default) for BinnedTemplate (raw histogram).
            "gp" for GPTemplate (GP-smoothed with eigenmode uncertainty).
            "bspline" for BSplineTemplate (P-spline smoothed).
        **template_kwargs
            Extra keyword arguments forwarded to the template constructor
            (e.g. kernel=GPKernel.MATERN_52 for GPTemplate).
        """
        from histimator.data import Dataset as _Dataset
        if not isinstance(dataset, _Dataset):
            raise TypeError(f"Expected Dataset, got {type(dataset).__name__}")

        if template_type == "gp":
            from histimator.gp_template import GPTemplate
            if edges is None and strategy is not None:
                from histimator.binning import auto_edges
                edges = auto_edges(dataset, method=strategy)
            elif edges is None:
                raise ValueError("Either edges or strategy must be provided")
            edges = np.asarray(edges, dtype=np.float64)
            template = GPTemplate.from_dataset(dataset, edges=edges, **template_kwargs)
            return cls(name, template)

        histogram = dataset.to_histogram(edges=edges, strategy=strategy)
        return cls.from_histogram(name, histogram, template_type=template_type,
                                  **template_kwargs)

    def add_histosys_from_datasets(
        self, name, lo_dataset, hi_dataset, edges=None, strategy=None,
        interp_code=InterpolationCode.PIECEWISE_EXPONENTIAL,
    ):
        """Add a HistoSys modifier by binning nominal/lo/hi datasets.

        All three datasets are binned using the same edges, guaranteeing
        consistency.  If edges is None, uses the sample's existing template
        edges.  If strategy is provided, auto-computes edges from the
        nominal template.

        Parameters
        ----------
        name : str
            Systematic name.
        lo_dataset : Dataset
            Down-variation raw events.
        hi_dataset : Dataset
            Up-variation raw events.
        edges : array-like or None
            Explicit bin edges.
        strategy : str or None
            Binning strategy for auto edge computation.
        interp_code : InterpolationCode
            Interpolation scheme.
        """
        from histimator.data import Dataset as _Dataset
        if not isinstance(lo_dataset, _Dataset) or not isinstance(hi_dataset, _Dataset):
            raise TypeError("lo_dataset and hi_dataset must be Dataset instances")

        if edges is None and strategy is None:
            edges = self._template.edges
        elif edges is None and strategy is not None:
            from histimator.binning import auto_edges
            from histimator.data import Dataset
            edges = auto_edges(
                Dataset(np.concatenate([lo_dataset.values, hi_dataset.values])),
                method=strategy,
            )

        lo_hist = lo_dataset.to_histogram(edges)
        hi_hist = hi_dataset.to_histogram(edges)
        return self.add_histosys(name, lo_hist, hi_hist, interp_code=interp_code)

    def add_staterror_from_dataset(self, name, dataset=None, edges=None):
        """Add a StatError modifier from a Dataset's binning uncertainties.

        Computes relative uncertainties from the dataset's sumw2:
            rel_unc_j = sqrt(sumw2_j) / max(count_j, epsilon)

        For unweighted data this is 1/sqrt(n_j), the standard Poisson
        relative error.  For weighted data it uses the Barlow-Beeston
        sumw2-based variance.

        Parameters
        ----------
        name : str
            Systematic name.
        dataset : Dataset or None
            Raw event data.  If None, uses the sample's own histogram.
        edges : array-like or None
            Bin edges.  If None, uses sample's template edges.
        """
        if dataset is not None:
            from histimator.data import Dataset as _Dataset
            if not isinstance(dataset, _Dataset):
                raise TypeError(f"Expected Dataset, got {type(dataset).__name__}")
            if edges is None:
                edges = self._template.edges
            h = dataset.to_histogram(edges)
        else:
            if not hasattr(self._template, 'histogram'):
                raise ValueError("No dataset provided and template has no histogram")
            h = self._template.histogram

        rel_unc = np.sqrt(h.sumw2) / np.maximum(h.values, 1e-10)
        return self.add_staterror(name, rel_unc)
