"""Histimator: histogram-template statistical inference."""
from histimator.bspline_template import BSplineTemplate
from histimator.channels import Channel
from histimator.cls import cls_expected_band, cls_observed, upper_limit_cls
from histimator.data import Dataset
from histimator.diagnostics import (
    impacts,
    nuisance_parameter_pulls,
    nuisance_pulls,
    postfit_yields,
    prefit_postfit_yields,
    prefit_yields,
)
from histimator.eigenmode_constraint import (
    EigenmodeConstraint,
    SystematicDirection,
    make_systematic_direction,
)
from histimator.gp_template import GPKernel, GPTemplate
from histimator.histograms import Histogram
from histimator.interpolation import InterpolationCode, interpolate
from histimator.likelihood import BinnedNLL, FitResult, fit
from histimator.model import Model
from histimator.parameters import Parameter
from histimator.samples import (
    HistoSys,
    LumiSys,
    NormFactor,
    NormSys,
    Sample,
    ShapeFactor,
    ShapeSys,
    StatError,
)
from histimator.templates import BinnedTemplate, Template

__all__ = [
    "Parameter",
    "Histogram",
    "Template",
    "BinnedTemplate",
    "BSplineTemplate",
    "GPTemplate",
    "GPKernel",
    "EigenmodeConstraint",
    "SystematicDirection",
    "make_systematic_direction",
    "InterpolationCode",
    "interpolate",
    "Sample",
    "NormFactor",
    "NormSys",
    "HistoSys",
    "StatError",
    "ShapeSys",
    "ShapeFactor",
    "LumiSys",
    "Channel",
    "Model",
    "Dataset",
    "BinnedNLL",
    "FitResult",
    "fit",
    "prefit_yields",
    "postfit_yields",
    "nuisance_pulls",
    "prefit_postfit_yields",
    "nuisance_parameter_pulls",
    "impacts",
    "cls_observed",
    "cls_expected_band",
    "upper_limit_cls",
]

__version__ = "0.3.1"
