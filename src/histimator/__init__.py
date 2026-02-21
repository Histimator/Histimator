"""Histimator: histogram-template statistical inference."""
from histimator.parameters import Parameter
from histimator.histograms import Histogram
from histimator.templates import Template, BinnedTemplate
from histimator.bspline_template import BSplineTemplate
from histimator.gp_template import GPTemplate, GPKernel
from histimator.eigenmode_constraint import (
    EigenmodeConstraint, SystematicDirection, make_systematic_direction,
)
from histimator.interpolation import InterpolationCode, interpolate
from histimator.samples import (Sample, NormFactor, NormSys, HistoSys,
                                 StatError, ShapeSys, ShapeFactor, LumiSys)
from histimator.channels import Channel
from histimator.model import Model
from histimator.data import Dataset
from histimator.likelihood import BinnedNLL, FitResult, fit
__version__ = "0.3.0a1"
