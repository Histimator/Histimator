"""Histimator: histogram-template statistical inference.

Build statistical models from histogram templates and perform
maximum-likelihood estimation with systematic uncertainties.
"""

from __future__ import annotations

__version__ = "0.3.0"
__all__ = [
    "Parameter",
    "Histogram",
    "Sample",
    "Channel",
    "Model",
    "BinnedNLL",
    "InterpolationCode",
    "interpolate",
]


def __getattr__(name: str):
    """Lazy imports so the package loads fast."""
    if name == "Parameter":
        from histimator.parameters import Parameter
        return Parameter
    if name == "Histogram":
        from histimator.histograms import Histogram
        return Histogram
    if name == "Sample":
        from histimator.samples import Sample
        return Sample
    if name == "Channel":
        from histimator.channels import Channel
        return Channel
    if name == "Model":
        from histimator.model import Model
        return Model
    if name == "BinnedNLL":
        from histimator.likelihood import BinnedNLL
        return BinnedNLL
    if name in ("InterpolationCode", "interpolate"):
        from histimator import interpolation as _interp
        if name == "InterpolationCode":
            return _interp.InterpolationCode
        return _interp.interpolate
    raise AttributeError(f"module 'histimator' has no attribute {name!r}")
