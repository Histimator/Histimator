"""Likelihood functions and fitting machinery.

The central object is ``BinnedNLL``, a callable that computes the
negative log-likelihood for a Histimator ``Model``.  The likelihood
is the product of Poisson probabilities per bin (optionally extended)
times Gaussian constraint terms for nuisance parameters.

The ``fit`` function wraps iminuit to perform the actual minimisation,
returning a ``FitResult`` with best-fit values, uncertainties, and
correlation information.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.special as sp
from iminuit import Minuit

from histimator.model import Model
from histimator.samples import HistoSys, NormSys

# --------------------------------------------------------------------------
# Poisson log-likelihood (vectorised, continuous approximation)
# --------------------------------------------------------------------------

def _poisson_logpdf(k: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Continuous Poisson log-probability.

    Uses the Stirling approximation via ``gammaln`` so it works for
    non-integer k (which arise with weighted data).
    """
    # Protect against mu <= 0
    safe_mu = np.maximum(mu, 1e-10)
    return sp.xlogy(k, safe_mu) - sp.gammaln(k + 1) - safe_mu


# --------------------------------------------------------------------------
# BinnedNLL cost function
# --------------------------------------------------------------------------

class BinnedNLL:
    """Binned negative log-likelihood for a Histimator Model.

    Parameters
    ----------
    model : Model
        A fully constructed model with data.
    extended : bool
        If ``True`` (default), include the extended term that constrains
        the total expected yield.

    Examples
    --------
    >>> nll = BinnedNLL(model)
    >>> nll(mu=1.0, bkg_syst=0.0)  # evaluate at specific parameter values
    42.7
    """

    def __init__(self, model: Model, extended: bool = True) -> None:
        self._model = model
        self._extended = extended

        # Observed data (cached)
        self._data = model.data
        self._data_total = float(self._data.sum())

        # Parameter names in a fixed order for iminuit
        self._par_names = model.parameter_names

        # Identify which parameters are nuisance parameters (have Gaussian
        # constraints). These are parameters attached to NormSys or HistoSys.
        self._constrained: set[str] = set()
        for ch in model.channels:
            for sample in ch.samples:
                for mod in sample.modifiers:
                    if isinstance(mod, (NormSys, HistoSys)):
                        self._constrained.add(mod.parameter.name)

        # iminuit interface: expose parameter names.
        # Modern iminuit (>=2.x) uses _parameters dict.
        self._parameters = {name: None for name in self._par_names}

    @property
    def model(self) -> Model:
        return self._model

    def __call__(self, par_array) -> float:
        """Evaluate the negative log-likelihood.

        Parameters are passed as a single numpy array by iminuit v2.
        """
        params = dict(zip(self._par_names, par_array, strict=True))

        # Expected yields
        expected = self._model.expected(params)

        # Poisson log-likelihood per bin
        ll = _poisson_logpdf(self._data, expected).sum()

        # Extended term: Poisson probability for the total count
        if self._extended:
            ll += _poisson_logpdf(
                np.array([self._data_total]),
                np.array([expected.sum()])
            ).item()

        # Gaussian constraint terms for nuisance parameters
        # NP convention: alpha ~ N(0, 1)
        constraint = 0.0
        for name in self._constrained:
            alpha = params.get(name, 0.0)
            constraint += -0.5 * alpha * alpha

        return -(ll + constraint)


# --------------------------------------------------------------------------
# Fit result
# --------------------------------------------------------------------------

@dataclass
class FitResult:
    """Container for minimisation results.

    Attributes
    ----------
    bestfit : dict[str, float]
        Best-fit parameter values.
    errors : dict[str, float]
        Parabolic (Hesse) errors.
    covariance : Optional[dict[tuple[str,str], float]]
        Covariance matrix entries.
    minos_errors : Optional[dict[str, tuple[float, float]]]
        Asymmetric Minos errors ``{name: (lower, upper)}``.
    minuit : Minuit
        The underlying iminuit object for advanced access.
    valid : bool
        Whether the fit converged.
    nll_min : float
        Minimum negative log-likelihood value.
    """

    bestfit: dict[str, float]
    errors: dict[str, float]
    covariance: dict[tuple[str, str], float] | None
    minos_errors: dict[str, tuple[float, float]] | None
    minuit: Minuit
    valid: bool
    nll_min: float


# --------------------------------------------------------------------------
# Convenience fitter
# --------------------------------------------------------------------------

def fit(
    model: Model,
    extended: bool = True,
    run_minos: bool = False,
    **minuit_kwargs,
) -> FitResult:
    """Fit a model to its observed data.

    Parameters
    ----------
    model : Model
        Fully constructed model with data attached.
    extended : bool
        Use extended likelihood (default True).
    run_minos : bool
        If True, also run MINOS after MIGRAD + HESSE.
    **minuit_kwargs
        Overrides for iminuit (e.g. ``mu=0.5`` to change start value).

    Returns
    -------
    FitResult
    """
    nll = BinnedNLL(model, extended=extended)

    # Build initial parameter values
    start_values = [p.value for p in model.parameters]

    # Apply user overrides for start values
    for i, p in enumerate(model.parameters):
        if p.name in minuit_kwargs:
            start_values[i] = minuit_kwargs[p.name]

    m = Minuit(nll, start_values, name=nll._par_names)
    m.errordef = 1  # likelihood fit

    # Apply limits and fixed flags from model
    for p in model.parameters:
        if p.bounds is not None:
            m.limits[p.name] = p.bounds
        if p.fixed:
            m.fixed[p.name] = True

    m.migrad()
    m.hesse()

    minos_errors = None
    if run_minos:
        m.minos()
        minos_errors = {}
        for pname in nll._par_names:
            if pname in m.merrors:
                me = m.merrors[pname]
                minos_errors[pname] = (me.lower, me.upper)

    bestfit = {name: float(m.values[name]) for name in nll._par_names}
    errors = {name: float(m.errors[name]) for name in nll._par_names}

    cov = None
    if m.covariance is not None:
        cov = {}
        names = nll._par_names
        for i, ni in enumerate(names):
            for j, nj in enumerate(names):
                cov[(ni, nj)] = m.covariance[i][j]

    return FitResult(
        bestfit=bestfit,
        errors=errors,
        covariance=cov,
        minos_errors=minos_errors,
        minuit=m,
        valid=m.valid,
        nll_min=float(m.fval),
    )
