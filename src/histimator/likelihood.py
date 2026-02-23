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
from histimator.samples import (
    HistoSys,
    LumiSys,
    NormSys,
    ShapeFactor,
    ShapeSys,
    StatError,
)

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

        # Build constraint registries.
        #
        # Standard alpha constraints: NormSys/HistoSys alphas ~ N(0, 1)
        self._constrained: set[str] = set()
        # Gaussian constraints at non-zero mean: {name: (mean, sigma)}
        self._gaussian_constraints: dict[str, tuple[float, float]] = {}
        # Poisson auxiliary constraints: {name: aux_data}
        self._poisson_constraints: dict[str, float] = {}

        for ch in model.channels:
            for sample in ch.samples:
                for mod in sample.modifiers:
                    if isinstance(mod, (NormSys, HistoSys)):
                        self._constrained.add(mod.parameter.name)
                    elif isinstance(mod, StatError):
                        for p, delta in zip(
                                mod.parameters, mod.rel_uncertainties,
                                strict=True
                        ):
                            self._gaussian_constraints[p.name] = (1.0, float(delta))
                    elif isinstance(mod, LumiSys):
                        self._gaussian_constraints[mod.parameter.name] = (
                            1.0,
                            mod.uncertainty,
                        )
                    elif isinstance(mod, ShapeSys):
                        for p, rel in zip(
                                mod.parameters, mod.rel_uncertainties,
                                strict=True
                        ):
                            # Poisson auxiliary data: tau = 1/rel^2
                            tau = 1.0 / (float(rel) ** 2)
                            self._poisson_constraints[p.name] = tau

        # iminuit interface: expose parameter names.
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

        # Gaussian constraints at non-zero mean: gamma ~ N(mean, sigma)
        for name, (mean, sigma) in self._gaussian_constraints.items():
            val = params.get(name, mean)
            constraint += -0.5 * ((val - mean) / sigma) ** 2

        # Poisson auxiliary constraints for ShapeSys
        for name, tau in self._poisson_constraints.items():
            gamma = params.get(name, 1.0)
            # P(tau | tau * gamma) = Poisson(aux_data=tau, mean=tau*gamma)
            safe_mean = max(tau * gamma, 1e-10)
            constraint += tau * np.log(safe_mean) - safe_mean - sp.gammaln(tau + 1)

        return -(ll + constraint)


# --------------------------------------------------------------------------
# Unbinned extended NLL
# --------------------------------------------------------------------------

_PER_BIN_MODIFIERS = (HistoSys, StatError, ShapeSys, ShapeFactor)


class UnbinnedNLL:
    """Unbinned extended negative log-likelihood for a Histimator Model.

    The likelihood for each channel c with observed events {x_i} is:

        log L_c = sum_i log( sum_j N_j(theta) * p_j(x_i) )
                  - sum_j N_j(theta)

    where p_j(x) is the probability density from template.evaluate_density
    and N_j(theta) is the total expected yield for sample j after scalar
    modifiers.  Since scalar modifiers do not change the shape, p_j(x_i)
    is cached at construction time.

    Only scalar modifiers (NormFactor, NormSys, LumiSys) are supported.
    Per-bin modifiers (HistoSys, StatError, ShapeSys, ShapeFactor) raise
    a clear error because the unbinned density does not capture per-bin
    shape changes.

    Parameters
    ----------
    model : Model
        A fully constructed model.  Each channel must have unbinned data
        set via ``channel.set_unbinned_data(events)``.
    """

    def __init__(self, model: Model) -> None:
        self._model = model
        self._par_names = model.parameter_names

        # Validate: no per-bin modifiers
        for ch in model.channels:
            for sample in ch.samples:
                for mod in sample.modifiers:
                    if isinstance(mod, _PER_BIN_MODIFIERS):
                        raise ValueError(
                            f"Sample '{sample.name}' in channel '{ch.name}' "
                            f"has per-bin modifier {type(mod).__name__}, "
                            f"which is incompatible with unbinned likelihood. "
                            f"Only NormFactor, NormSys, and LumiSys are supported."
                        )

        # Validate: all channels have unbinned data
        for ch in model.channels:
            if ch.unbinned_data is None:
                raise ValueError(
                    f"Channel '{ch.name}' has no unbinned data. "
                    f"Call channel.set_unbinned_data(events) first."
                )

        # Cache per-sample density evaluations at each channel's events.
        # _density_cache[c][j] is a 1-D array of p_j(x_i) for channel c.
        # _n_events[c] is the number of events in channel c.
        self._density_cache: list[list[np.ndarray]] = []
        self._n_events: list[int] = []
        for ch in model.channels:
            events = ch.unbinned_data
            self._n_events.append(len(events))
            ch_densities = []
            for sample in ch.samples:
                p = sample.template.evaluate_density(events)
                ch_densities.append(p)
            self._density_cache.append(ch_densities)

        # Build constraint registries (same logic as BinnedNLL)
        self._constrained: set[str] = set()
        self._gaussian_constraints: dict[str, tuple[float, float]] = {}
        for ch in model.channels:
            for sample in ch.samples:
                for mod in sample.modifiers:
                    if isinstance(mod, (NormSys, HistoSys)):
                        self._constrained.add(mod.parameter.name)
                    elif isinstance(mod, LumiSys):
                        self._gaussian_constraints[mod.parameter.name] = (
                            1.0,
                            mod.uncertainty,
                        )

    @property
    def model(self) -> Model:
        return self._model

    def __call__(self, par_array) -> float:
        """Evaluate the negative log-likelihood."""
        params = dict(zip(self._par_names, par_array, strict=True))

        ll = 0.0
        for c, ch in enumerate(self._model.channels):
            # Compute total expected yield per sample
            n_totals = []
            for sample in ch.samples:
                n_j = float(sample.expected(params).sum())
                n_totals.append(n_j)

            # Event-level log-likelihood
            if self._n_events[c] > 0:
                # rate(x_i) = sum_j N_j * p_j(x_i)
                rate = np.zeros(self._n_events[c], dtype=np.float64)
                for j, n_j in enumerate(n_totals):
                    rate += n_j * self._density_cache[c][j]
                # Protect against log(0)
                safe_rate = np.maximum(rate, 1e-300)
                ll += np.sum(np.log(safe_rate))

            # Extended term: -N_total
            n_total = sum(n_totals)
            ll -= n_total

        # Constraint terms
        constraint = 0.0
        for name in self._constrained:
            alpha = params.get(name, 0.0)
            constraint += -0.5 * alpha * alpha
        for name, (mean, sigma) in self._gaussian_constraints.items():
            val = params.get(name, mean)
            constraint += -0.5 * ((val - mean) / sigma) ** 2

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
    unbinned: bool = False,
    run_minos: bool = False,
    **minuit_kwargs,
) -> FitResult:
    """Fit a model to its observed data.

    Parameters
    ----------
    model : Model
        Fully constructed model with data attached.
    extended : bool
        Use extended likelihood (default True).  Ignored when
        ``unbinned=True`` (the unbinned likelihood is always extended).
    unbinned : bool
        If True, use the unbinned extended likelihood.  Each channel
        must have unbinned data set via ``channel.set_unbinned_data``.
        Only scalar modifiers (NormFactor, NormSys, LumiSys) are
        supported in unbinned mode.
    run_minos : bool
        If True, also run MINOS after MIGRAD + HESSE.
    **minuit_kwargs
        Overrides for iminuit (e.g. ``mu=0.5`` to change start value).

    Returns
    -------
    FitResult
    """
    nll = UnbinnedNLL(model) if unbinned else BinnedNLL(model, extended=extended)

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
