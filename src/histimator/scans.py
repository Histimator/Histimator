"""Likelihood scans and goodness-of-fit tests.

Provides tools for visualising and interpreting the likelihood surface:

- ``likelihood_scan_1d``: profile -2*delta(ln L) vs one parameter
- ``likelihood_scan_2d``: profile -2*delta(ln L) on a 2D grid
- ``scan_intervals``: extract confidence intervals from a 1D scan
- ``goodness_of_fit``: saturated-model GoF test

The 1D scan is the standard way to extract confidence intervals that
account for non-parabolic likelihoods.  The 2D scan produces the
contour data for two-parameter confidence regions.  The GoF test
checks whether the best-fit model adequately describes the data.

References
----------
.. [Combine] CMS Collaboration, "The CMS statistical analysis and
   combination tool: COMBINE", CMS-CAT-23-001.
"""

from __future__ import annotations

import numpy as np
from iminuit import Minuit
from scipy import stats

from histimator.likelihood import BinnedNLL, _poisson_logpdf, fit
from histimator.model import Model

# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _make_nll(model: Model, extended: bool = True, unbinned: bool = False):
    """Construct the appropriate NLL object."""
    if unbinned:
        from histimator.likelihood import UnbinnedNLL
        return UnbinnedNLL(model)
    return BinnedNLL(model, extended=extended)


def _profile_nll_at(
    model: Model,
    fixed_params: dict[str, float],
    extended: bool = True,
    unbinned: bool = False,
) -> float:
    """Minimise the NLL with one or more parameters fixed.

    All parameters not in fixed_params are profiled (floated).
    Returns the minimum NLL value.
    """
    nll = _make_nll(model, extended=extended, unbinned=unbinned)
    par_names = nll._par_names
    start = [p.value for p in model.parameters]

    m = Minuit(nll, start, name=par_names)
    m.errordef = Minuit.LIKELIHOOD

    for p in model.parameters:
        if p.bounds is not None:
            m.limits[p.name] = p.bounds
        if p.fixed:
            m.fixed[p.name] = True

    for name, value in fixed_params.items():
        m.values[name] = value
        m.fixed[name] = True

    m.migrad()
    return float(m.fval)


# ------------------------------------------------------------------
# 1D likelihood scan
# ------------------------------------------------------------------

def likelihood_scan_1d(
    model: Model,
    par_name: str,
    bounds: tuple[float, float] | None = None,
    n_points: int = 21,
    extended: bool = True,
    unbinned: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Profile -2*delta(ln L) as a function of one parameter.

    All other parameters are profiled (re-minimised) at each scan
    point.  The result is referenced to the global minimum, so the
    minimum of the scan is approximately zero.

    Parameters
    ----------
    model : Model
        Fully constructed model with data.
    par_name : str
        Name of the parameter to scan.
    bounds : tuple[float, float] or None
        Scan range.  If None, uses 3*sigma around the best fit.
    n_points : int
        Number of scan points.
    extended : bool
        Use extended likelihood.
    unbinned : bool
        If True, use the unbinned extended likelihood.

    Returns
    -------
    par_values : numpy.ndarray
        The scanned parameter values.
    delta_nll : numpy.ndarray
        -2 * (ln L(par) - ln L_max), non-negative.
    """
    # Global minimum
    result = fit(model, extended=extended, unbinned=unbinned)
    nll_min = result.nll_min

    if bounds is None:
        par_hat = result.bestfit[par_name]
        sigma = result.errors[par_name]
        bounds = (par_hat - 3 * sigma, par_hat + 3 * sigma)

    par_values = np.linspace(bounds[0], bounds[1], n_points)
    delta_nll = np.zeros(n_points)

    for i, val in enumerate(par_values):
        nll_cond = _profile_nll_at(
            model, {par_name: val}, extended=extended, unbinned=unbinned,
        )
        delta_nll[i] = max(0.0, 2.0 * (nll_cond - nll_min))

    return par_values, delta_nll


# ------------------------------------------------------------------
# Interval extraction from 1D scan
# ------------------------------------------------------------------

def scan_intervals(
    par_values: np.ndarray,
    delta_nll: np.ndarray,
    level: float = 1.0,
) -> tuple[float, float]:
    """Extract a confidence interval from a 1D likelihood scan.

    Finds the lower and upper parameter values where delta_nll
    crosses the specified level.  The level corresponds to:
    - 1.0  for 68% CL (1 sigma)
    - 3.84 for 95% CL
    - 9.0  for 3 sigma

    Uses linear interpolation between scan points.

    Parameters
    ----------
    par_values : numpy.ndarray
        Scanned parameter values.
    delta_nll : numpy.ndarray
        Corresponding -2*delta(ln L) values.
    level : float
        The delta_nll threshold for the interval.

    Returns
    -------
    lo : float
        Lower bound of the interval.
    hi : float
        Upper bound of the interval.
    """
    # Find the minimum
    i_min = np.argmin(delta_nll)

    # Search left from minimum for the lower crossing
    lo = par_values[0]
    for i in range(i_min, 0, -1):
        if delta_nll[i - 1] >= level >= delta_nll[i]:
            # Linear interpolation
            frac = (level - delta_nll[i]) / (delta_nll[i - 1] - delta_nll[i])
            lo = par_values[i] - frac * (par_values[i] - par_values[i - 1])
            break

    # Search right from minimum for the upper crossing
    hi = par_values[-1]
    for i in range(i_min, len(par_values) - 1):
        if delta_nll[i] <= level <= delta_nll[i + 1]:
            frac = (level - delta_nll[i]) / (delta_nll[i + 1] - delta_nll[i])
            hi = par_values[i] + frac * (par_values[i + 1] - par_values[i])
            break

    return lo, hi


# ------------------------------------------------------------------
# 2D likelihood scan
# ------------------------------------------------------------------

def likelihood_scan_2d(
    model: Model,
    par_x: str,
    par_y: str,
    bounds_x: tuple[float, float] | None = None,
    bounds_y: tuple[float, float] | None = None,
    n_points_x: int = 11,
    n_points_y: int = 11,
    extended: bool = True,
    unbinned: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Profile -2*delta(ln L) on a 2D grid of two parameters.

    All parameters except par_x and par_y are profiled at each
    grid point.

    Parameters
    ----------
    model : Model
        Fully constructed model with data.
    par_x, par_y : str
        Names of the two parameters to scan.
    bounds_x, bounds_y : tuple[float, float] or None
        Scan ranges.  Auto-determined if None.
    n_points_x, n_points_y : int
        Number of grid points per axis.
    extended : bool
        Use extended likelihood.
    unbinned : bool
        If True, use the unbinned extended likelihood.

    Returns
    -------
    x_values : numpy.ndarray
        Shape (n_points_x,).
    y_values : numpy.ndarray
        Shape (n_points_y,).
    delta_nll : numpy.ndarray
        Shape (n_points_x, n_points_y). delta_nll[i, j] is the
        profiled -2*delta(ln L) at (x_values[i], y_values[j]).
    """
    result = fit(model, extended=extended, unbinned=unbinned)
    nll_min = result.nll_min

    if bounds_x is None:
        xhat = result.bestfit[par_x]
        sx = result.errors[par_x]
        bounds_x = (xhat - 3 * sx, xhat + 3 * sx)

    if bounds_y is None:
        yhat = result.bestfit[par_y]
        sy = result.errors[par_y]
        bounds_y = (yhat - 3 * sy, yhat + 3 * sy)

    x_values = np.linspace(bounds_x[0], bounds_x[1], n_points_x)
    y_values = np.linspace(bounds_y[0], bounds_y[1], n_points_y)
    dnll_grid = np.zeros((n_points_x, n_points_y))

    for i, xv in enumerate(x_values):
        for j, yv in enumerate(y_values):
            nll_cond = _profile_nll_at(
                model, {par_x: xv, par_y: yv},
                extended=extended, unbinned=unbinned,
            )
            dnll_grid[i, j] = max(0.0, 2.0 * (nll_cond - nll_min))

    return x_values, y_values, dnll_grid


# ------------------------------------------------------------------
# Goodness-of-fit (saturated model)
# ------------------------------------------------------------------

def goodness_of_fit(
    model: Model,
    extended: bool = True,
    unbinned: bool = False,
) -> tuple[float, int, float]:
    """Saturated-model goodness-of-fit test.

    Compares the best-fit NLL to the saturated model NLL (where the
    expected yield in every bin equals the observed data).  The test
    statistic is:

        chi2 = 2 * (NLL_bestfit - NLL_saturated)

    which is asymptotically chi2-distributed with ndof = nbins - nfloated.

    Parameters
    ----------
    model : Model
        Fully constructed model with data.
    extended : bool
        Use extended likelihood.
    unbinned : bool
        If True, use the unbinned extended likelihood for the best-fit.
        The saturated-model NLL is not yet implemented for unbinned
        data, so this will raise NotImplementedError.

    Returns
    -------
    chi2_val : float
        The goodness-of-fit test statistic.
    ndof : int
        Number of degrees of freedom.
    p_value : float
        The p-value from the chi2 distribution.
    """
    # Best-fit
    result = fit(model, extended=extended, unbinned=unbinned)

    if unbinned:
        # For unbinned data, the fit is done with the unbinned likelihood
        # (better parameter estimates), but the GoF test itself is a
        # binned deviance computed on post-hoc binned events versus
        # expected yields at the best-fit parameters.  This is standard
        # practice: the unbinned fit constrains the parameters, and the
        # binned chi2 tests the shape agreement.
        parts = []
        for ch in model.channels:
            events = ch.unbinned_data
            if events is None:
                raise ValueError(
                    f"Channel '{ch.name}' has no unbinned data. "
                    f"Call channel.set_unbinned_data(events) first."
                )
            binned, _ = np.histogram(events, bins=ch.edges)
            parts.append(binned.astype(np.float64))
        data = np.concatenate(parts)
        expected = model.expected(result.bestfit)

        # Poisson deviance: 2 * sum [n_i * log(n_i / nu_i) - (n_i - nu_i)]
        # with the convention 0 * log(0) = 0.
        nll_bestfit_binned = -float(_poisson_logpdf(data, expected).sum())
        nll_saturated = -float(_poisson_logpdf(data, data).sum())
        chi2_val = max(0.0, 2.0 * (nll_bestfit_binned - nll_saturated))
    else:
        nll_bestfit = result.nll_min
        data = model.data
        nll_saturated = -float(_poisson_logpdf(data, data).sum())

        if extended:
            data_total = float(data.sum())
            nll_saturated -= float(
                _poisson_logpdf(
                    np.array([data_total]), np.array([data_total])
                ).item()
            )

        chi2_val = max(0.0, 2.0 * (nll_bestfit - nll_saturated))

    # Count floated parameters
    n_floated = sum(1 for p in model.parameters if not p.fixed)
    ndof = model.total_nbins - n_floated

    if ndof <= 0:
        return chi2_val, ndof, 1.0

    p_value = float(stats.chi2.sf(chi2_val, ndof))

    return chi2_val, ndof, p_value
