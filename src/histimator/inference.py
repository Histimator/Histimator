"""Statistical inference: hypothesis tests and confidence intervals.

Provides profile-likelihood-ratio test statistics and the tools to
compute exclusion limits and discovery significance.

The key functions are:

- ``upper_limit``: find the 95% CL upper limit on a parameter of interest
  using the profile likelihood ratio.
- ``discovery_pvalue``: compute the p-value for rejecting the background-only
  hypothesis using the q_0 test statistic.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats

from histimator.likelihood import BinnedNLL, fit
from histimator.model import Model


def _profile_nll(
    model: Model,
    poi_name: str,
    poi_value: float,
    extended: bool = True,
) -> float:
    """Minimise the NLL with the POI fixed at a given value.

    All other parameters (nuisance parameters) are profiled (floated).
    Returns the minimum NLL value.
    """
    nll = BinnedNLL(model, extended=extended)
    par_names = nll._par_names
    start = [p.value for p in model.parameters]

    from iminuit import Minuit

    m = Minuit(nll, start, name=par_names)
    m.errordef = 1

    for p in model.parameters:
        if p.bounds is not None:
            m.limits[p.name] = p.bounds
        if p.fixed:
            m.fixed[p.name] = True

    # Fix the POI at the test value
    m.values[poi_name] = poi_value
    m.fixed[poi_name] = True

    m.migrad()
    return float(m.fval)


def profile_likelihood_ratio(
    model: Model,
    poi_name: str,
    poi_value: float,
    unconditional_nll: float | None = None,
    extended: bool = True,
) -> float:
    """Compute the profile likelihood ratio test statistic.

    .. math::
        t(\\mu) = 2 [ \\ell(\\mu, \\hat{\\hat{\\nu}})
                   - \\ell(\\hat{\\mu}, \\hat{\\nu}) ]

    Parameters
    ----------
    model : Model
        Fully built model with data.
    poi_name : str
        Name of the parameter of interest.
    poi_value : float
        Test value of the POI.
    unconditional_nll : float or None
        If provided, skip the unconditional fit and use this value.
    extended : bool
        Use extended likelihood.

    Returns
    -------
    float
        The test statistic value (non-negative).
    """
    conditional = _profile_nll(model, poi_name, poi_value, extended=extended)

    if unconditional_nll is None:
        result = fit(model, extended=extended)
        unconditional_nll = result.nll_min

    return max(0.0, 2.0 * (conditional - unconditional_nll))


def upper_limit(
    model: Model,
    poi_name: str,
    cl: float = 0.95,
    extended: bool = True,
    scan_range: tuple[float, float] | None = None,
    tolerance: float = 1e-3,
) -> float:
    """Compute the upper limit on a parameter of interest.

    Uses the profile-likelihood-ratio inversion method. The upper limit
    mu_up is defined as the value of the POI where the test statistic
    equals the critical value from chi2(1 dof).

    Parameters
    ----------
    model : Model
        Fully built model with data.
    poi_name : str
        Name of the parameter of interest.
    cl : float
        Confidence level (default 0.95 for 95% CL).
    extended : bool
        Use extended likelihood.
    scan_range : tuple[float, float] or None
        Search range for the upper limit. If None, automatically determined.
    tolerance : float
        Numerical tolerance for the root finding.

    Returns
    -------
    float
        The upper limit on the POI.
    """
    # Critical value: chi2(1) quantile
    threshold = stats.chi2.ppf(cl, df=1)

    # Unconditional fit
    result = fit(model, extended=extended)
    mu_hat = result.bestfit[poi_name]
    nll_min = result.nll_min
    mu_error = result.errors[poi_name]

    # Determine scan range
    if scan_range is None:
        lo = max(mu_hat, 0.0)
        hi = mu_hat + 10 * mu_error
        # Find the POI parameter to get its upper bound
        for p in model.parameters:
            if p.name == poi_name and p.bounds is not None:
                hi = min(hi, p.bounds[1])
    else:
        lo, hi = scan_range

    # Root-finding: find mu where t(mu) = threshold
    def objective(mu):
        t = profile_likelihood_ratio(
            model, poi_name, mu, unconditional_nll=nll_min, extended=extended
        )
        return t - threshold

    # Make sure we bracket the root
    try:
        result_brent = optimize.brentq(objective, lo, hi, xtol=tolerance)
        return float(result_brent)
    except ValueError:
        # If we can't bracket, try expanding the range
        for factor in [2, 5, 10, 50]:
            try:
                result_brent = optimize.brentq(
                    objective, lo, hi * factor, xtol=tolerance
                )
                return float(result_brent)
            except ValueError:
                continue
        raise RuntimeError(
            f"Could not find upper limit. The test statistic may not cross "
            f"the threshold {threshold:.2f} in the search range."
        ) from None


def discovery_significance(
    model: Model,
    poi_name: str,
    extended: bool = True,
) -> tuple[float, float]:
    """Compute the discovery significance (Z-value and p-value).

    Tests the background-only hypothesis (POI = 0) against the
    best-fit alternative. Uses the q_0 test statistic:

    .. math::
        q_0 = \\begin{cases}
            2[\\ell(0, \\hat{\\hat{\\nu}})
            - \\ell(\\hat{\\mu}, \\hat{\\nu})]
            & \\text{if } \\hat{\\mu} \\geq 0 \\\\
            0 & \\text{if } \\hat{\\mu} < 0
        \\end{cases}

    Parameters
    ----------
    model : Model
        Fully built model with data.
    poi_name : str
        Name of the parameter of interest.
    extended : bool
        Use extended likelihood.

    Returns
    -------
    z_value : float
        Significance in units of standard deviations.
    p_value : float
        One-sided p-value for rejecting the background-only hypothesis.
    """
    # Unconditional fit
    result = fit(model, extended=extended)
    mu_hat = result.bestfit[poi_name]
    nll_min = result.nll_min

    if mu_hat < 0:
        return 0.0, 0.5

    # Background-only NLL (POI = 0)
    q0 = profile_likelihood_ratio(
        model, poi_name, 0.0, unconditional_nll=nll_min, extended=extended
    )

    z = np.sqrt(q0)
    p = stats.norm.sf(z)  # one-sided p-value

    return float(z), float(p)
