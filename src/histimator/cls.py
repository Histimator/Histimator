"""CLs method, Asimov datasets, and expected limits with Brazil bands.

The CLs method is the standard frequentist procedure used by every LHC
search to set exclusion limits on new physics.  It protects against
excluding signal hypotheses to which the experiment has no sensitivity,
by dividing the signal+background p-value by the background-only
p-value:

    CLs(mu) = CL_{s+b}(mu) / CL_b(mu)

The signal hypothesis mu is excluded at confidence level alpha when
CLs(mu) < alpha (typically alpha = 0.05 for 95% CL).

The expected limit uses the Asimov dataset (a synthetic dataset where
every bin equals the expected yield under a given hypothesis).  The
Brazil band provides the ±1-sigma and ±2-sigma range of expected limits,
showing the spread of limits that would be obtained from background-only
pseudo-experiments.

References
----------
.. [CLs] A. L. Read, "Presentation of search results: the CLs technique",
   J.Phys. G28 (2002) 2693-2704.
.. [CCGV] G. Cowan, K. Cranmer, E. Gross, O. Vitells,
   "Asymptotic formulae for likelihood-based tests of new physics",
   Eur.Phys.J. C71 (2011) 1554, arXiv:1007.1727.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats

from histimator.channels import Channel
from histimator.model import Model
from histimator.test_statistics import (
    compute_qtilde_mu,
    pvalues_qtilde_mu,
)


# ------------------------------------------------------------------
# Asimov dataset construction
# ------------------------------------------------------------------

def build_asimov_model(
    model: Model,
    params: dict[str, float],
) -> Model:
    """Build a copy of the model with Asimov data.

    The Asimov dataset replaces the observed data in each channel with
    the expected yields evaluated at the given parameter values.  When
    params sets the POI to 0, this gives the background-only Asimov;
    when it sets the POI to some mu, it gives the signal+background
    Asimov.

    Parameters
    ----------
    model : Model
        The original model (not modified).
    params : dict[str, float]
        Parameter values defining the hypothesis.  Must include the POI
        and any nuisance parameters (defaults to nominal for missing).

    Returns
    -------
    Model
        A new Model with identical structure but Asimov data.
    """
    full_params = model.nominal_values()
    full_params.update(params)

    asimov = Model(f"{model.name}_asimov")
    for ch in model.channels:
        new_ch = Channel(ch.name)
        for sample in ch.samples:
            new_ch.add_sample(sample)
        asimov_yields = ch.expected(full_params)
        new_ch.set_data(asimov_yields)
        asimov.add_channel(new_ch)
    return asimov


# ------------------------------------------------------------------
# Asimov sigma computation
# ------------------------------------------------------------------

def compute_asimov_sigma(
    model: Model,
    poi_name: str,
    mu_test: float,
) -> float:
    """Compute the standard deviation of mu_hat from the Asimov dataset.

    Under the asymptotic approximation, mu_hat is Gaussian with variance
    sigma^2.  The Asimov test statistic at mu_test under the background-
    only hypothesis gives:

        sigma = mu_test / sqrt(q_{mu, Asimov})

    Parameters
    ----------
    model : Model
        The original model.
    poi_name : str
        Name of the signal-strength parameter.
    mu_test : float
        The signal strength at which to evaluate sigma.

    Returns
    -------
    float
        The estimated sigma.
    """
    # Build b-only Asimov
    asimov = build_asimov_model(model, {poi_name: 0.0})

    # Compute q_tilde_mu on Asimov data
    q_asimov, _ = compute_qtilde_mu(asimov, poi_name, mu_test)

    if q_asimov <= 0:
        # Fallback: use a small positive value to avoid division by zero
        # This can happen if the model has zero sensitivity
        return mu_test * 100.0  # very large sigma = no sensitivity

    return mu_test / np.sqrt(q_asimov)


# ------------------------------------------------------------------
# CLs computation at a single mu value
# ------------------------------------------------------------------

def cls_observed(
    model: Model,
    poi_name: str,
    mu_test: float,
) -> float:
    """Compute the observed CLs value at a specific signal strength.

    Uses the q_tilde_mu test statistic on observed data with the
    asymptotic p-value formulas.

    Parameters
    ----------
    model : Model
        Model with observed data.
    poi_name : str
        Name of the signal-strength parameter.
    mu_test : float
        The signal strength to test.

    Returns
    -------
    float
        CLs = CL_{s+b} / CL_b.
    """
    qtilde, mu_hat = compute_qtilde_mu(model, poi_name, mu_test)
    sigma = compute_asimov_sigma(model, poi_name, mu_test)
    clsb, clb = pvalues_qtilde_mu(qtilde, mu_test, mu_hat, sigma)

    if clb <= 0:
        return 0.0
    return clsb / clb


def cls_expected_band(
    model: Model,
    poi_name: str,
    mu_test: float,
) -> list[float]:
    """Compute the expected CLs at the five standard quantiles.

    Under the background-only hypothesis, mu_hat ~ N(0, sigma^2).
    The N-sigma quantile of the background-only distribution gives
    mu_hat = N * sigma.  From this, the expected test statistic and
    p-values at each quantile can be computed analytically.

    Returns CLs values at [-2sigma, -1sigma, median, +1sigma, +2sigma].

    Parameters
    ----------
    model : Model
        Model with observed data (used for Asimov sigma computation).
    poi_name : str
        Name of the signal-strength parameter.
    mu_test : float
        The signal strength to test.

    Returns
    -------
    list[float]
        Five CLs values: [-2sigma, -1sigma, median, +1sigma, +2sigma].
    """
    sigma = compute_asimov_sigma(model, poi_name, mu_test)

    band = []
    for n_sigma in [-2, -1, 0, 1, 2]:
        # Under b-only, mu_hat = n_sigma * sigma
        mu_hat_exp = n_sigma * sigma

        # Expected test statistic
        if mu_hat_exp > mu_test:
            # Data prefers more signal than tested: q_tilde = 0
            qtilde_exp = 0.0
        elif mu_hat_exp >= 0:
            # Standard regime: q_tilde = (mu - mu_hat)^2 / sigma^2
            qtilde_exp = ((mu_test - mu_hat_exp) / sigma) ** 2
        else:
            # Negative mu_hat: q_tilde = mu^2/sigma^2 - 2*mu*mu_hat/sigma^2
            qtilde_exp = (
                mu_test ** 2 / sigma ** 2
                - 2 * mu_test * mu_hat_exp / sigma ** 2
            )

        clsb, clb = pvalues_qtilde_mu(
            qtilde_exp, mu_test, mu_hat_exp, sigma
        )
        cls_val = clsb / clb if clb > 0 else 0.0
        band.append(cls_val)

    return band


# ------------------------------------------------------------------
# Upper limit finding
# ------------------------------------------------------------------

def upper_limit_cls(
    model: Model,
    poi_name: str,
    cl: float = 0.95,
    scan_range: tuple[float, float] | None = None,
    tolerance: float = 1e-3,
) -> tuple[float, dict[str, float]]:
    """Compute the observed and expected CLs upper limits.

    Scans over mu to find where CLs = 1 - cl (default 0.05).  Returns
    both the observed limit and the expected limit band (the five
    quantiles of the background-only distribution).

    Parameters
    ----------
    model : Model
        Model with observed data.
    poi_name : str
        Name of the signal-strength parameter.
    cl : float
        Confidence level (default 0.95).
    scan_range : tuple or None
        Search range for the limit.  If None, auto-determined.
    tolerance : float
        Root-finding tolerance.

    Returns
    -------
    observed : float
        The observed 95% CL upper limit.
    expected : dict
        Dictionary with keys '-2sig', '-1sig', 'median', '+1sig', '+2sig'
        giving the expected limit band.
    """
    alpha = 1.0 - cl

    # Determine scan range
    if scan_range is None:
        lo = 0.01
        hi = 10.0
        # Expand if needed
        for p in model.parameters:
            if p.name == poi_name and p.bounds is not None:
                hi = min(p.bounds[1], 50.0)
    else:
        lo, hi = scan_range

    # --- Observed limit ---
    def obs_objective(mu):
        return cls_observed(model, poi_name, mu) - alpha

    observed = _find_crossing(obs_objective, lo, hi, tolerance)

    # --- Expected limits at each quantile ---
    band_labels = ["-2sig", "-1sig", "median", "+1sig", "+2sig"]
    expected = {}

    for i, label in enumerate(band_labels):
        def exp_objective(mu, idx=i):
            band = cls_expected_band(model, poi_name, mu)
            return band[idx] - alpha

        expected[label] = _find_crossing(exp_objective, lo, hi, tolerance)

    return observed, expected


def _find_crossing(objective, lo, hi, tolerance, max_expansions=5):
    """Find the zero crossing of objective(mu) in [lo, hi].

    Expands the upper bound if the root is not bracketed.
    """
    # Check if we bracket the root
    f_lo = objective(lo)
    f_hi = objective(hi)

    # If f_lo and f_hi have the same sign, try expanding
    for _ in range(max_expansions):
        if f_lo * f_hi < 0:
            break
        hi *= 2.0
        f_hi = objective(hi)
    else:
        # If still can't bracket, return hi as a conservative limit
        if f_lo < 0:
            # CLs already below alpha at lo: limit is very small
            return lo
        return hi

    try:
        return float(optimize.brentq(objective, lo, hi, xtol=tolerance))
    except ValueError:
        return hi
