"""One-sided test statistics and asymptotic p-values (CCGV formalism).

Implements the three test statistics used by the LHC programme for
hypothesis testing, as defined in Cowan, Cranmer, Gross, and Vitells
(arXiv:1007.1727):

- ``q_0``  : discovery test statistic (is there signal?)
- ``q_mu`` : exclusion test statistic (is signal strength mu excluded?)
- ``q_tilde_mu`` : same as q_mu with the physical boundary mu_hat >= 0

Each statistic is a bounded variant of the profile likelihood ratio
that zeroes out when the data fluctuation goes the "wrong way" for the
hypothesis being tested.  The asymptotic distributions of these bounded
statistics are not plain chi2(1) but specific mixtures involving delta
functions and half-chi2, which is why the p-value functions here exist.

The p-value functions return ``(CLsb, CLb)`` pairs suitable for direct
use in the CLs construction: ``CLs = CLsb / CLb``.

References
----------
.. [CCGV] G. Cowan, K. Cranmer, E. Gross, O. Vitells,
   "Asymptotic formulae for likelihood-based tests of new physics",
   Eur.Phys.J. C71 (2011) 1554, arXiv:1007.1727.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from histimator.likelihood import BinnedNLL
from histimator.model import Model

# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _fit_unconstrained(model: Model, poi_name: str):
    """Unconditional fit with the POI lower bound extended to allow
    negative values.

    The standard model often sets the POI bounds to (0, 10) or similar,
    but the CCGV test statistics require the true unconstrained MLE
    which may be negative (indicating a data deficit relative to
    background).  This helper temporarily widens the lower bound.

    Uses extended=False because the per-bin Poisson sum already gives
    the standard extended likelihood (the -nu_i terms provide the
    normalisation constraint).  This matches the textbook formulas
    from CCGV.

    Returns (nll_min, mu_hat, fit_result).
    """
    nll = BinnedNLL(model, extended=False)
    par_names = nll._par_names
    start = [p.value for p in model.parameters]

    from iminuit import Minuit

    m = Minuit(nll, start, name=par_names)
    m.errordef = 1

    for p in model.parameters:
        if p.name == poi_name:
            # Extend lower bound to allow negative mu_hat
            lo = -10.0
            hi = p.bounds[1] if p.bounds is not None else 100.0
            m.limits[poi_name] = (lo, hi)
        else:
            if p.bounds is not None:
                m.limits[p.name] = p.bounds
            if p.fixed:
                m.fixed[p.name] = True

    m.migrad()
    mu_hat = float(m.values[poi_name])
    nll_min = float(m.fval)
    return nll_min, mu_hat, m


def _profile_nll(model: Model, poi_name: str, poi_value: float) -> float:
    """Minimise the NLL with the POI fixed at a given value.

    All other parameters (nuisance parameters) are profiled (floated).
    Returns the minimum NLL value.
    """
    nll = BinnedNLL(model, extended=False)
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

    m.values[poi_name] = poi_value
    m.fixed[poi_name] = True

    m.migrad()
    return float(m.fval)


# ------------------------------------------------------------------
# Test statistic computation
# ------------------------------------------------------------------

def compute_q0(
    model: Model,
    poi_name: str,
) -> tuple[float, float]:
    """Discovery test statistic q_0 (CCGV eq 12).

    Tests the background-only hypothesis (mu = 0).  Returns zero when
    the best-fit mu is negative, because a deficit should never count
    as evidence for new signal.

    Parameters
    ----------
    model : Model
        Fully constructed model with observed data.
    poi_name : str
        Name of the signal-strength parameter.

    Returns
    -------
    q0 : float
        The test statistic value (non-negative).
    mu_hat : float
        The unconstrained best-fit value of the POI.
    """
    nll_uncond, mu_hat, _ = _fit_unconstrained(model, poi_name)

    if mu_hat < 0:
        return 0.0, mu_hat

    nll_cond = _profile_nll(model, poi_name, 0.0)
    q0 = max(0.0, 2.0 * (nll_cond - nll_uncond))
    return q0, mu_hat


def compute_qmu(
    model: Model,
    poi_name: str,
    mu_test: float,
) -> tuple[float, float]:
    """One-sided exclusion test statistic q_mu (CCGV eq 14).

    Tests whether signal strength mu_test is compatible with the data.
    Returns zero when mu_hat > mu_test, because an upward fluctuation
    beyond the tested strength should not count as evidence against it.

    Parameters
    ----------
    model : Model
        Fully constructed model with observed data.
    poi_name : str
        Name of the signal-strength parameter.
    mu_test : float
        The signal strength value to test.

    Returns
    -------
    qmu : float
        The test statistic value (non-negative).
    mu_hat : float
        The unconstrained best-fit value of the POI.
    """
    nll_uncond, mu_hat, _ = _fit_unconstrained(model, poi_name)

    if mu_hat > mu_test:
        return 0.0, mu_hat

    nll_cond = _profile_nll(model, poi_name, mu_test)
    qmu = max(0.0, 2.0 * (nll_cond - nll_uncond))
    return qmu, mu_hat


def compute_qtilde_mu(
    model: Model,
    poi_name: str,
    mu_test: float,
) -> tuple[float, float]:
    """Exclusion test statistic with physical boundary q_tilde_mu
    (CCGV eq 16).

    Like q_mu but additionally enforces the physical constraint that
    the signal strength cannot be negative.  When mu_hat < 0, the
    denominator of the likelihood ratio changes from NLL(mu_hat) to
    NLL(0), representing the boundary of the physical region.

    Parameters
    ----------
    model : Model
        Fully constructed model with observed data.
    poi_name : str
        Name of the signal-strength parameter.
    mu_test : float
        The signal strength value to test.

    Returns
    -------
    qtilde : float
        The test statistic value (non-negative).
    mu_hat : float
        The unconstrained best-fit value of the POI (may be negative).
    """
    nll_uncond, mu_hat, _ = _fit_unconstrained(model, poi_name)

    if mu_hat > mu_test:
        return 0.0, mu_hat

    nll_cond = _profile_nll(model, poi_name, mu_test)

    if mu_hat >= 0:
        # Same as q_mu: denominator is the unconditional NLL
        qtilde = max(0.0, 2.0 * (nll_cond - nll_uncond))
    else:
        # mu_hat < 0: denominator switches to NLL at mu = 0
        nll_at_zero = _profile_nll(model, poi_name, 0.0)
        qtilde = max(0.0, 2.0 * (nll_cond - nll_at_zero))

    return qtilde, mu_hat


# ------------------------------------------------------------------
# Asymptotic p-value functions
# ------------------------------------------------------------------

def pvalue_q0(q0: float) -> float:
    """Asymptotic p-value for the discovery test statistic q_0.

    Under the background-only hypothesis, q_0 follows a half-chi2(1)
    distribution: it is zero with probability 1/2, and chi2(1) with
    probability 1/2.  The one-sided p-value is therefore:

        p_0 = 1 - Phi(sqrt(q_0))    when q_0 > 0
        p_0 = 0.5                    when q_0 = 0

    (CCGV eqs 52-53.)

    Parameters
    ----------
    q0 : float
        Observed value of the q_0 test statistic.

    Returns
    -------
    float
        The p-value for rejecting the background-only hypothesis.
    """
    if q0 <= 0.0:
        return 0.5
    return float(stats.norm.sf(np.sqrt(q0)))


def pvalues_qmu(
    qmu: float,
    mu_test: float,
    sigma: float,
) -> tuple[float, float]:
    """Asymptotic p-values for the q_mu test statistic.

    Returns the pair (CLsb, CLb) needed for the CLs construction,
    where CLs = CLsb / CLb.

    CLsb is the tail probability of q_mu under the signal+background
    hypothesis (mu = mu_test).  CLb is the tail probability under the
    background-only hypothesis (mu = 0).

    The parameter sigma is the standard deviation of the mu_hat
    estimator, which can be obtained from the Asimov dataset via
    sigma = mu_test / sqrt(q_{mu,Asimov}).

    (CCGV eqs 59-62.)

    Parameters
    ----------
    qmu : float
        Observed value of the q_mu test statistic.
    mu_test : float
        The signal strength being tested.
    sigma : float
        Standard deviation of mu_hat (from Asimov or Hesse).

    Returns
    -------
    CLsb : float
        Tail probability under the s+b hypothesis.
    CLb : float
        Tail probability under the background-only hypothesis.
    """
    if qmu <= 0.0:
        clsb = 0.5
        clb = float(stats.norm.cdf(mu_test / sigma))
        return clsb, clb

    sqrt_q = np.sqrt(qmu)
    clsb = float(stats.norm.sf(sqrt_q))
    clb = float(stats.norm.sf(sqrt_q - mu_test / sigma))
    return clsb, clb


def pvalues_qtilde_mu(
    qtilde: float,
    mu_test: float,
    mu_hat: float,
    sigma: float,
) -> tuple[float, float]:
    """Asymptotic p-values for the q_tilde_mu test statistic.

    When mu_hat >= 0, the formulas are identical to those for q_mu.
    When mu_hat < 0, the physical boundary changes the distribution
    and the p-value expressions differ (CCGV eqs 64-66).

    The key difference in the mu_hat < 0 branch: the tail probability
    is computed directly from the Gaussian distribution of mu_hat
    rather than from the chi2 distribution of the test statistic,
    because the test statistic has a different functional form in
    this region.

    Parameters
    ----------
    qtilde : float
        Observed value of the q_tilde_mu test statistic.
    mu_test : float
        The signal strength being tested.
    mu_hat : float
        The unconstrained best-fit signal strength.
    sigma : float
        Standard deviation of mu_hat.

    Returns
    -------
    CLsb : float
        Tail probability under the s+b hypothesis.
    CLb : float
        Tail probability under the background-only hypothesis.
    """
    if qtilde <= 0.0:
        # mu_hat > mu_test: data prefers more signal than tested
        clsb = 0.5
        clb = float(stats.norm.cdf(mu_test / sigma))
        return clsb, clb

    if mu_hat >= 0:
        # Same as q_mu
        return pvalues_qmu(qtilde, mu_test, sigma)

    # mu_hat < 0: physical boundary branch (CCGV eq 66)
    #
    # The tail probability P(q_tilde >= q_obs | mu') equals
    # P(mu_hat <= mu_hat_obs | mu_hat ~ N(mu', sigma^2)) because
    # q_tilde is monotonically decreasing in mu_hat for mu_hat < 0.
    #
    # Under s+b (mu' = mu_test):
    #   CLsb = Phi((mu_hat - mu_test) / sigma)
    # Under b-only (mu' = 0):
    #   CLb = Phi(mu_hat / sigma)
    clsb = float(stats.norm.cdf((mu_hat - mu_test) / sigma))
    clb = float(stats.norm.cdf(mu_hat / sigma))
    return clsb, clb
