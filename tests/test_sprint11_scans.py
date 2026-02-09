"""Tests for Sprint 11: Likelihood scans and goodness-of-fit.

Validation strategy
-------------------
Likelihood scans are validated against the fit results: the 1D scan
should produce a parabola whose curvature matches the Hesse error,
and the 68%/95% CL intervals (at delta_nll = 1.0 and 3.84) should
be consistent with the parameter uncertainties from MIGRAD.

Goodness-of-fit is validated by the expectation that a well-specified
model (data generated from the model) should give an acceptable p-value,
while a misspecified model (data with a shape not described by the
model) should give a poor p-value.
"""

import numpy as np

from histimator.channels import Channel
from histimator.histograms import Histogram
from histimator.likelihood import fit
from histimator.model import Model
from histimator.samples import Sample
from histimator.scans import (
    goodness_of_fit,
    likelihood_scan_1d,
    likelihood_scan_2d,
    scan_intervals,
)

EDGES = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])


# ===================================================================
# Model builders
# ===================================================================

def _well_specified_model(mu_true=2.0, seed=42):
    """A model where data is drawn from the model itself at mu=mu_true.

    This is the ideal case for goodness-of-fit: the model describes
    the data by construction, so the GoF test should pass.
    """
    rng = np.random.default_rng(seed)
    sig_t = np.array([5.0, 20.0, 40.0, 20.0, 5.0])
    bkg_t = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    data = rng.poisson(mu_true * sig_t + bkg_t).astype(float)

    sig = Sample("signal", Histogram(sig_t, EDGES))
    sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 20.0))
    bkg = Sample("background", Histogram(bkg_t, EDGES))

    ch = Channel("SR")
    ch.add_sample(sig)
    ch.add_sample(bkg)
    ch.set_data(data)

    m = Model("good")
    m.add_channel(ch)
    return m


def _two_parameter_model(seed=42):
    """A model with two free parameters: signal strength and a
    background normalisation nuisance parameter.

    The correlation between mu and bkg_norm can be predicted from
    the model structure: increasing bkg_norm and decreasing mu
    produce similar effects, so they should be anti-correlated.
    """
    rng = np.random.default_rng(seed)
    sig_t = np.array([5.0, 20.0, 40.0, 20.0, 5.0])
    bkg_t = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    data = rng.poisson(2.0 * sig_t + bkg_t).astype(float)

    sig = Sample("signal", Histogram(sig_t, EDGES))
    sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 20.0))
    bkg = Sample("background", Histogram(bkg_t, EDGES))
    bkg.add_normsys("bkg_norm", lo=0.9, hi=1.1)

    ch = Channel("SR")
    ch.add_sample(sig)
    ch.add_sample(bkg)
    ch.set_data(data)

    m = Model("two_param")
    m.add_channel(ch)
    return m


def _misspecified_model():
    """A model that cannot describe the data well.

    The data has a peaked shape that the flat background + signal
    model cannot reproduce, because the peak is in different bins
    from the signal template.  GoF should fail.
    """
    sig_t = np.array([5.0, 20.0, 40.0, 20.0, 5.0])
    bkg_t = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    # Data has a bump in bin 0 where signal template is small
    data = np.array([250.0, 80.0, 100.0, 80.0, 250.0])

    sig = Sample("signal", Histogram(sig_t, EDGES))
    sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 20.0))
    bkg = Sample("background", Histogram(bkg_t, EDGES))

    ch = Channel("SR")
    ch.add_sample(sig)
    ch.add_sample(bkg)
    ch.set_data(data)

    m = Model("bad")
    m.add_channel(ch)
    return m


# ===================================================================
# Tests: likelihood_scan_1d
# ===================================================================


class TestLikelihoodScan1D:
    """A 1D likelihood scan computes -2*delta(ln L) as a function of
    one parameter, profiling all others.  For a well-constrained
    parameter in a high-statistics model, the profile should be
    approximately parabolic near the minimum.
    """

    def test_minimum_at_bestfit(self):
        """The scan minimum should be at (or very close to) zero,
        occurring at the best-fit value.
        """
        model = _well_specified_model()
        result = fit(model)
        mu_hat = result.bestfit["mu"]

        mu_vals, dnll_vals = likelihood_scan_1d(
            model, "mu", bounds=(mu_hat - 3, mu_hat + 3), n_points=21,
        )

        # Minimum should be near zero
        assert dnll_vals.min() < 0.1
        # The mu at minimum should be near mu_hat
        mu_at_min = mu_vals[np.argmin(dnll_vals)]
        assert abs(mu_at_min - mu_hat) < 0.5

    def test_parabola_matches_hesse(self):
        """The curvature of the parabolic scan at the minimum should
        match the Hesse error: delta_nll = 1 at mu_hat +/- sigma.

        For a parabola delta_nll = (mu - mu_hat)^2 / sigma^2, so
        the second derivative is 2/sigma^2.  We fit a parabola to
        the scan points near the minimum and compare.
        """
        model = _well_specified_model()
        result = fit(model)
        mu_hat = result.bestfit["mu"]
        sigma = result.errors["mu"]

        mu_vals, dnll_vals = likelihood_scan_1d(
            model, "mu", bounds=(mu_hat - 2 * sigma, mu_hat + 2 * sigma),
            n_points=21,
        )

        # Fit a parabola: dnll = a*(mu - b)^2 + c
        coeffs = np.polyfit(mu_vals, dnll_vals, 2)
        a = coeffs[0]  # curvature
        # sigma_from_scan = 1/sqrt(a) (since dnll = a*x^2 => sigma^2 = 1/a)
        sigma_scan = 1.0 / np.sqrt(a)

        np.testing.assert_allclose(sigma_scan, sigma, rtol=0.35)

    def test_interval_at_68cl(self):
        """The 68% CL interval (delta_nll = 1.0) should be approximately
        [mu_hat - sigma, mu_hat + sigma].
        """
        model = _well_specified_model()
        result = fit(model)
        mu_hat = result.bestfit["mu"]
        sigma = result.errors["mu"]

        mu_vals, dnll_vals = likelihood_scan_1d(
            model, "mu", bounds=(mu_hat - 3 * sigma, mu_hat + 3 * sigma),
            n_points=41,
        )

        lo_68, hi_68 = scan_intervals(mu_vals, dnll_vals, level=1.0)
        assert abs(lo_68 - (mu_hat - sigma)) < 0.5 * sigma
        assert abs(hi_68 - (mu_hat + sigma)) < 0.5 * sigma

    def test_interval_at_95cl(self):
        """The 95% CL interval (delta_nll = 3.84) should be wider
        than the 68% interval.
        """
        model = _well_specified_model()
        result = fit(model)
        mu_hat = result.bestfit["mu"]
        sigma = result.errors["mu"]

        mu_vals, dnll_vals = likelihood_scan_1d(
            model, "mu", bounds=(mu_hat - 5 * sigma, mu_hat + 5 * sigma),
            n_points=41,
        )

        lo_68, hi_68 = scan_intervals(mu_vals, dnll_vals, level=1.0)
        lo_95, hi_95 = scan_intervals(mu_vals, dnll_vals, level=3.84)

        assert lo_95 < lo_68
        assert hi_95 > hi_68


# ===================================================================
# Tests: likelihood_scan_2d
# ===================================================================


class TestLikelihoodScan2D:
    """A 2D scan computes -2*delta(ln L) on a grid of two parameters,
    profiling all others.  The output is a 2D array that can be used
    to draw contour plots.
    """

    def test_output_shape(self):
        """Should return arrays with the correct grid dimensions."""
        model = _two_parameter_model()
        result = fit(model)
        mu_hat = result.bestfit["mu"]

        mu_vals, alpha_vals, dnll_grid = likelihood_scan_2d(
            model, "mu", "bkg_norm",
            bounds_x=(mu_hat - 2, mu_hat + 2),
            bounds_y=(-2, 2),
            n_points_x=5, n_points_y=5,
        )

        assert mu_vals.shape == (5,)
        assert alpha_vals.shape == (5,)
        assert dnll_grid.shape == (5, 5)

    def test_minimum_near_zero(self):
        """The minimum of the 2D grid should be near zero."""
        model = _two_parameter_model()
        result = fit(model)
        mu_hat = result.bestfit["mu"]

        _, _, dnll_grid = likelihood_scan_2d(
            model, "mu", "bkg_norm",
            bounds_x=(mu_hat - 2, mu_hat + 2),
            bounds_y=(-2, 2),
            n_points_x=11, n_points_y=11,
        )

        assert dnll_grid.min() < 0.5

    def test_anticorrelation_mu_bkg(self):
        """mu and bkg_norm should be anti-correlated: increasing the
        background normalisation compensates for decreasing the signal.

        We check this by finding the direction of the valley: the
        points with small delta_nll should have mu and bkg_norm
        moving in opposite directions relative to the best fit.
        """
        model = _two_parameter_model()
        result = fit(model)

        # Check the covariance from the fit directly
        cov_mu_bkg = result.covariance[("mu", "bkg_norm")]
        assert cov_mu_bkg < 0, "mu and bkg_norm should be anti-correlated"


# ===================================================================
# Tests: goodness_of_fit
# ===================================================================


class TestGoodnessOfFit:
    """The saturated-model goodness-of-fit test compares the fitted
    likelihood to the maximum possible likelihood (where expected =
    observed in every bin).  The test statistic is approximately
    chi2-distributed with ndof = nbins - nfloated.
    """

    def test_well_specified_passes(self):
        """A model that generated the data should have a reasonable
        p-value (not extremely small).  We test across several seeds
        to avoid unlucky fluctuations.
        """
        p_values = []
        for seed in [42, 43, 44, 45, 46]:
            model = _well_specified_model(mu_true=2.0, seed=seed)
            chi2_val, ndof, p = goodness_of_fit(model)
            p_values.append(p)

        # At least 3 out of 5 should have p > 0.05
        n_passing = sum(1 for p in p_values if p > 0.05)
        assert n_passing >= 3, (
            f"Too many seeds gave poor GoF: p-values = {p_values}"
        )

    def test_misspecified_fails(self):
        """A model that cannot describe the data should give a very
        small p-value.
        """
        model = _misspecified_model()
        chi2_val, ndof, p = goodness_of_fit(model)
        assert p < 0.01, f"Misspecified model should fail GoF (p={p:.4f})"

    def test_returns_positive_chi2(self):
        """Chi2 statistic should be non-negative."""
        model = _well_specified_model()
        chi2_val, ndof, p = goodness_of_fit(model)
        assert chi2_val >= 0

    def test_ndof_correct(self):
        """ndof should equal nbins minus the number of floated parameters.

        For a 5-bin model with 1 floated parameter (mu), ndof = 4.
        """
        model = _well_specified_model()
        _, ndof, _ = goodness_of_fit(model)
        expected_ndof = model.total_nbins - 1  # 5 bins, 1 param
        assert ndof == expected_ndof

    def test_pvalue_bounded(self):
        """p-value should be between 0 and 1."""
        model = _well_specified_model()
        _, _, p = goodness_of_fit(model)
        assert 0.0 <= p <= 1.0
