"""Tests for Sprint 5: BinnedNLL and fitting.

The core validation strategy here is to construct models with known
true parameter values, generate pseudo-data from those true values,
and then verify that the fit recovers the injected parameters within
statistical tolerance.
"""

import numpy as np

from histimator.channels import Channel
from histimator.histograms import Histogram
from histimator.likelihood import BinnedNLL, _poisson_logpdf, fit
from histimator.model import Model
from histimator.samples import Sample

EDGES = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])


# ===========================================================================
# Poisson log-pdf unit tests
# ===========================================================================

class TestPoissonLogPdf:
    def test_known_value(self):
        """Poisson(k=5, mu=5) has a known log-probability."""
        # log(5^5 * exp(-5) / 5!) = 5*ln(5) - 5 - ln(120)
        expected = 5 * np.log(5) - 5 - np.log(120)
        result = _poisson_logpdf(np.array([5.0]), np.array([5.0]))
        np.testing.assert_allclose(result, [expected], rtol=1e-10)

    def test_zero_data(self):
        """Poisson(k=0, mu=3) = exp(-3)."""
        result = _poisson_logpdf(np.array([0.0]), np.array([3.0]))
        np.testing.assert_allclose(result, [-3.0], rtol=1e-10)

    def test_vectorised(self):
        k = np.array([0, 1, 2, 3])
        mu = np.array([1.0, 1.0, 1.0, 1.0])
        result = _poisson_logpdf(k, mu)
        assert result.shape == (4,)


# ===========================================================================
# BinnedNLL tests
# ===========================================================================

class TestBinnedNLL:
    def _make_simple_model(self):
        """Signal (mu-scaled) + flat background, data = sig + bkg at mu=1."""
        sig = Sample("signal", Histogram([10, 20, 30, 20, 10], EDGES))
        sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 5.0))

        bkg = Sample("background", Histogram([50, 50, 50, 50, 50], EDGES))

        ch = Channel("SR")
        ch.add_sample(sig)
        ch.add_sample(bkg)
        # Data = expected at mu=1
        ch.set_data([60, 70, 80, 70, 60])

        m = Model("test")
        m.add_channel(ch)
        return m

    def test_nll_callable(self):
        model = self._make_simple_model()
        nll = BinnedNLL(model)
        val = nll(np.array([1.0]))
        assert np.isfinite(val)

    def test_nll_minimum_at_true(self):
        """NLL should be minimised near the true parameter value."""
        model = self._make_simple_model()
        nll = BinnedNLL(model)
        nll_at_1 = nll(np.array([1.0]))
        nll_at_05 = nll(np.array([0.5]))
        nll_at_2 = nll(np.array([2.0]))
        # The minimum should be at or near mu=1 since data was generated there
        assert nll_at_1 < nll_at_05
        assert nll_at_1 < nll_at_2

    def test_nll_has_parameters(self):
        """Modern iminuit uses _parameters dict."""
        model = self._make_simple_model()
        nll = BinnedNLL(model)
        assert hasattr(nll, "_parameters")
        assert "mu" in nll._parameters


# ===========================================================================
# Fit tests
# ===========================================================================

class TestFit:
    def _make_model_with_poisson_data(self, true_mu=1.5, seed=42):
        """Build a model and generate Poisson pseudo-data at a known mu."""
        rng = np.random.default_rng(seed)

        sig_template = np.array([10.0, 30.0, 50.0, 30.0, 10.0])
        bkg_template = np.array([50.0, 50.0, 50.0, 50.0, 50.0])

        expected = true_mu * sig_template + bkg_template
        data = rng.poisson(expected).astype(float)

        sig = Sample("signal", Histogram(sig_template, EDGES))
        sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 10.0))

        bkg = Sample("background", Histogram(bkg_template, EDGES))

        ch = Channel("SR")
        ch.add_sample(sig)
        ch.add_sample(bkg)
        ch.set_data(data)

        m = Model("test")
        m.add_channel(ch)
        return m, true_mu

    def test_fit_recovers_mu(self):
        model, true_mu = self._make_model_with_poisson_data(true_mu=1.5)
        result = fit(model)
        assert result.valid
        # Should recover mu within ~2 sigma
        assert abs(result.bestfit["mu"] - true_mu) < 2 * result.errors["mu"]

    def test_fit_result_has_errors(self):
        model, _ = self._make_model_with_poisson_data()
        result = fit(model)
        assert "mu" in result.errors
        assert result.errors["mu"] > 0

    def test_fit_result_has_covariance(self):
        model, _ = self._make_model_with_poisson_data()
        result = fit(model)
        assert result.covariance is not None
        assert ("mu", "mu") in result.covariance
        # Variance should be positive
        assert result.covariance[("mu", "mu")] > 0

    def test_fit_with_minos(self):
        model, _ = self._make_model_with_poisson_data()
        result = fit(model, run_minos=True)
        assert result.minos_errors is not None
        assert "mu" in result.minos_errors
        lo, hi = result.minos_errors["mu"]
        assert lo < 0  # lower error is negative
        assert hi > 0  # upper error is positive

    def test_fit_with_nuisance(self):
        """Model with a nuisance parameter should still converge."""
        rng = np.random.default_rng(123)

        sig_t = np.array([20.0, 40.0, 20.0])
        bkg_t = np.array([100.0, 100.0, 100.0])
        edges = np.array([0.0, 1.0, 2.0, 3.0])

        true_mu = 2.0
        data = rng.poisson(true_mu * sig_t + bkg_t).astype(float)

        sig = Sample("signal", Histogram(sig_t, edges))
        sig.add_normfactor("mu", nominal=1.0, bounds=(0, 10))

        bkg = Sample("background", Histogram(bkg_t, edges))
        bkg.add_normsys("bkg_norm", lo=0.9, hi=1.1)

        ch = Channel("SR")
        ch.add_sample(sig)
        ch.add_sample(bkg)
        ch.set_data(data)

        m = Model("nuis_test")
        m.add_channel(ch)

        result = fit(m)
        assert result.valid
        assert len(result.bestfit) == 2  # mu and bkg_norm
        assert "bkg_norm" in result.bestfit

    def test_fit_multi_channel(self):
        """Two-channel fit with shared signal strength."""
        rng = np.random.default_rng(456)
        edges = np.array([0.0, 1.0, 2.0])
        true_mu = 1.8

        # Signal region
        sig_sr = Sample("sig", Histogram([30.0, 30.0], edges))
        sig_sr.add_normfactor("mu", nominal=1.0, bounds=(0, 10))
        bkg_sr = Sample("bkg", Histogram([100.0, 100.0], edges))
        data_sr = rng.poisson(true_mu * np.array([30.0, 30.0]) + np.array([100.0, 100.0]))

        sr = Channel("SR")
        sr.add_sample(sig_sr)
        sr.add_sample(bkg_sr)
        sr.set_data(data_sr)

        # Control region (background only)
        bkg_cr = Sample("bkg", Histogram([200.0, 200.0], edges))
        data_cr = rng.poisson([200.0, 200.0])

        cr = Channel("CR")
        cr.add_sample(bkg_cr)
        cr.set_data(data_cr)

        m = Model("multi")
        m.add_channel(sr)
        m.add_channel(cr)

        result = fit(m)
        assert result.valid
        assert abs(result.bestfit["mu"] - true_mu) < 3 * result.errors["mu"]

    def test_fit_nll_min_returned(self):
        model, _ = self._make_model_with_poisson_data()
        result = fit(model)
        assert np.isfinite(result.nll_min)
        assert result.nll_min > 0
