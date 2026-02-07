"""Tests for Sprint 6: Statistical inference.

Tests use scenarios with known analytical properties: a strong signal
should give high significance and a small upper limit close to the
best fit, while background-only data should give low significance and
an upper limit consistent with expected exclusion.
"""

import numpy as np

from histimator.channels import Channel
from histimator.histograms import Histogram
from histimator.inference import (
    discovery_significance,
    profile_likelihood_ratio,
    upper_limit,
)
from histimator.likelihood import fit
from histimator.model import Model
from histimator.samples import Sample

EDGES = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])


def _build_model(sig_scale=1.0, data_mu=1.0, seed=42):
    """Build a standard test model.

    sig_scale controls the signal template magnitude (sensitivity).
    data_mu controls how much signal is injected in the pseudo-data.
    """
    rng = np.random.default_rng(seed)
    sig_t = sig_scale * np.array([5.0, 20.0, 40.0, 20.0, 5.0])
    bkg_t = np.array([100.0, 100.0, 100.0, 100.0, 100.0])

    expected = data_mu * sig_t + bkg_t
    data = rng.poisson(expected).astype(float)

    sig = Sample("signal", Histogram(sig_t, EDGES))
    sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 20.0))
    bkg = Sample("background", Histogram(bkg_t, EDGES))

    ch = Channel("SR")
    ch.add_sample(sig)
    ch.add_sample(bkg)
    ch.set_data(data)

    m = Model("test")
    m.add_channel(ch)
    return m


class TestProfileLikelihoodRatio:
    def test_zero_at_bestfit(self):
        """PLR should be ~0 at the best-fit value."""
        model = _build_model(data_mu=1.0)
        result = fit(model)
        mu_hat = result.bestfit["mu"]
        t = profile_likelihood_ratio(model, "mu", mu_hat)
        assert t < 0.1  # should be very close to 0

    def test_increases_away_from_bestfit(self):
        """PLR should increase as we move away from the best fit."""
        model = _build_model(data_mu=1.0)
        result = fit(model)
        mu_hat = result.bestfit["mu"]

        t_at_hat = profile_likelihood_ratio(
            model, "mu", mu_hat, unconditional_nll=result.nll_min
        )
        t_far = profile_likelihood_ratio(
            model, "mu", mu_hat + 5.0, unconditional_nll=result.nll_min
        )
        assert t_far > t_at_hat

    def test_non_negative(self):
        """PLR should always be non-negative."""
        model = _build_model(data_mu=1.0)
        for mu in [0.0, 0.5, 1.0, 2.0, 5.0]:
            t = profile_likelihood_ratio(model, "mu", mu)
            assert t >= 0.0


class TestUpperLimit:
    def test_ul_above_bestfit(self):
        """Upper limit should be above the best-fit value."""
        model = _build_model(data_mu=1.0)
        result = fit(model)
        mu_hat = result.bestfit["mu"]
        ul = upper_limit(model, "mu")
        assert ul > mu_hat

    def test_ul_background_only(self):
        """With no signal injected, UL should be a reasonable small number."""
        model = _build_model(sig_scale=2.0, data_mu=0.0, seed=99)
        ul = upper_limit(model, "mu")
        # Should be a finite positive number
        assert 0 < ul < 10.0

    def test_ul_strong_signal(self):
        """With strong signal, UL should be close to the best fit."""
        model = _build_model(sig_scale=5.0, data_mu=3.0, seed=77)
        result = fit(model)
        mu_hat = result.bestfit["mu"]
        ul = upper_limit(model, "mu")
        # UL should be within a few sigma of best fit
        assert ul < mu_hat + 5 * result.errors["mu"]


class TestDiscoverySignificance:
    def test_strong_signal_high_z(self):
        """Large injected signal should give high significance."""
        model = _build_model(sig_scale=5.0, data_mu=5.0, seed=12)
        z, p = discovery_significance(model, "mu")
        assert z > 2.0  # should be quite significant
        assert p < 0.05

    def test_no_signal_low_z(self):
        """Background-only data should give low significance."""
        model = _build_model(sig_scale=1.0, data_mu=0.0, seed=200)
        z, p = discovery_significance(model, "mu")
        assert z < 3.0  # shouldn't claim discovery
        assert p > 0.001

    def test_z_p_consistency(self):
        """Z and p-value should be consistent: p = Phi(-z)."""
        from scipy import stats
        model = _build_model(data_mu=2.0, seed=55)
        z, p = discovery_significance(model, "mu")
        expected_p = stats.norm.sf(z)
        np.testing.assert_allclose(p, expected_p, rtol=1e-10)
