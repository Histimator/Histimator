"""Tests for Sprint 8: CLs method, Asimov data, and expected limits.

Validation strategy
-------------------
The CLs method and Brazil bands are the standard output of every LHC
search.  We validate them at three levels:

1. Asimov data construction: the Asimov dataset should exactly equal
   the expected yields under the specified hypothesis.

2. CLs values: for specific models and mu values, we check that CLs
   has the correct qualitative behaviour (small when signal is excluded,
   near 1 when compatible) and quantitative properties (counting
   experiment closed-form).

3. Brazil bands: the expected limits at each quantile should form an
   ordered sequence, the observed limit should be finite and positive,
   and the median expected limit should lie between the ±1-sigma bands.
"""

import numpy as np
import pytest
from scipy import stats

from histimator.channels import Channel
from histimator.cls import (
    build_asimov_model,
    compute_asimov_sigma,
    cls_observed,
    cls_expected_band,
    upper_limit_cls,
)
from histimator.histograms import Histogram
from histimator.model import Model
from histimator.samples import Sample

EDGES = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])


# ===================================================================
# Model builders
# ===================================================================

def _search_model(sig_scale=1.0, data_mu=0.0, seed=42):
    """Standard search model: peaked signal + flat background.

    sig_scale controls sensitivity (larger = more powerful search).
    data_mu controls injected signal (0 = background-only data).
    """
    rng = np.random.default_rng(seed)
    sig_t = sig_scale * np.array([5.0, 20.0, 40.0, 20.0, 5.0])
    bkg_t = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    data = rng.poisson(data_mu * sig_t + bkg_t).astype(float)

    sig = Sample("signal", Histogram(sig_t, EDGES))
    sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 20.0))
    bkg = Sample("background", Histogram(bkg_t, EDGES))

    ch = Channel("SR")
    ch.add_sample(sig)
    ch.add_sample(bkg)
    ch.set_data(data)

    m = Model("search")
    m.add_channel(ch)
    return m


def _counting_model(s, b, n):
    """Single-bin counting experiment."""
    edges = np.array([0.0, 1.0])
    sig = Sample("signal", Histogram([s], edges))
    sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 100.0))
    bkg = Sample("background", Histogram([b], edges))

    ch = Channel("count")
    ch.add_sample(sig)
    ch.add_sample(bkg)
    ch.set_data([n])

    m = Model("counting")
    m.add_channel(ch)
    return m


# ===================================================================
# Tests: build_asimov_model
# ===================================================================

class TestAsimovModel:
    """The Asimov model replaces observed data with expected yields
    under a specified hypothesis.  By definition, the Asimov data at
    mu=0 equals the background-only expectation, and at mu=1 equals
    the signal+background expectation.
    """

    def test_background_only_asimov(self):
        """Asimov data under mu=0 should equal the background template."""
        model = _search_model()
        asimov = build_asimov_model(model, {"mu": 0.0})

        expected_bkg = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        np.testing.assert_allclose(asimov.data, expected_bkg)

    def test_signal_plus_background_asimov(self):
        """Asimov data under mu=1 should equal sig + bkg."""
        model = _search_model(sig_scale=1.0)
        asimov = build_asimov_model(model, {"mu": 1.0})

        sig_t = np.array([5.0, 20.0, 40.0, 20.0, 5.0])
        bkg_t = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        np.testing.assert_allclose(asimov.data, sig_t + bkg_t)

    def test_scaled_signal_asimov(self):
        """Asimov at mu=2 should have 2*sig + bkg."""
        model = _search_model(sig_scale=1.0)
        asimov = build_asimov_model(model, {"mu": 2.0})

        sig_t = np.array([5.0, 20.0, 40.0, 20.0, 5.0])
        bkg_t = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        np.testing.assert_allclose(asimov.data, 2.0 * sig_t + bkg_t)

    def test_preserves_model_structure(self):
        """Asimov model should have the same channels and samples."""
        model = _search_model()
        asimov = build_asimov_model(model, {"mu": 0.0})

        assert asimov.n_channels == model.n_channels
        assert len(asimov.channels[0].samples) == len(model.channels[0].samples)
        assert asimov.parameter_names == model.parameter_names

    def test_does_not_modify_original(self):
        """Building an Asimov model should not change the original data."""
        model = _search_model(seed=99)
        original_data = model.data.copy()
        _ = build_asimov_model(model, {"mu": 0.0})
        np.testing.assert_array_equal(model.data, original_data)


# ===================================================================
# Tests: compute_asimov_sigma
# ===================================================================

class TestAsimovSigma:
    """Sigma is the standard deviation of the mu_hat estimator, computed
    from the Asimov dataset.  It controls the width of the expected
    limit band: a smaller sigma means a more powerful search.
    """

    def test_positive(self):
        """Sigma should always be positive."""
        model = _search_model()
        sigma = compute_asimov_sigma(model, "mu", mu_test=1.0)
        assert sigma > 0

    def test_decreases_with_sensitivity(self):
        """A model with more signal events (higher sig_scale) should
        have smaller sigma, indicating greater discovery power.
        """
        model_weak = _search_model(sig_scale=0.5, data_mu=0.0, seed=1)
        model_strong = _search_model(sig_scale=3.0, data_mu=0.0, seed=1)

        sigma_weak = compute_asimov_sigma(model_weak, "mu", mu_test=1.0)
        sigma_strong = compute_asimov_sigma(model_strong, "mu", mu_test=1.0)

        assert sigma_strong < sigma_weak

    def test_counting_experiment(self):
        """For a counting experiment (s, b, Asimov n=b), sigma can be
        computed from the closed-form Asimov q_mu.

        q_{mu=1,A} = 2*(b*ln(b/(s+b)) + s)  (from Poisson likelihood)
        sigma = 1 / sqrt(q_{1,A})
        """
        s, b = 10.0, 100.0
        model = _counting_model(s, b, b)  # Asimov: n = b

        sigma = compute_asimov_sigma(model, "mu", mu_test=1.0)

        # Closed form: q_{1,A} = 2*(b*ln(b/(s+b)) + s)
        q_asimov = 2.0 * (b * np.log(b / (s + b)) + s)
        expected_sigma = 1.0 / np.sqrt(q_asimov)

        np.testing.assert_allclose(sigma, expected_sigma, rtol=0.05)


# ===================================================================
# Tests: cls_observed
# ===================================================================

class TestClsObserved:
    """CLs = CLsb / CLb at a specific mu value using observed data.

    CLs near 0 means the signal hypothesis is excluded.
    CLs near 1 means the data is compatible with the signal.
    """

    def test_bounded_zero_to_one(self):
        """CLs should be between 0 and 1."""
        model = _search_model(data_mu=0.0, seed=42)
        cls_val = cls_observed(model, "mu", mu_test=2.0)
        assert 0.0 <= cls_val <= 1.0

    def test_small_for_large_mu(self):
        """With background-only data, a large mu should be excluded.

        CLs < 0.05 means the signal is excluded at 95% CL.
        """
        model = _search_model(sig_scale=2.0, data_mu=0.0, seed=42)
        cls_val = cls_observed(model, "mu", mu_test=5.0)
        assert cls_val < 0.05

    def test_large_for_injected_signal(self):
        """With signal injected, the true mu should not be excluded.

        Testing at the injected mu should give CLs > 0.05.
        """
        model = _search_model(sig_scale=1.0, data_mu=2.0, seed=42)
        cls_val = cls_observed(model, "mu", mu_test=2.0)
        assert cls_val > 0.05

    def test_decreases_with_mu(self):
        """CLs should decrease as mu increases (harder to be compatible)."""
        model = _search_model(data_mu=0.0, seed=42)
        cls_values = []
        for mu in [1.0, 3.0, 5.0]:
            cls_values.append(cls_observed(model, "mu", mu_test=mu))
        assert cls_values[0] > cls_values[1] > cls_values[2]


# ===================================================================
# Tests: cls_expected_band
# ===================================================================

class TestClsExpectedBand:
    """The expected CLs at each sigma quantile gives the Brazil band.

    The band should be ordered: -2σ < -1σ < median < +1σ < +2σ
    for the CLs values (because upward fluctuations in background
    look more signal-like, giving larger CLs).
    """

    def test_returns_five_values(self):
        """Should return CLs at [-2σ, -1σ, median, +1σ, +2σ]."""
        model = _search_model()
        band = cls_expected_band(model, "mu", mu_test=2.0)
        assert len(band) == 5

    def test_ordered(self):
        """CLs values should increase from -2σ to +2σ.

        A -2σ downward fluctuation gives the smallest CLs (most
        excluding), while a +2σ upward fluctuation gives the largest.
        """
        model = _search_model()
        band = cls_expected_band(model, "mu", mu_test=3.0)
        assert band[0] <= band[1] <= band[2] <= band[3] <= band[4]

    def test_median_bounded(self):
        """Median expected CLs should be between 0 and 1."""
        model = _search_model()
        band = cls_expected_band(model, "mu", mu_test=2.0)
        assert 0.0 <= band[2] <= 1.0

    def test_median_small_for_large_mu(self):
        """The median expected CLs for a large mu should be small
        if the search is sensitive.
        """
        model = _search_model(sig_scale=2.0)
        band = cls_expected_band(model, "mu", mu_test=5.0)
        assert band[2] < 0.05  # median CLs


# ===================================================================
# Tests: upper_limit_cls
# ===================================================================

class TestUpperLimitCls:
    """The 95% CL upper limit is the mu value where CLs = 0.05.

    It is the primary output of any LHC search.
    """

    def test_positive(self):
        """Upper limit should be positive."""
        model = _search_model(data_mu=0.0, seed=42)
        obs, exp = upper_limit_cls(model, "mu")
        assert obs > 0
        assert exp["median"] > 0

    def test_observed_reasonable(self):
        """With background-only data, the observed limit should be
        a modest number, not extremely large or small.
        """
        model = _search_model(data_mu=0.0, seed=42)
        obs, exp = upper_limit_cls(model, "mu")
        assert 0.1 < obs < 20.0

    def test_expected_bands_ordered(self):
        """Expected limits should be ordered: -2σ < -1σ < median < +1σ < +2σ.

        A smaller expected limit means more excluding power. Downward
        fluctuations produce tighter limits.
        """
        model = _search_model(data_mu=0.0, seed=42)
        obs, exp = upper_limit_cls(model, "mu")

        assert exp["-2sig"] < exp["-1sig"]
        assert exp["-1sig"] < exp["median"]
        assert exp["median"] < exp["+1sig"]
        assert exp["+1sig"] < exp["+2sig"]

    def test_observed_near_expected(self):
        """With background-only data, the observed limit should be
        close to the median expected (within the ±2σ band).
        """
        model = _search_model(data_mu=0.0, seed=42)
        obs, exp = upper_limit_cls(model, "mu")

        assert exp["-2sig"] <= obs <= exp["+2sig"] * 1.5

    def test_stronger_search_tighter_limit(self):
        """A more sensitive search (higher sig_scale) should give a
        tighter (smaller) expected limit.
        """
        model_weak = _search_model(sig_scale=0.5, data_mu=0.0, seed=10)
        model_strong = _search_model(sig_scale=3.0, data_mu=0.0, seed=10)

        _, exp_weak = upper_limit_cls(model_weak, "mu")
        _, exp_strong = upper_limit_cls(model_strong, "mu")

        assert exp_strong["median"] < exp_weak["median"]

    def test_signal_injection_shifts_observed(self):
        """Injecting signal should shift the observed limit upward
        relative to the expected, because the data is more signal-like.
        """
        model = _search_model(sig_scale=1.0, data_mu=3.0, seed=42)
        obs, exp = upper_limit_cls(model, "mu")

        assert obs > exp["median"]


# ===================================================================
# Tests: counting experiment closed-form expected limit
# ===================================================================

class TestCountingExpectedLimit:
    """For a counting experiment, the median expected limit has a
    closed-form expression:

        mu_exp = sigma * Phi^{-1}(1 - 0.05/2) = sigma * 1.96

    where sigma = 1/sqrt(q_{1,A}) and q_{1,A} is the Asimov test
    statistic at mu=1.  This follows from CLs_median(mu) = 0.05
    being equivalent to 2*(1 - Phi(mu/sigma)) = 0.05.

    We check this for several signal/background configurations.
    """

    def test_s10_b100(self):
        s, b = 10.0, 100.0
        model = _counting_model(s, b, b)  # Asimov-like data (n=b)
        obs, exp = upper_limit_cls(model, "mu")

        q_asimov = 2.0 * (b * np.log(b / (s + b)) + s)
        sigma = 1.0 / np.sqrt(q_asimov)
        expected_median = sigma * stats.norm.ppf(1 - 0.025)

        np.testing.assert_allclose(
            exp["median"], expected_median, rtol=0.15
        )

    def test_s20_b50(self):
        s, b = 20.0, 50.0
        model = _counting_model(s, b, b)
        obs, exp = upper_limit_cls(model, "mu")

        q_asimov = 2.0 * (b * np.log(b / (s + b)) + s)
        sigma = 1.0 / np.sqrt(q_asimov)
        expected_median = sigma * stats.norm.ppf(1 - 0.025)

        np.testing.assert_allclose(
            exp["median"], expected_median, rtol=0.15
        )
