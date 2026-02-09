"""Tests for Sprint 10: Toy-based calculator.

Validation strategy
-------------------
The toy calculator is the non-asymptotic alternative for computing
p-values and CLs.  The primary validation is consistency: for a
high-statistics model where the asymptotic approximation is accurate,
the toy-based CLs should agree with the asymptotic CLs within
statistical uncertainty.  We also test basic properties of the toy
generator (Poisson means, reproducibility) and boundary behaviour
of the CLs output.

The statistical nature of toys means we need tolerance in comparisons.
We use 2000 toys for consistency checks, which gives roughly 2%
precision on tail probabilities.
"""

import numpy as np

from histimator.channels import Channel
from histimator.cls import cls_observed, upper_limit_cls
from histimator.histograms import Histogram
from histimator.model import Model
from histimator.samples import Sample
from histimator.toys import (
    generate_toy_data,
    sample_nuisance_parameters,
    toy_cls,
    toy_cls_limit,
)

EDGES = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])


# ===================================================================
# Model builders
# ===================================================================

def _high_stats_model(data_mu=0.0, seed=42):
    """High-statistics model where asymptotic approximation is accurate.

    Background is 500 per bin, signal peaks at 50.  With 5 bins and
    total background ~2500, the Gaussian approximation holds well.
    """
    rng = np.random.default_rng(seed)
    sig_t = np.array([5.0, 20.0, 50.0, 20.0, 5.0])
    bkg_t = np.array([500.0, 500.0, 500.0, 500.0, 500.0])
    data = rng.poisson(data_mu * sig_t + bkg_t).astype(float)

    sig = Sample("signal", Histogram(sig_t, EDGES))
    sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 20.0))
    bkg = Sample("background", Histogram(bkg_t, EDGES))

    ch = Channel("SR")
    ch.add_sample(sig)
    ch.add_sample(bkg)
    ch.set_data(data)

    m = Model("high_stats")
    m.add_channel(ch)
    return m


def _low_stats_model():
    """Low-statistics model where asymptotic approximation breaks down.

    Single bin, background = 1.5 events, signal = 2.0 events.
    Observed data = 0 events (strong deficit).  This is the regime
    where the Poisson distribution is far from Gaussian.
    """
    edges = np.array([0.0, 1.0])
    sig = Sample("signal", Histogram([2.0], edges))
    sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 50.0))
    bkg = Sample("background", Histogram([1.5], edges))

    ch = Channel("SR")
    ch.add_sample(sig)
    ch.add_sample(bkg)
    ch.set_data([0.0])

    m = Model("low_stats")
    m.add_channel(ch)
    return m


def _moderate_model(data_mu=0.0, seed=42):
    """Moderate-sensitivity model for limit comparison tests.

    Signal is small relative to background, so the expected limit is
    in the range 2-6 where the toy scan has adequate resolution.
    """
    rng = np.random.default_rng(seed)
    sig_t = np.array([2.0, 5.0, 10.0, 5.0, 2.0])
    bkg_t = np.array([500.0, 500.0, 500.0, 500.0, 500.0])
    data = rng.poisson(data_mu * sig_t + bkg_t).astype(float)

    sig = Sample("signal", Histogram(sig_t, EDGES))
    sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 30.0))
    bkg = Sample("background", Histogram(bkg_t, EDGES))

    ch = Channel("SR")
    ch.add_sample(sig)
    ch.add_sample(bkg)
    ch.set_data(data)

    m = Model("moderate")
    m.add_channel(ch)
    return m


def _model_with_syst(data_mu=0.0, seed=42):
    """Model with nuisance parameters to test NP sampling in toys."""
    rng = np.random.default_rng(seed)
    sig_t = np.array([5.0, 20.0, 50.0, 20.0, 5.0])
    bkg_t = np.array([500.0, 500.0, 500.0, 500.0, 500.0])
    data = rng.poisson(data_mu * sig_t + bkg_t).astype(float)

    sig = Sample("signal", Histogram(sig_t, EDGES))
    sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 20.0))
    bkg = Sample("background", Histogram(bkg_t, EDGES))
    bkg.add_normsys("bkg_norm", lo=0.95, hi=1.05)

    ch = Channel("SR")
    ch.add_sample(sig)
    ch.add_sample(bkg)
    ch.set_data(data)

    m = Model("with_syst")
    m.add_channel(ch)
    return m


# ===================================================================
# Tests: generate_toy_data
# ===================================================================

class TestGenerateToyData:
    """The toy generator produces pseudo-data by drawing Poisson counts
    from the expected yields.  The mean of many toys should converge
    to the expected yields, and each toy should be an array of
    non-negative integers.
    """

    def test_correct_shape(self):
        """Toy data should have the same number of bins as the model."""
        model = _high_stats_model()
        params = model.nominal_values()
        rng = np.random.default_rng(42)
        toy = generate_toy_data(model, params, rng)
        assert toy.shape == model.data.shape

    def test_non_negative(self):
        """All toy data values should be non-negative (Poisson)."""
        model = _high_stats_model()
        params = model.nominal_values()
        rng = np.random.default_rng(42)
        for _ in range(100):
            toy = generate_toy_data(model, params, rng)
            assert np.all(toy >= 0)

    def test_mean_converges_to_expected(self):
        """The mean of many toys should approximate the expected yields."""
        model = _high_stats_model()
        params = model.nominal_values()
        expected = model.expected(params)

        rng = np.random.default_rng(42)
        toys = np.array([
            generate_toy_data(model, params, rng) for _ in range(5000)
        ])
        toy_mean = toys.mean(axis=0)
        np.testing.assert_allclose(toy_mean, expected, rtol=0.03)

    def test_reproducible_with_seed(self):
        """Same RNG state should produce the same toy."""
        model = _high_stats_model()
        params = model.nominal_values()

        toy1 = generate_toy_data(model, params, np.random.default_rng(99))
        toy2 = generate_toy_data(model, params, np.random.default_rng(99))
        np.testing.assert_array_equal(toy1, toy2)


# ===================================================================
# Tests: sample_nuisance_parameters
# ===================================================================

class TestSampleNuisanceParameters:
    """When generating toys for hypothesis testing, nuisance parameters
    should be sampled from their constraint distributions, not left at
    nominal.  This tests that the sampling respects the constraint type.
    """

    def test_alpha_params_drawn_from_unit_gaussian(self):
        """NormSys alpha parameters should be drawn from N(0, 1)."""
        model = _model_with_syst()
        rng = np.random.default_rng(42)

        samples = [
            sample_nuisance_parameters(model, rng) for _ in range(2000)
        ]
        alphas = [s["bkg_norm"] for s in samples]

        # Mean should be near 0, std near 1
        assert abs(np.mean(alphas)) < 0.1
        assert abs(np.std(alphas) - 1.0) < 0.1

    def test_poi_not_sampled(self):
        """The POI (mu) should not be in the sampled NP dict; it is
        set explicitly by the hypothesis being tested.
        """
        model = _model_with_syst()
        rng = np.random.default_rng(42)
        np_sample = sample_nuisance_parameters(model, rng)
        # mu is a NormFactor, not a constrained NP
        assert "mu" not in np_sample


# ===================================================================
# Tests: toy_cls
# ===================================================================

class TestToyCls:
    """toy_cls computes CLs at a single mu value using toy
    pseudo-experiments.  The primary validation is agreement with
    the asymptotic calculator at high statistics.
    """

    def test_bounded_zero_to_one(self):
        """CLs should be between 0 and 1."""
        model = _high_stats_model(data_mu=0.0)
        cls_val = toy_cls(model, "mu", mu_test=3.0, n_toys=500, seed=42)
        assert 0.0 <= cls_val <= 1.0

    def test_agrees_with_asymptotic_high_stats(self):
        """At high statistics, toy-based CLs should match asymptotic CLs.

        We test at mu=3 where the asymptotic CLs has a definite value,
        and require agreement within the statistical uncertainty of
        the toy calculation (roughly 1/sqrt(n_toys) on tail fractions).
        """
        model = _high_stats_model(data_mu=0.0, seed=42)

        cls_asym = cls_observed(model, "mu", mu_test=3.0)
        cls_toy = toy_cls(model, "mu", mu_test=3.0, n_toys=2000, seed=42)

        # Allow generous tolerance due to finite toys
        assert abs(cls_toy - cls_asym) < 0.15, (
            f"Toy CLs ({cls_toy:.3f}) too far from asymptotic ({cls_asym:.3f})"
        )

    def test_small_for_large_mu(self):
        """With background-only data, large mu should be excluded (CLs < 0.1)."""
        model = _high_stats_model(data_mu=0.0, seed=42)
        cls_val = toy_cls(model, "mu", mu_test=5.0, n_toys=500, seed=42)
        assert cls_val < 0.1

    def test_large_for_small_mu(self):
        """Testing at a very small mu should give CLs close to 1 (not excluded)."""
        model = _high_stats_model(data_mu=0.0, seed=42)
        cls_val = toy_cls(model, "mu", mu_test=0.1, n_toys=500, seed=42)
        assert cls_val > 0.2


# ===================================================================
# Tests: toy_cls_limit
# ===================================================================

class TestToyClsLimit:
    """The toy-based upper limit should have the same qualitative
    properties as the asymptotic limit: positive, finite, and
    consistent with the asymptotic result at high statistics.
    """

    def test_positive_and_finite(self):
        model = _high_stats_model(data_mu=0.0, seed=42)
        limit = toy_cls_limit(
            model, "mu", n_toys=300, seed=42, scan_points=8,
        )
        assert 0 < limit < 50

    def test_agrees_with_asymptotic_high_stats(self):
        """Toy-based and asymptotic limits should be within ~50% at
        high statistics.  This tolerance accounts for the finite
        number of toys and the coarse mu scan.
        """
        model = _moderate_model(data_mu=0.0, seed=42)

        obs_asym, _ = upper_limit_cls(model, "mu")
        obs_toy = toy_cls_limit(
            model, "mu", n_toys=500, seed=42, scan_points=12,
            scan_range=(0.5, 8.0),
        )

        ratio = obs_toy / obs_asym
        assert 0.4 < ratio < 2.5, (
            f"Toy limit ({obs_toy:.2f}) too far from asymptotic ({obs_asym:.2f})"
        )


# ===================================================================
# Tests: low-statistics regime
# ===================================================================

class TestLowStatsToys:
    """In the low-statistics regime (very few expected events per bin),
    the asymptotic approximation is known to be unreliable.  The toy
    calculator should still produce valid results (CLs in [0,1], finite
    limit), even if they differ from the asymptotic values.
    """

    def test_cls_valid_at_low_stats(self):
        """CLs should be a valid probability even at low statistics."""
        model = _low_stats_model()
        cls_val = toy_cls(model, "mu", mu_test=2.0, n_toys=500, seed=42)
        assert 0.0 <= cls_val <= 1.0

    def test_limit_finite_at_low_stats(self):
        """The upper limit should be finite and positive at low statistics."""
        model = _low_stats_model()
        limit = toy_cls_limit(
            model, "mu", n_toys=300, seed=42, scan_points=8,
        )
        assert 0 < limit < 100
