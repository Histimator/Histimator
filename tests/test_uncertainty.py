"""Tests for Sprint 4: Uncertainty equivalence.

The mathematical foundation being tested:

For a bin with n_j events drawn from Poisson(lambda_j):
    Var(n_j) = lambda_j ≈ n_j  (for observed counts)

For bootstrap resampling of n events with bin probability p_j = n_j/n:
    n_j* ~ Binomial(n, p_j)
    Var(n_j*) = n * p_j * (1 - p_j) = n_j * (1 - n_j/n)
    → n_j  as n → ∞  (Poisson limit when p_j << 1)

For weighted events with sum-of-weights s_j = sum(w_i for i in bin j):
    Var(s_j) = sum(w_i^2 for i in bin j) = sumw2_j
    Bootstrap resampling converges to this same quantity.

These tests verify the convergence empirically on controlled datasets.
"""

import numpy as np
import pytest

from histimator.channels import Channel
from histimator.data import Dataset
from histimator.histograms import Histogram
from histimator.likelihood import fit
from histimator.model import Model
from histimator.samples import Sample
from histimator.uncertainty import (
    bootstrap_histogram_variance,
    compare_uncertainties,
    poisson_variance,
    sumw2_variance,
)


# ── Poisson-bootstrap equivalence ────────────────────────────────────────

class TestPoissonBootstrapEquivalence:
    def test_unweighted_gaussian_sample(self):
        """For a large unweighted Gaussian sample, Poisson and bootstrap
        variances should agree per-bin within bootstrap sampling error."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 10000)
        edges = np.linspace(-3, 3, 21)
        ds = Dataset(data)

        h = ds.to_histogram(edges)
        var_poisson = poisson_variance(h)
        var_bootstrap = bootstrap_histogram_variance(ds, edges, n_bootstrap=2000, rng=rng)

        # Only test bins with substantial counts (n > 20) to avoid
        # small-sample effects where the Poisson approximation is less precise
        mask = h.values > 20
        np.testing.assert_allclose(
            var_bootstrap[mask], var_poisson[mask], rtol=0.30,
            err_msg="Bootstrap variance should approximate Poisson variance"
        )

    def test_unweighted_uniform_sample(self):
        """Uniform distribution: all bins have similar counts, so the
        Poisson approximation is uniform across bins."""
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 10, 5000)
        edges = np.linspace(0, 10, 11)
        ds = Dataset(data)

        h = ds.to_histogram(edges)
        var_poisson = poisson_variance(h)
        var_bootstrap = bootstrap_histogram_variance(ds, edges, n_bootstrap=2000, rng=rng)

        np.testing.assert_allclose(
            var_bootstrap, var_poisson, rtol=0.25,
            err_msg="All bins should show Poisson-bootstrap agreement"
        )

    def test_direction_of_finite_n_correction(self):
        """Bootstrap variance is n_j*(1 - n_j/n), which is strictly less
        than Poisson variance n_j.  The difference vanishes as n → ∞."""
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 1, 500)
        edges = np.array([0.0, 0.5, 1.0])
        ds = Dataset(data)
        h = ds.to_histogram(edges)

        # Theoretical bootstrap variance: n_j * (1 - n_j/n)
        n = len(data)
        var_theory = h.values * (1 - h.values / n)
        var_boot = bootstrap_histogram_variance(ds, edges, n_bootstrap=5000, rng=rng)
        np.testing.assert_allclose(var_boot, var_theory, rtol=0.15)


# ── sumw2-bootstrap equivalence ──────────────────────────────────────────

class TestSumw2BootstrapEquivalence:
    def test_weighted_sample(self):
        """For weighted events, bootstrap variance should converge to sumw2."""
        rng = np.random.default_rng(42)
        n = 5000
        data = rng.uniform(0, 10, n)
        weights = rng.exponential(2.0, n)
        edges = np.linspace(0, 10, 11)
        ds = Dataset(data, weights=weights)

        h = ds.to_histogram(edges)
        var_sumw2 = sumw2_variance(h)
        var_bootstrap = bootstrap_histogram_variance(ds, edges, n_bootstrap=2000, rng=rng)

        mask = h.values > 5  # require minimum statistics
        np.testing.assert_allclose(
            var_bootstrap[mask], var_sumw2[mask], rtol=0.35,
            err_msg="Bootstrap should converge to sumw2 for weighted events"
        )

    def test_unit_weights_reduce_to_poisson(self):
        """sumw2 with unit weights equals counts (Poisson variance)."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 1000)
        edges = np.linspace(-3, 3, 11)
        ds = Dataset(data, weights=np.ones(1000))
        h = ds.to_histogram(edges)

        np.testing.assert_allclose(sumw2_variance(h), poisson_variance(h))


# ── Scaling behaviour ────────────────────────────────────────────────────

class TestScaling:
    def test_convergence_with_many_bins(self):
        """With many bins (small p_j per bin), bootstrap converges to Poisson.

        The exact bootstrap variance is n*p_j*(1-p_j) = n_j*(1 - n_j/n).
        For k bins with uniform data, p_j = 1/k, so the correction factor
        is (1 - 1/k).  With k=50 bins, this is 0.98, i.e. 2% difference
        from Poisson, which is within bootstrap sampling error.
        """
        rng = np.random.default_rng(42)
        n = 10000
        data = rng.uniform(0, 10, n)
        edges = np.linspace(0, 10, 51)  # 50 bins, p_j ≈ 0.02
        ds = Dataset(data)

        h = ds.to_histogram(edges)
        var_poisson = poisson_variance(h)
        var_bootstrap = bootstrap_histogram_variance(ds, edges, n_bootstrap=2000, rng=rng)

        # With 50 bins, theoretical correction is (1 - 1/50) = 0.98
        # Bootstrap should be within 20% of Poisson for well-populated bins
        mask = h.values > 50
        np.testing.assert_allclose(var_bootstrap[mask], var_poisson[mask], rtol=0.20)

    def test_multinomial_correction_factor(self):
        """Verify the exact multinomial relationship:
        Var_bootstrap(n_j) = n_j * (1 - n_j/n).

        This is the fundamental identity connecting bootstrap to Poisson.
        The Poisson limit is recovered when n_j/n → 0.
        """
        rng = np.random.default_rng(42)
        n = 5000
        data = rng.uniform(0, 10, n)
        edges = np.linspace(0, 10, 11)  # 10 bins, p_j ≈ 0.1
        ds = Dataset(data)

        h = ds.to_histogram(edges)
        # Exact theoretical bootstrap variance
        var_theory = h.values * (1.0 - h.values / n)
        var_bootstrap = bootstrap_histogram_variance(ds, edges, n_bootstrap=3000, rng=rng)

        np.testing.assert_allclose(var_bootstrap, var_theory, rtol=0.15)

    def test_weighted_convergence(self):
        """Weighted bootstrap converges to sumw2 with many bins."""
        rng = np.random.default_rng(42)
        n = 5000
        data = rng.uniform(0, 10, n)
        weights = rng.exponential(1.0, n)
        edges = np.linspace(0, 10, 51)  # 50 bins for small p_j
        ds = Dataset(data, weights=weights)

        h = ds.to_histogram(edges)
        var_sw2 = sumw2_variance(h)
        var_boot = bootstrap_histogram_variance(ds, edges, n_bootstrap=2000, rng=rng)

        # For weighted data the same multinomial correction applies,
        # but with many bins (small p_j) it's negligible
        mask = h.values > 5
        np.testing.assert_allclose(var_boot[mask], var_sw2[mask], rtol=0.35)


# ── compare_uncertainties utility ────────────────────────────────────────

class TestCompareUncertainties:
    def test_returns_all_three(self):
        rng = np.random.default_rng(42)
        ds = Dataset(rng.normal(0, 1, 500))
        edges = np.linspace(-3, 3, 11)
        result = compare_uncertainties(ds, edges, n_bootstrap=100, rng=rng)
        assert set(result.keys()) == {"poisson", "sumw2", "bootstrap"}
        for v in result.values():
            assert v.shape == (10,)

    def test_shapes_consistent(self):
        rng = np.random.default_rng(42)
        ds = Dataset(rng.normal(0, 1, 500))
        edges = np.linspace(-3, 3, 7)
        result = compare_uncertainties(ds, edges, n_bootstrap=100, rng=rng)
        assert all(v.shape == (6,) for v in result.values())


# ── Integration test: the proof ──────────────────────────────────────────

class TestWithModifiers:
    def test_bootstrap_vs_sumw2_model_equivalence(self):
        """THE CRITICAL TEST: a model using bootstrap-estimated relative
        uncertainties as StatError must produce statistically consistent
        fit results with a model using sumw2-estimated uncertainties.

        This proves that future non-histogram representations can use
        bootstrap variance and get equivalent statistical behaviour to
        the histogram-based Barlow-Beeston treatment.
        """
        rng = np.random.default_rng(42)
        edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

        # Generate controlled data
        sig_events = rng.choice(np.arange(0.5, 5.0, 1.0), size=500,
                                p=[0.1, 0.3, 0.3, 0.2, 0.1])
        bkg_events = rng.uniform(0, 5, 3000)
        data_events = np.concatenate([sig_events, bkg_events])

        sig_ds = Dataset(sig_events)
        bkg_ds = Dataset(bkg_events)
        data_ds = Dataset(data_events)

        sig_h = sig_ds.to_histogram(edges)
        bkg_h = bkg_ds.to_histogram(edges)
        data_h = data_ds.to_histogram(edges)

        # Path A: sumw2-based uncertainties (Barlow-Beeston)
        rel_unc_sw2 = np.sqrt(bkg_h.sumw2) / np.maximum(bkg_h.values, 1e-10)

        sig_a = Sample("signal", sig_h)
        sig_a.add_normfactor("mu", nominal=1.0, bounds=(0.0, 5.0))
        bkg_a = Sample("background", bkg_h)
        bkg_a.add_staterror("mc_stat", rel_unc_sw2)
        ch_a = Channel("SR")
        ch_a.add_sample(sig_a)
        ch_a.add_sample(bkg_a)
        ch_a.set_data(data_h)
        m_a = Model("sumw2_model")
        m_a.add_channel(ch_a)
        r_a = fit(m_a)

        # Path B: bootstrap-based uncertainties
        var_boot = bootstrap_histogram_variance(
            bkg_ds, edges, n_bootstrap=2000, rng=np.random.default_rng(42)
        )
        rel_unc_boot = np.sqrt(var_boot) / np.maximum(bkg_h.values, 1e-10)

        sig_b = Sample("signal", sig_h)
        sig_b.add_normfactor("mu", nominal=1.0, bounds=(0.0, 5.0))
        bkg_b = Sample("background", bkg_h)
        bkg_b.add_staterror("mc_stat", rel_unc_boot)
        ch_b = Channel("SR")
        ch_b.add_sample(sig_b)
        ch_b.add_sample(bkg_b)
        ch_b.set_data(data_h)
        m_b = Model("bootstrap_model")
        m_b.add_channel(ch_b)
        r_b = fit(m_b)

        # The two approaches should give consistent mu estimates.
        # "Consistent" means the difference is small relative to the
        # statistical uncertainty on mu.
        assert r_a.valid and r_b.valid
        delta_mu = abs(r_a.bestfit["mu"] - r_b.bestfit["mu"])
        avg_error = 0.5 * (r_a.errors["mu"] + r_b.errors["mu"])
        assert delta_mu < 2.0 * avg_error, (
            f"sumw2 mu={r_a.bestfit['mu']:.4f}, bootstrap mu={r_b.bestfit['mu']:.4f}, "
            f"delta={delta_mu:.4f}, avg_error={avg_error:.4f}"
        )

    def test_uncertainty_errors_are_consistent(self):
        """The fitted error on mu should be similar whether using sumw2
        or bootstrap uncertainties for the background stat error."""
        rng = np.random.default_rng(99)
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        bkg_events = rng.uniform(0, 3, 5000)
        bkg_ds = Dataset(bkg_events)
        bkg_h = bkg_ds.to_histogram(edges)

        rel_sw2 = np.sqrt(bkg_h.sumw2) / np.maximum(bkg_h.values, 1e-10)
        var_boot = bootstrap_histogram_variance(
            bkg_ds, edges, n_bootstrap=2000, rng=rng)
        rel_boot = np.sqrt(var_boot) / np.maximum(bkg_h.values, 1e-10)

        # The relative uncertainties should agree within ~30%
        mask = bkg_h.values > 50
        np.testing.assert_allclose(rel_boot[mask], rel_sw2[mask], rtol=0.30)
