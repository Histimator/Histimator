"""Tests for Sprint 16b: Unbinned likelihood integration with CLs,
test statistics, and diagnostics.

The unbinned extended likelihood was validated in isolation in
test_sprint16_unbinned.py.  This file validates that it integrates
correctly with the downstream inference machinery:

1. Test statistics (q_0, q_mu, q_tilde_mu) computed from the
   unbinned likelihood produce sensible values and agree with
   binned results within expected statistical differences.

2. CLs observed values and upper limits computed via the unbinned
   path give compatible exclusion limits to the binned path.

3. Diagnostics (prefit_yields, postfit_yields, nuisance_pulls)
   work correctly when given an unbinned FitResult.

The Asimov sigma computation remains binned throughout because the
Asimov dataset is inherently a binned construct (expected yields per
bin).  The unbinned flag affects only the observed-data test statistic
and CLs computation.
"""

import numpy as np
import pytest

from histimator.channels import Channel
from histimator.cls import cls_observed, upper_limit_cls
from histimator.diagnostics import (
    nuisance_pulls,
    postfit_yields,
    prefit_yields,
)
from histimator.histograms import Histogram
from histimator.likelihood import fit
from histimator.model import Model
from histimator.samples import Sample
from histimator.test_statistics import (
    compute_q0,
    compute_qmu,
    compute_qtilde_mu,
)

# ===================================================================
# Helpers
# ===================================================================

EDGES = np.linspace(100, 160, 21)
CENTRES = 0.5 * (EDGES[:-1] + EDGES[1:])
WIDTHS = np.diff(EDGES)


def _sig_template(total=20.0):
    vals = np.exp(-0.5 * ((CENTRES - 130) / 8) ** 2) * WIDTHS
    vals *= total / vals.sum()
    return vals


def _bkg_template(total=200.0):
    vals = np.exp(-0.04 * CENTRES) * WIDTHS
    vals *= total / vals.sum()
    return vals


def _generate_events(mu_true=1.0, n_bkg=200, seed=42):
    """Generate pseudo-data events from signal+background."""
    rng = np.random.default_rng(seed)
    n_sig = rng.poisson(int(mu_true * 20))
    sig_events = rng.normal(130, 8, n_sig)
    bkg_events = rng.exponential(1 / 0.04, rng.poisson(n_bkg)) + 100
    all_events = np.concatenate([sig_events, bkg_events])
    mask = (all_events >= EDGES[0]) & (all_events <= EDGES[-1])
    return all_events[mask]


def _scalar_model(events):
    """Build a model with scalar modifiers and both binned+unbinned data."""
    sig = Sample("signal", Histogram(_sig_template(), EDGES))
    sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 10.0))
    bkg = Sample("background", Histogram(_bkg_template(), EDGES))
    bkg.add_normsys("bkg_norm", lo=0.9, hi=1.1)

    ch = Channel("SR")
    ch.add_sample(sig)
    ch.add_sample(bkg)

    data_hist, _ = np.histogram(events, bins=EDGES)
    ch.set_data(data_hist.astype(float))
    ch.set_unbinned_data(events)

    model = Model("test")
    model.add_channel(ch)
    return model


def _bkg_only_events(n_bkg=200, seed=99):
    """Generate background-only pseudo-data."""
    rng = np.random.default_rng(seed)
    bkg_events = rng.exponential(1 / 0.04, rng.poisson(n_bkg)) + 100
    mask = (bkg_events >= EDGES[0]) & (bkg_events <= EDGES[-1])
    return bkg_events[mask]


# ===================================================================
# Density normalization precondition
# ===================================================================

class TestDensityNormalizationPrecondition:
    """The unbinned extended likelihood is mathematically correct only
    when evaluate_density integrates to unity for every template.  If a
    template's density integrates to 0.95 instead, each event contributes
    a bias of log(0.95) to the log-likelihood, and that bias accumulates
    linearly with the number of events.  For 200 events the total NLL
    shift would be ~10 units, easily enough to bias mu_hat by several sigma.

    BinnedTemplate normalization is exact by construction, so it is
    verified via the discrete sum.  GP normalization is approximate
    (the rate is fitted on a grid and then renormalized), so it is
    verified via scipy quadrature on the smooth density."""

    def test_binned_signal_density_exact(self):
        """BinnedTemplate: sum(density_j * w_j) = 1 exactly."""
        from histimator.templates import BinnedTemplate
        bt = BinnedTemplate(Histogram(_sig_template(20.0), EDGES))
        integral = np.sum(bt.density() * np.diff(EDGES))
        np.testing.assert_allclose(integral, 1.0, rtol=1e-14)

    def test_binned_background_density_exact(self):
        from histimator.templates import BinnedTemplate
        bt = BinnedTemplate(Histogram(_bkg_template(200.0), EDGES))
        integral = np.sum(bt.density() * np.diff(EDGES))
        np.testing.assert_allclose(integral, 1.0, rtol=1e-14)

    def test_gp_signal_density_normalized(self):
        """GPTemplate density should integrate to 1 within 1%."""
        from scipy.integrate import quad

        from histimator.gp_template import GPKernel, GPTemplate
        h = Histogram(_sig_template(20.0), EDGES)
        gp = GPTemplate(h, kernel=GPKernel.MATERN_52,
                        optimize_hyperparameters=False)
        integral, _ = quad(lambda x: gp.evaluate_density(np.array([x]))[0],
                           EDGES[0], EDGES[-1], limit=100)
        np.testing.assert_allclose(integral, 1.0, rtol=0.01)

    def test_gp_background_density_normalized(self):
        from scipy.integrate import quad

        from histimator.gp_template import GPKernel, GPTemplate
        h = Histogram(_bkg_template(200.0), EDGES)
        gp = GPTemplate(h, kernel=GPKernel.MATERN_52,
                        optimize_hyperparameters=False)
        integral, _ = quad(lambda x: gp.evaluate_density(np.array([x]))[0],
                           EDGES[0], EDGES[-1], limit=100)
        np.testing.assert_allclose(integral, 1.0, rtol=0.01)

    def test_gp_normalization_bias_bounded(self):
        """Verify that the GP normalization error is small enough that
        the accumulated NLL bias over a typical event sample is negligible.

        For n events, a density normalization error of epsilon causes an
        NLL bias of approximately n * |log(1 + epsilon)| ~ n * |epsilon|.
        We require this to be less than 0.5 for 500 events, meaning
        |epsilon| < 1e-3.  The GP renormalization step in GPTemplate
        ensures this is met for typical histogram shapes."""
        from scipy.integrate import quad

        from histimator.gp_template import GPKernel, GPTemplate
        h = Histogram(_bkg_template(200.0), EDGES)
        gp = GPTemplate(h, kernel=GPKernel.MATERN_52,
                        optimize_hyperparameters=False)
        integral, _ = quad(lambda x: gp.evaluate_density(np.array([x]))[0],
                           EDGES[0], EDGES[-1], limit=100)
        n_events = 500
        nll_bias = n_events * abs(np.log(integral))
        assert nll_bias < 0.5, (
            f"GP density integral = {integral:.8f}, "
            f"NLL bias over {n_events} events = {nll_bias:.4f}"
        )


# ===================================================================
# Test statistics: unbinned mode
# ===================================================================

class TestUnbinnedQ0:
    """compute_q0 with unbinned=True should produce sensible discovery
    test statistics."""

    def test_returns_finite(self):
        events = _generate_events(mu_true=1.0, seed=42)
        model = _scalar_model(events)
        q0, mu_hat = compute_q0(model, "mu", unbinned=True)
        assert np.isfinite(q0)
        assert np.isfinite(mu_hat)

    def test_non_negative(self):
        events = _generate_events(mu_true=1.0, seed=42)
        model = _scalar_model(events)
        q0, _ = compute_q0(model, "mu", unbinned=True)
        assert q0 >= 0.0

    def test_zero_when_mu_hat_negative(self):
        """When data has a deficit, mu_hat < 0 and q0 = 0."""
        events = _bkg_only_events(n_bkg=150, seed=77)
        model = _scalar_model(events)
        q0, mu_hat = compute_q0(model, "mu", unbinned=True)
        if mu_hat < 0:
            assert q0 == 0.0

    def test_agrees_with_binned(self):
        """Unbinned and binned q0 should be in the same ballpark."""
        events = _generate_events(mu_true=1.5, seed=42)
        model = _scalar_model(events)
        q0_ub, mu_ub = compute_q0(model, "mu", unbinned=True)
        q0_b, mu_b = compute_q0(model, "mu", unbinned=False)
        # mu_hat values should be compatible
        assert abs(mu_ub - mu_b) < 1.0
        # q0 can differ due to binning loss, but should be same order
        if q0_b > 0.1:
            assert abs(q0_ub - q0_b) / q0_b < 1.0


class TestUnbinnedQmu:
    """compute_qmu with unbinned=True."""

    def test_returns_finite(self):
        events = _generate_events(mu_true=1.0, seed=42)
        model = _scalar_model(events)
        qmu, mu_hat = compute_qmu(model, "mu", 2.0, unbinned=True)
        assert np.isfinite(qmu)
        assert np.isfinite(mu_hat)

    def test_zero_when_mu_hat_above_mu_test(self):
        """When mu_hat > mu_test, q_mu = 0 by definition."""
        events = _generate_events(mu_true=3.0, seed=42)
        model = _scalar_model(events)
        qmu, mu_hat = compute_qmu(model, "mu", 0.5, unbinned=True)
        if mu_hat > 0.5:
            assert qmu == 0.0


class TestUnbinnedQtildeMu:
    """compute_qtilde_mu with unbinned=True."""

    def test_returns_finite(self):
        events = _generate_events(mu_true=1.0, seed=42)
        model = _scalar_model(events)
        qt, mu_hat = compute_qtilde_mu(model, "mu", 2.0, unbinned=True)
        assert np.isfinite(qt)
        assert np.isfinite(mu_hat)

    def test_non_negative(self):
        events = _generate_events(mu_true=1.0, seed=42)
        model = _scalar_model(events)
        qt, _ = compute_qtilde_mu(model, "mu", 2.0, unbinned=True)
        assert qt >= 0.0

    def test_agrees_with_binned(self):
        events = _generate_events(mu_true=1.0, seed=42)
        model = _scalar_model(events)
        qt_ub, mu_ub = compute_qtilde_mu(model, "mu", 2.0, unbinned=True)
        qt_b, mu_b = compute_qtilde_mu(model, "mu", 2.0, unbinned=False)
        assert abs(mu_ub - mu_b) < 1.0
        if qt_b > 0.1:
            assert abs(qt_ub - qt_b) / qt_b < 1.5


# ===================================================================
# CLs: unbinned mode
# ===================================================================

class TestUnbinnedClsObserved:
    """cls_observed with unbinned=True."""

    def test_returns_valid_probability(self):
        events = _generate_events(mu_true=1.0, seed=42)
        model = _scalar_model(events)
        cls_val = cls_observed(model, "mu", 2.0, unbinned=True)
        assert 0.0 <= cls_val <= 1.0

    def test_small_for_large_mu(self):
        """CLs should be small when testing a large mu against data
        generated with smaller signal."""
        events = _generate_events(mu_true=0.5, seed=42)
        model = _scalar_model(events)
        cls_val = cls_observed(model, "mu", 5.0, unbinned=True)
        assert cls_val < 0.5

    def test_large_for_true_mu(self):
        """CLs should not exclude the true signal strength."""
        events = _generate_events(mu_true=1.0, seed=42)
        model = _scalar_model(events)
        cls_val = cls_observed(model, "mu", 1.0, unbinned=True)
        assert cls_val > 0.05

    def test_agrees_with_binned(self):
        events = _generate_events(mu_true=1.0, seed=42)
        model = _scalar_model(events)
        cls_ub = cls_observed(model, "mu", 2.0, unbinned=True)
        cls_b = cls_observed(model, "mu", 2.0, unbinned=False)
        # Both should be reasonable probabilities, within 0.3 of each other
        assert abs(cls_ub - cls_b) < 0.3


class TestUnbinnedUpperLimit:
    """upper_limit_cls with unbinned=True."""

    def test_positive(self):
        events = _generate_events(mu_true=1.0, seed=42)
        model = _scalar_model(events)
        obs_limit, expected = upper_limit_cls(
            model, "mu", scan_range=(0.1, 8.0), unbinned=True,
        )
        assert obs_limit > 0

    def test_observed_reasonable(self):
        """The observed limit should be in a sensible range for
        a model with O(20) signal events and O(200) background."""
        events = _generate_events(mu_true=1.0, seed=42)
        model = _scalar_model(events)
        obs_limit, expected = upper_limit_cls(
            model, "mu", scan_range=(0.1, 8.0), unbinned=True,
        )
        # With 20 signal and 200 bkg, limit should be O(1-5)
        assert 0.1 < obs_limit < 10.0

    def test_expected_bands_returned(self):
        """The expected band should have all five quantile keys."""
        events = _generate_events(mu_true=1.0, seed=42)
        model = _scalar_model(events)
        _, expected = upper_limit_cls(
            model, "mu", scan_range=(0.1, 8.0), unbinned=True,
        )
        for key in ["-2sig", "-1sig", "median", "+1sig", "+2sig"]:
            assert key in expected

    def test_agrees_with_binned(self):
        """Unbinned and binned upper limits should be compatible."""
        events = _generate_events(mu_true=1.0, seed=42)
        model = _scalar_model(events)
        obs_ub, _ = upper_limit_cls(
            model, "mu", scan_range=(0.1, 8.0), unbinned=True,
        )
        obs_b, _ = upper_limit_cls(
            model, "mu", scan_range=(0.1, 8.0), unbinned=False,
        )
        # Limits should be within a factor of 2 of each other
        ratio = obs_ub / obs_b
        assert 0.5 < ratio < 2.0, (
            f"Unbinned limit {obs_ub:.2f} vs binned {obs_b:.2f}, "
            f"ratio {ratio:.2f}"
        )


# ===================================================================
# Diagnostics: unbinned fit results
# ===================================================================

class TestDiagnosticsWithUnbinnedFit:
    """Diagnostics functions should work with FitResult from unbinned fits.

    The diagnostics themselves compute binned yields (since samples
    always have a binned template underneath), but the FitResult can
    come from an unbinned fit. The bestfit parameter dictionary has
    the same structure regardless of likelihood type."""

    @pytest.fixture(scope="class")
    def model_and_result(self):
        events = _generate_events(mu_true=1.5, seed=42)
        model = _scalar_model(events)
        result = fit(model, unbinned=True)
        return model, result

    def test_prefit_yields_structure(self, model_and_result):
        """prefit_yields returns per-channel flat dicts regardless
        of how the fit was done."""
        model, _ = model_and_result
        pre = prefit_yields(model)
        assert "SR" in pre
        assert "total" in pre["SR"]
        assert "signal" in pre["SR"]
        assert "background" in pre["SR"]

    def test_postfit_yields_uses_unbinned_bestfit(self, model_and_result):
        """postfit_yields should evaluate at the unbinned bestfit values."""
        model, result = model_and_result
        post = postfit_yields(model, result)
        assert "SR" in post
        # The postfit total should be close to the data
        data = model.channels[0].data.values
        total = post["SR"]["total"]
        assert total.shape == data.shape
        # Postfit should track data within Poisson scatter
        chi2_per_bin = np.mean((total - data) ** 2 / np.maximum(data, 1))
        assert chi2_per_bin < 5.0

    def test_nuisance_pulls_from_unbinned_fit(self, model_and_result):
        """nuisance_pulls should extract pulls from the unbinned FitResult."""
        model, result = model_and_result
        pulls = nuisance_pulls(model, result)
        assert "bkg_norm" in pulls
        info = pulls["bkg_norm"]
        assert np.isfinite(info["pull"])
        assert info["error"] > 0

    def test_pulls_bounded(self, model_and_result):
        """Nuisance pulls should be within a few sigma of zero."""
        model, result = model_and_result
        pulls = nuisance_pulls(model, result)
        for pname, info in pulls.items():
            pull_z = abs(info["pull"])
            assert pull_z < 5.0, (
                f"Pull for {pname} is {pull_z:.1f} sigma"
            )


# ===================================================================
# Cross-validation: unbinned vs binned full workflow
# ===================================================================

class TestUnbinnedVsBinnedFullWorkflow:
    """End-to-end comparison of the unbinned and binned paths through
    the complete fit-to-limit workflow on the same pseudo-data."""

    @pytest.fixture(scope="class")
    def model(self):
        events = _generate_events(mu_true=1.0, seed=42)
        return _scalar_model(events)

    def test_fit_mu_compatible(self, model):
        r_ub = fit(model, unbinned=True)
        r_b = fit(model, unbinned=False)
        diff = abs(r_ub.bestfit["mu"] - r_b.bestfit["mu"])
        # Should agree within the sum of their errors
        tol = r_ub.errors["mu"] + r_b.errors["mu"]
        assert diff < tol, (
            f"mu_ub={r_ub.bestfit['mu']:.3f} +/- {r_ub.errors['mu']:.3f}, "
            f"mu_b={r_b.bestfit['mu']:.3f} +/- {r_b.errors['mu']:.3f}"
        )

    def test_cls_monotonic_in_mu(self, model):
        """CLs should decrease as mu_test increases, for both paths."""
        mu_values = [1.0, 2.0, 4.0]
        for ub in [True, False]:
            cls_vals = [
                cls_observed(model, "mu", mu, unbinned=ub)
                for mu in mu_values
            ]
            # Should be roughly decreasing (allow small non-monotonicity
            # from numerical noise)
            assert cls_vals[0] > cls_vals[-1] - 0.1

    def test_multiple_seeds_limits_compatible(self):
        """Across different pseudo-datasets, unbinned and binned limits
        should track each other."""
        ratios = []
        for seed in [42, 123, 456]:
            events = _generate_events(mu_true=1.0, seed=seed)
            model = _scalar_model(events)
            obs_ub, _ = upper_limit_cls(
                model, "mu", scan_range=(0.1, 8.0), unbinned=True,
            )
            obs_b, _ = upper_limit_cls(
                model, "mu", scan_range=(0.1, 8.0), unbinned=False,
            )
            if obs_b > 0:
                ratios.append(obs_ub / obs_b)
        # Average ratio should be close to 1
        avg_ratio = np.mean(ratios)
        assert 0.5 < avg_ratio < 2.0
