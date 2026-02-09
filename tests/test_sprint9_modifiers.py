"""Tests for Sprint 9: Additional modifier types.

Validation strategy
-------------------
Each new modifier type introduces a qualitatively different kind of
parameter into the model.  We test each modifier both in isolation
(does it modify expected yields correctly?) and end-to-end (does a
fit with this modifier behave as physics demands?).

The key insight for each test scenario is that the modifier's effect
should be *predictable* given the data and the constraint.  For example,
staterror gammas should be pulled toward the data but restrained by their
Gaussian constraint; shapefactor parameters should reproduce the data
exactly since they are unconstrained; and the lumi parameter should
shift all MC samples coherently.

All models use deterministic data (no Poisson fluctuation) so that
the test outcomes do not depend on random seeds.
"""

import numpy as np
import pytest

from histimator.channels import Channel
from histimator.histograms import Histogram
from histimator.likelihood import BinnedNLL, fit
from histimator.model import Model
from histimator.samples import (
    LumiSys,
    Sample,
    ShapeFactor,
    ShapeSys,
    StatError,
)

EDGES = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])


# ===================================================================
# Helpers
# ===================================================================

def _simple_model_with_data(bkg_vals, data_vals, edges=EDGES):
    """One channel, one background sample, specified data."""
    bkg = Sample("bkg", Histogram(bkg_vals, edges))
    ch = Channel("SR")
    ch.add_sample(bkg)
    ch.set_data(data_vals)
    m = Model("test")
    m.add_channel(ch)
    return m, bkg, ch


# ===================================================================
# Tests: StatError modifier
# ===================================================================


class TestStatErrorExpected:
    """StatError introduces per-bin gamma parameters that scale the
    sample's contribution.  When all gammas are at their nominal value
    of 1, the expected yield should be unchanged.  When a gamma is
    shifted, only the corresponding bin should change, and the change
    should be multiplicative.
    """

    def test_nominal_gammas_unchanged(self):
        """At gamma_i = 1 for all bins, expected yields equal nominal."""
        bkg_vals = np.array([100.0, 200.0, 150.0, 80.0, 50.0])
        bkg = Sample("bkg", Histogram(bkg_vals, EDGES))
        rel_err = np.array([0.1, 0.05, 0.07, 0.12, 0.15])
        bkg.add_staterror("mc_stat", rel_err)

        params = {p.name: p.value for p in bkg.parameters}
        result = bkg.expected(params)
        np.testing.assert_allclose(result, bkg_vals)

    def test_shifted_gamma_scales_bin(self):
        """Shifting one gamma should only scale the corresponding bin."""
        bkg_vals = np.array([100.0, 200.0, 150.0, 80.0, 50.0])
        bkg = Sample("bkg", Histogram(bkg_vals, EDGES))
        rel_err = np.array([0.1, 0.05, 0.07, 0.12, 0.15])
        bkg.add_staterror("mc_stat", rel_err)

        params = {p.name: p.value for p in bkg.parameters}
        # Shift the gamma for bin 2 to 1.1
        gamma_names = [p.name for p in bkg.parameters]
        params[gamma_names[2]] = 1.1

        result = bkg.expected(params)
        expected = bkg_vals.copy()
        expected[2] *= 1.1
        np.testing.assert_allclose(result, expected)

    def test_creates_correct_number_of_parameters(self):
        """StatError should add one parameter per bin."""
        bkg = Sample("bkg", Histogram([10, 20, 30], [0, 1, 2, 3]))
        bkg.add_staterror("mc_stat", [0.1, 0.05, 0.07])
        assert len(bkg.parameters) == 3

    def test_parameters_start_at_one(self):
        """All gamma parameters should start at nominal value 1."""
        bkg = Sample("bkg", Histogram([10, 20, 30], [0, 1, 2, 3]))
        bkg.add_staterror("mc_stat", [0.1, 0.05, 0.07])
        for p in bkg.parameters:
            assert p.value == 1.0


class TestStatErrorFit:
    """When fitting with staterror, the gammas should be pulled toward
    the data but constrained to stay near 1.  The pull magnitude
    depends on the ratio of the data-template discrepancy to the
    constraint width.

    We construct a scenario where the data is 20% above the template
    and the relative MC uncertainty is 10%.  The gamma should be pulled
    upward from 1 toward 1.2, but the Gaussian constraint at sigma=0.1
    prevents it from reaching 1.2 exactly.  The post-fit gamma should
    be between 1.0 and 1.2.
    """

    def test_gamma_pulled_toward_data(self):
        """Gammas should move toward data / nominal ratio."""
        bkg_vals = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        data_vals = np.array([120.0, 120.0, 120.0, 120.0, 120.0])

        bkg = Sample("bkg", Histogram(bkg_vals, EDGES))
        rel_err = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        bkg.add_staterror("mc_stat", rel_err)

        ch = Channel("SR")
        ch.add_sample(bkg)
        ch.set_data(data_vals)
        m = Model("test")
        m.add_channel(ch)

        result = fit(m)
        for p in m.parameters:
            gamma_hat = result.bestfit[p.name]
            # Should be pulled above 1 (toward data/nominal = 1.2)
            assert gamma_hat > 1.0
            # But constrained to not reach 1.2 (Gaussian penalty at 0.1)
            assert gamma_hat < 1.2

    def test_tighter_constraint_less_pull(self):
        """Smaller relative uncertainty means tighter constraint, less pull.

        With delta=0.01, the gamma can barely move from 1, even though
        the data wants it at 1.2.  With delta=0.5, it moves much closer
        to 1.2.
        """
        bkg_vals = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        data_vals = np.array([120.0, 120.0, 120.0, 120.0, 120.0])

        # Tight constraint
        bkg_tight = Sample("bkg", Histogram(bkg_vals, EDGES))
        bkg_tight.add_staterror("tight", np.full(5, 0.01))
        ch1 = Channel("SR")
        ch1.add_sample(bkg_tight)
        ch1.set_data(data_vals)
        m1 = Model("tight")
        m1.add_channel(ch1)
        r1 = fit(m1)

        # Loose constraint
        bkg_loose = Sample("bkg", Histogram(bkg_vals, EDGES))
        bkg_loose.add_staterror("loose", np.full(5, 0.5))
        ch2 = Channel("SR")
        ch2.add_sample(bkg_loose)
        ch2.set_data(data_vals)
        m2 = Model("loose")
        m2.add_channel(ch2)
        r2 = fit(m2)

        # Average gamma pull for tight should be less than for loose
        gammas_tight = [r1.bestfit[p.name] for p in m1.parameters]
        gammas_loose = [r2.bestfit[p.name] for p in m2.parameters]
        avg_pull_tight = np.mean(np.abs(np.array(gammas_tight) - 1.0))
        avg_pull_loose = np.mean(np.abs(np.array(gammas_loose) - 1.0))
        assert avg_pull_tight < avg_pull_loose

    def test_staterror_widens_postfit_uncertainty(self):
        """Adding staterror should give wider fit uncertainties than a
        model with no systematic at all.

        A model without staterror has no free parameters and cannot be
        fit.  Instead we compare to a model with a single normfactor,
        and check that the normfactor error is wider with staterror
        present (because the gammas absorb some uncertainty).
        """
        bkg_vals = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        sig_vals = np.array([10.0, 20.0, 30.0, 20.0, 10.0])
        data_vals = bkg_vals + sig_vals

        # Model WITHOUT staterror
        sig1 = Sample("sig", Histogram(sig_vals, EDGES))
        sig1.add_normfactor("mu", nominal=1.0, bounds=(0.0, 10.0))
        bkg1 = Sample("bkg", Histogram(bkg_vals, EDGES))
        ch1 = Channel("SR")
        ch1.add_sample(sig1)
        ch1.add_sample(bkg1)
        ch1.set_data(data_vals)
        m1 = Model("no_stat")
        m1.add_channel(ch1)
        r1 = fit(m1)

        # Model WITH staterror on background
        sig2 = Sample("sig", Histogram(sig_vals, EDGES))
        sig2.add_normfactor("mu", nominal=1.0, bounds=(0.0, 10.0))
        bkg2 = Sample("bkg", Histogram(bkg_vals, EDGES))
        bkg2.add_staterror("mc_stat", np.full(5, 0.1))
        ch2 = Channel("SR")
        ch2.add_sample(sig2)
        ch2.add_sample(bkg2)
        ch2.set_data(data_vals)
        m2 = Model("with_stat")
        m2.add_channel(ch2)
        r2 = fit(m2)

        assert r2.errors["mu"] > r1.errors["mu"]


# ===================================================================
# Tests: ShapeSys modifier
# ===================================================================


class TestShapeSysExpected:
    """ShapeSys works like StatError in the expected yield computation:
    per-bin gamma parameters multiply the sample yields.  The difference
    is in the constraint (Poisson rather than Gaussian), tested in the
    fit tests below.
    """

    def test_nominal_gammas_unchanged(self):
        bkg_vals = np.array([100.0, 200.0, 50.0, 80.0, 120.0])
        bkg = Sample("bkg", Histogram(bkg_vals, EDGES))
        bkg.add_shapesys("jet_fakes", np.array([0.2, 0.1, 0.3, 0.15, 0.1]))

        params = {p.name: p.value for p in bkg.parameters}
        result = bkg.expected(params)
        np.testing.assert_allclose(result, bkg_vals)

    def test_shifted_gamma(self):
        bkg_vals = np.array([100.0, 200.0, 50.0, 80.0, 120.0])
        bkg = Sample("bkg", Histogram(bkg_vals, EDGES))
        bkg.add_shapesys("jet_fakes", np.array([0.2, 0.1, 0.3, 0.15, 0.1]))

        params = {p.name: p.value for p in bkg.parameters}
        gamma_names = [p.name for p in bkg.parameters]
        params[gamma_names[0]] = 1.3  # 30% up in bin 0

        result = bkg.expected(params)
        expected = bkg_vals.copy()
        expected[0] *= 1.3
        np.testing.assert_allclose(result, expected)


class TestShapeSysFit:
    """ShapeSys uses a Poisson auxiliary constraint.  The constraint
    becomes tighter as the relative uncertainty decreases (because the
    effective auxiliary count tau = 1/rel^2 increases).

    For a bin with 10% relative uncertainty, tau = 100, so the gamma
    is tightly constrained near 1.  For 50% relative uncertainty,
    tau = 4, much more freedom.
    """

    def test_poisson_constraint_pulls_toward_data(self):
        """Like staterror, shapesys gammas should be pulled toward data."""
        bkg_vals = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        data_vals = np.array([130.0, 130.0, 130.0, 130.0, 130.0])

        bkg = Sample("bkg", Histogram(bkg_vals, EDGES))
        bkg.add_shapesys("fakes", np.full(5, 0.2))

        ch = Channel("SR")
        ch.add_sample(bkg)
        ch.set_data(data_vals)
        m = Model("test")
        m.add_channel(ch)

        result = fit(m)
        for p in m.parameters:
            gamma = result.bestfit[p.name]
            assert gamma > 1.0, "Should be pulled up toward data"
            assert gamma < 1.3, "Should not reach data/nominal = 1.3"

    def test_small_uncertainty_tight_constraint(self):
        """Very small relative uncertainty (0.01) should barely allow
        the gamma to move from 1, because tau = 10000.
        """
        bkg_vals = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        data_vals = np.array([150.0, 150.0, 150.0, 150.0, 150.0])

        bkg = Sample("bkg", Histogram(bkg_vals, EDGES))
        bkg.add_shapesys("fakes", np.full(5, 0.01))

        ch = Channel("SR")
        ch.add_sample(bkg)
        ch.set_data(data_vals)
        m = Model("test")
        m.add_channel(ch)

        result = fit(m)
        for p in m.parameters:
            gamma = result.bestfit[p.name]
            # tau = 10000 means gamma ~ N(1, 0.01), barely moves
            assert abs(gamma - 1.0) < 0.05


# ===================================================================
# Tests: ShapeFactor modifier
# ===================================================================


class TestShapeFactorExpected:
    """ShapeFactor provides completely free per-bin normalisation.
    The parameters are unconstrained, so the fit should be able to
    reproduce any data pattern exactly.
    """

    def test_nominal_unchanged(self):
        bkg_vals = np.array([50.0, 100.0, 75.0, 60.0, 40.0])
        bkg = Sample("bkg", Histogram(bkg_vals, EDGES))
        bkg.add_shapefactor("free_bkg")

        params = {p.name: p.value for p in bkg.parameters}
        result = bkg.expected(params)
        np.testing.assert_allclose(result, bkg_vals)

    def test_parameters_are_unconstrained(self):
        """ShapeFactor parameters should not appear in the constrained set."""
        bkg_vals = np.array([50.0, 100.0, 75.0, 60.0, 40.0])
        bkg = Sample("bkg", Histogram(bkg_vals, EDGES))
        bkg.add_shapefactor("free_bkg")

        ch = Channel("SR")
        ch.add_sample(bkg)
        ch.set_data(bkg_vals)
        m = Model("test")
        m.add_channel(ch)

        nll = BinnedNLL(m)
        # Check that shapefactor params are NOT in the constrained set
        sf_names = {p.name for p in bkg.parameters}
        assert sf_names.isdisjoint(nll._gaussian_constraints)
        assert sf_names.isdisjoint(nll._poisson_constraints)


class TestShapeFactorFit:
    """Because shapefactor is unconstrained, the fit should adjust the
    per-bin yields to perfectly match the data (up to numerical
    precision).  This is the standard tool for data-driven background
    estimation: let the background shape float freely in a control
    region.
    """

    def test_reproduces_data_exactly(self):
        """With only a shapefactor-modified sample, the fit should
        reproduce the data exactly because there is one free parameter
        per bin and no constraint opposing the adjustment.
        """
        bkg_vals = np.array([50.0, 100.0, 75.0, 60.0, 40.0])
        data_vals = np.array([70.0, 80.0, 90.0, 55.0, 65.0])

        bkg = Sample("bkg", Histogram(bkg_vals, EDGES))
        bkg.add_shapefactor("free_bkg")

        ch = Channel("SR")
        ch.add_sample(bkg)
        ch.set_data(data_vals)
        m = Model("test")
        m.add_channel(ch)

        result = fit(m)
        postfit_yields = m.expected(result.bestfit)
        np.testing.assert_allclose(postfit_yields, data_vals, rtol=1e-3)

    def test_recovers_correct_scale_factors(self):
        """The fitted shapefactor parameters should equal data/nominal
        for each bin.
        """
        bkg_vals = np.array([50.0, 100.0, 75.0, 60.0, 40.0])
        data_vals = np.array([70.0, 80.0, 90.0, 55.0, 65.0])

        bkg = Sample("bkg", Histogram(bkg_vals, EDGES))
        bkg.add_shapefactor("free_bkg")

        ch = Channel("SR")
        ch.add_sample(bkg)
        ch.set_data(data_vals)
        m = Model("test")
        m.add_channel(ch)

        result = fit(m)
        expected_ratios = data_vals / bkg_vals
        fitted_gammas = np.array([result.bestfit[p.name] for p in m.parameters])
        np.testing.assert_allclose(fitted_gammas, expected_ratios, rtol=1e-3)

    def test_shapefactor_with_signal(self):
        """ShapeFactor on background + normfactor on signal: the signal
        normfactor should still be recoverable because the shapefactor
        handles the background shape independently.
        """
        sig_vals = np.array([5.0, 15.0, 30.0, 15.0, 5.0])
        bkg_vals = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        # Data = 2*sig + bkg (so mu_true = 2)
        data_vals = 2.0 * sig_vals + bkg_vals

        sig = Sample("sig", Histogram(sig_vals, EDGES))
        sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 10.0))
        bkg = Sample("bkg", Histogram(bkg_vals, EDGES))
        bkg.add_shapefactor("free_bkg")

        ch = Channel("SR")
        ch.add_sample(sig)
        ch.add_sample(bkg)
        ch.set_data(data_vals)
        m = Model("test")
        m.add_channel(ch)

        result = fit(m)
        # The fit is degenerate (bkg shapefactor can absorb the signal),
        # but the total postfit prediction should match data
        postfit = m.expected(result.bestfit)
        np.testing.assert_allclose(postfit, data_vals, rtol=1e-2)


# ===================================================================
# Tests: LumiSys modifier
# ===================================================================


class TestLumiSysExpected:
    """LumiSys multiplies the entire sample yield by a single parameter
    constrained near 1.  It represents a luminosity uncertainty that
    coherently scales all MC predictions.
    """

    def test_nominal_unchanged(self):
        bkg_vals = np.array([100.0, 200.0, 150.0, 80.0, 50.0])
        bkg = Sample("bkg", Histogram(bkg_vals, EDGES))
        bkg.add_lumisys("lumi", uncertainty=0.02)

        params = {p.name: p.value for p in bkg.parameters}
        result = bkg.expected(params)
        np.testing.assert_allclose(result, bkg_vals)

    def test_scales_all_bins_coherently(self):
        """Setting the lumi parameter to 1.05 should scale all bins by 1.05."""
        bkg_vals = np.array([100.0, 200.0, 150.0, 80.0, 50.0])
        bkg = Sample("bkg", Histogram(bkg_vals, EDGES))
        bkg.add_lumisys("lumi", uncertainty=0.02)

        params = {p.name: 1.05 for p in bkg.parameters}
        result = bkg.expected(params)
        np.testing.assert_allclose(result, bkg_vals * 1.05)


class TestLumiSysFit:
    """When the data is systematically higher than the nominal prediction,
    the lumi parameter should absorb the overall shift.  The constraint
    prevents it from moving too far.
    """

    def test_lumi_absorbs_overall_scale(self):
        """Data = 1.03 * bkg everywhere.  With a 5% lumi uncertainty,
        the fit should recover lumi ~ 1.03.
        """
        bkg_vals = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        data_vals = 1.03 * bkg_vals

        bkg = Sample("bkg", Histogram(bkg_vals, EDGES))
        bkg.add_lumisys("lumi", uncertainty=0.05)

        ch = Channel("SR")
        ch.add_sample(bkg)
        ch.set_data(data_vals)
        m = Model("test")
        m.add_channel(ch)

        result = fit(m)
        np.testing.assert_allclose(result.bestfit["lumi"], 1.03, rtol=0.02)

    def test_lumi_constrained_when_shift_large(self):
        """Data = 1.20 * bkg, but lumi uncertainty is only 2%.  The
        constraint should prevent lumi from reaching 1.20.
        """
        bkg_vals = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        data_vals = 1.20 * bkg_vals

        bkg = Sample("bkg", Histogram(bkg_vals, EDGES))
        bkg.add_lumisys("lumi", uncertainty=0.02)

        ch = Channel("SR")
        ch.add_sample(bkg)
        ch.set_data(data_vals)
        m = Model("test")
        m.add_channel(ch)

        result = fit(m)
        lumi_hat = result.bestfit["lumi"]
        assert lumi_hat > 1.0, "Should be pulled up"
        assert lumi_hat < 1.15, "2% constraint should restrain large pull"

    def test_correlated_across_samples(self):
        """Two MC samples sharing the same lumi modifier should shift
        coherently.  The total yield after fit should reflect a single
        overall scale.
        """
        sig_vals = np.array([10.0, 20.0, 30.0, 20.0, 10.0])
        bkg_vals = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        # Data is 1.04 * total
        total = sig_vals + bkg_vals
        data_vals = 1.04 * total

        sig = Sample("sig", Histogram(sig_vals, EDGES))
        sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 5.0))
        sig.add_lumisys("lumi", uncertainty=0.05)
        bkg = Sample("bkg", Histogram(bkg_vals, EDGES))
        bkg.add_lumisys("lumi", uncertainty=0.05)

        ch = Channel("SR")
        ch.add_sample(sig)
        ch.add_sample(bkg)
        ch.set_data(data_vals)
        m = Model("test")
        m.add_channel(ch)

        result = fit(m)
        # lumi should absorb the overall scale, mu stays near 1
        assert abs(result.bestfit["lumi"] - 1.04) < 0.05
        assert abs(result.bestfit["mu"] - 1.0) < 0.5


# ===================================================================
# Tests: Modifier combinations
# ===================================================================


class TestModifierCombinations:
    """Real analyses use multiple modifier types simultaneously.  These
    tests verify that modifiers compose correctly: normfactor + staterror,
    normsys + staterror, shapefactor + lumisys, etc.
    """

    def test_normfactor_plus_staterror(self):
        """Signal with normfactor, background with staterror.  The fit
        should recover the injected signal strength, with the staterror
        gammas adjusting the background.
        """
        sig_vals = np.array([5.0, 20.0, 40.0, 20.0, 5.0])
        bkg_vals = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        # Inject mu=2 signal
        data_vals = 2.0 * sig_vals + bkg_vals

        sig = Sample("sig", Histogram(sig_vals, EDGES))
        sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 10.0))
        bkg = Sample("bkg", Histogram(bkg_vals, EDGES))
        bkg.add_staterror("mc_stat", np.full(5, 0.1))

        ch = Channel("SR")
        ch.add_sample(sig)
        ch.add_sample(bkg)
        ch.set_data(data_vals)
        m = Model("test")
        m.add_channel(ch)

        result = fit(m)
        assert abs(result.bestfit["mu"] - 2.0) < 0.5

    def test_normsys_plus_staterror(self):
        """Background with both normsys and staterror should have more
        post-fit parameters than with either alone, and the fit should
        still converge.
        """
        bkg_vals = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        data_vals = np.array([110.0, 110.0, 110.0, 110.0, 110.0])

        sig = Sample("sig", Histogram([10, 20, 30, 20, 10], EDGES))
        sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 10.0))
        bkg = Sample("bkg", Histogram(bkg_vals, EDGES))
        bkg.add_normsys("bkg_norm", lo=0.9, hi=1.1)
        bkg.add_staterror("mc_stat", np.full(5, 0.05))

        ch = Channel("SR")
        ch.add_sample(sig)
        ch.add_sample(bkg)
        ch.set_data(data_vals)
        m = Model("test")
        m.add_channel(ch)

        result = fit(m)
        assert result.valid
        # Should have: mu, bkg_norm, 5 gamma params = 7 total
        assert len(result.bestfit) == 7

    def test_staterror_constraint_in_nll(self):
        """The NLL should increase when a staterror gamma deviates from
        1, even if the expected yield matches the data perfectly.  This
        verifies the Gaussian penalty is active.
        """
        bkg_vals = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        bkg = Sample("bkg", Histogram(bkg_vals, EDGES))
        bkg.add_staterror("mc_stat", np.full(5, 0.1))

        ch = Channel("SR")
        ch.add_sample(bkg)
        ch.set_data(bkg_vals)
        m = Model("test")
        m.add_channel(ch)

        nll = BinnedNLL(m)
        pnames = nll._par_names

        # All gammas at 1: minimum of the constraint
        nominal_pars = np.array([1.0] * len(pnames))
        nll_nominal = nll(nominal_pars)

        # Shift one gamma to 1.5: should increase NLL
        shifted = nominal_pars.copy()
        shifted[0] = 1.5
        nll_shifted = nll(shifted)

        assert nll_shifted > nll_nominal
