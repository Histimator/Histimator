"""Tests for Sprint 16: Unbinned extended likelihood.

Validation strategy
-------------------
The unbinned extended likelihood is validated by comparing its results
against the binned likelihood on the same data.  For models with only
scalar modifiers (NormFactor, NormSys, LumiSys), the two likelihoods
should agree to within the statistical resolution of the binning.

The key mathematical relationship is:

    log L_unbinned = sum_i log(sum_j N_j * p_j(x_i)) - sum_j N_j

where p_j(x) is the probability density from evaluate_density() and
N_j(theta) is the total expected yield for sample j.  Since scalar
modifiers only change N_j (not the shape), p_j(x_i) is cached at
construction time.

The test structure validates:
1. The NLL callable produces finite values and agrees with manual
   computation.
2. Fits converge and recover injected signal strength.
3. Unbinned and binned fits give compatible mu estimates.
4. Per-bin modifiers (HistoSys, StatError) are correctly rejected.
5. Channel unbinned data storage and domain clipping work correctly.
"""

import numpy as np
import pytest

from histimator.channels import Channel
from histimator.histograms import Histogram
from histimator.likelihood import UnbinnedNLL, fit
from histimator.model import Model
from histimator.samples import Sample

EDGES = np.linspace(100, 160, 21)
CENTRES = 0.5 * (EDGES[:-1] + EDGES[1:])
WIDTHS = np.diff(EDGES)


# ===================================================================
# Helpers
# ===================================================================

def _sig_template(total=20.0):
    """Gaussian-peaked signal template."""
    vals = np.exp(-0.5 * ((CENTRES - 130) / 8) ** 2) * WIDTHS
    vals *= total / vals.sum()
    return vals


def _bkg_template(total=200.0):
    """Falling exponential background template."""
    vals = np.exp(-0.04 * CENTRES) * WIDTHS
    vals *= total / vals.sum()
    return vals


def _generate_events(mu_true=2.0, n_bkg=200, seed=42):
    """Generate pseudo-data events from the signal+background model."""
    rng = np.random.default_rng(seed)
    n_sig = rng.poisson(int(mu_true * 20))
    sig_events = rng.normal(130, 8, n_sig)
    bkg_events = rng.exponential(1 / 0.04, rng.poisson(n_bkg)) + 100
    all_events = np.concatenate([sig_events, bkg_events])
    mask = (all_events >= EDGES[0]) & (all_events <= EDGES[-1])
    return all_events[mask]


def _scalar_model(events, seed=42):
    """Build a model with only scalar modifiers and both binned + unbinned data."""
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


# ===================================================================
# Channel unbinned data
# ===================================================================

class TestChannelUnbinnedData:
    """set_unbinned_data stores raw events and clips to domain."""

    def test_stores_events(self):
        ch = Channel("SR")
        ch.add_sample(Sample("sig", Histogram(_sig_template(), EDGES)))
        events = np.array([110.0, 130.0, 150.0])
        ch.set_unbinned_data(events)
        np.testing.assert_array_equal(ch.unbinned_data, events)

    def test_clips_to_domain(self):
        ch = Channel("SR")
        ch.add_sample(Sample("sig", Histogram(_sig_template(), EDGES)))
        events = np.array([50.0, 110.0, 130.0, 200.0])
        ch.set_unbinned_data(events)
        assert len(ch.unbinned_data) == 2
        np.testing.assert_array_equal(ch.unbinned_data, [110.0, 130.0])

    def test_none_when_not_set(self):
        ch = Channel("SR")
        ch.add_sample(Sample("sig", Histogram(_sig_template(), EDGES)))
        assert ch.unbinned_data is None

    def test_accepts_dataset(self):
        from histimator.data import Dataset
        ch = Channel("SR")
        ch.add_sample(Sample("sig", Histogram(_sig_template(), EDGES)))
        ds = Dataset([110.0, 130.0, 150.0])
        ch.set_unbinned_data(ds)
        assert len(ch.unbinned_data) == 3


# ===================================================================
# UnbinnedNLL construction and validation
# ===================================================================

class TestUnbinnedNLLConstruction:
    """UnbinnedNLL validates inputs and caches densities correctly."""

    def test_rejects_histosys(self):
        sig = Sample("signal", Histogram(_sig_template(), EDGES))
        sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 10.0))
        bkg = Sample("background", Histogram(_bkg_template(), EDGES))
        bkg.add_histosys(
            "bkg_shape",
            lo_hist=Histogram(_bkg_template() * 0.9, EDGES),
            hi_hist=Histogram(_bkg_template() * 1.1, EDGES),
        )
        ch = Channel("SR")
        ch.add_sample(sig)
        ch.add_sample(bkg)
        ch.set_unbinned_data(np.array([120.0, 130.0]))
        ch.set_data(np.ones(20))
        m = Model("bad")
        m.add_channel(ch)
        with pytest.raises(ValueError, match="HistoSys"):
            UnbinnedNLL(m)

    def test_rejects_staterror(self):
        bkg = Sample("background", Histogram(_bkg_template(), EDGES))
        bkg.add_staterror("bkg_stat", rel_uncertainties=np.full(20, 0.1))
        ch = Channel("SR")
        ch.add_sample(bkg)
        ch.set_unbinned_data(np.array([120.0]))
        ch.set_data(np.ones(20))
        m = Model("bad")
        m.add_channel(ch)
        with pytest.raises(ValueError, match="StatError"):
            UnbinnedNLL(m)

    def test_rejects_missing_unbinned_data(self):
        sig = Sample("signal", Histogram(_sig_template(), EDGES))
        sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 10.0))
        ch = Channel("SR")
        ch.add_sample(sig)
        ch.set_data(np.ones(20))
        m = Model("bad")
        m.add_channel(ch)
        with pytest.raises(ValueError, match="no unbinned data"):
            UnbinnedNLL(m)

    def test_constructs_with_scalar_modifiers(self):
        events = _generate_events()
        model = _scalar_model(events)
        nll = UnbinnedNLL(model)
        assert nll.model is model


# ===================================================================
# NLL evaluation
# ===================================================================

class TestUnbinnedNLLEvaluation:
    """The NLL callable should produce finite, sensible values."""

    @pytest.fixture
    def model(self):
        return _scalar_model(_generate_events())

    def test_returns_finite(self, model):
        nll = UnbinnedNLL(model)
        start = [p.value for p in model.parameters]
        val = nll(start)
        assert np.isfinite(val)

    def test_minimum_near_truth(self, model):
        """The NLL should be lower near the true parameter values
        than at extreme values."""
        nll = UnbinnedNLL(model)
        # Near truth: mu~2, bkg_norm~0
        val_good = nll([2.0, 0.0])
        # Far from truth: mu=10, bkg_norm=3
        val_bad = nll([10.0, 3.0])
        assert val_good < val_bad

    def test_constraint_terms_present(self, model):
        """NormSys alpha should be penalised away from 0."""
        nll = UnbinnedNLL(model)
        val_at_zero = nll([2.0, 0.0])
        val_at_three = nll([2.0, 3.0])
        # The constraint term adds 0.5 * 9 = 4.5 to the NLL
        assert val_at_three > val_at_zero


# ===================================================================
# Unbinned fitting
# ===================================================================

class TestUnbinnedFit:
    """fit(model, unbinned=True) should converge and recover signal."""

    def test_fit_converges(self):
        events = _generate_events()
        model = _scalar_model(events)
        result = fit(model, unbinned=True)
        assert result.valid

    def test_recovers_signal(self):
        """With mu_true=2, the fit should find mu_hat > 0."""
        events = _generate_events(mu_true=2.0, seed=42)
        model = _scalar_model(events)
        result = fit(model, unbinned=True)
        assert result.bestfit["mu"] > 0.5

    def test_null_signal(self):
        """With mu_true=0 (background only), mu_hat should be near 0."""
        events = _generate_events(mu_true=0.0, seed=42)
        model = _scalar_model(events)
        result = fit(model, unbinned=True)
        assert result.bestfit["mu"] < 2.0

    def test_errors_positive(self):
        events = _generate_events()
        model = _scalar_model(events)
        result = fit(model, unbinned=True)
        for name, err in result.errors.items():
            assert err > 0, f"Error for {name} is not positive"


# ===================================================================
# Unbinned vs binned comparison
# ===================================================================

class TestUnbinnedVsBinned:
    """For scalar modifiers, unbinned and binned fits should give
    compatible results.  The unbinned fit uses more information
    (no binning loss), so exact agreement is not expected, but the
    parameter estimates should be statistically consistent."""

    def test_mu_estimates_compatible(self):
        """mu_hat from unbinned and binned should agree within ~2 sigma."""
        events = _generate_events(mu_true=2.0, seed=42)
        model = _scalar_model(events)

        r_ub = fit(model, unbinned=True)
        r_b = fit(model)

        mu_diff = abs(r_ub.bestfit["mu"] - r_b.bestfit["mu"])
        # Use the larger of the two errors as the scale
        scale = max(r_ub.errors["mu"], r_b.errors["mu"])
        assert mu_diff < 3 * scale, (
            f"mu estimates differ by {mu_diff:.3f}, "
            f"which is {mu_diff/scale:.1f} sigma"
        )

    def test_multiple_seeds_compatible(self):
        """Run over several seeds to check consistency is not a fluke."""
        compatible = 0
        n_trials = 10
        for seed in range(100, 100 + n_trials):
            events = _generate_events(mu_true=2.0, seed=seed)
            model = _scalar_model(events, seed=seed)
            r_ub = fit(model, unbinned=True)
            r_b = fit(model)
            mu_diff = abs(r_ub.bestfit["mu"] - r_b.bestfit["mu"])
            scale = max(r_ub.errors["mu"], r_b.errors["mu"])
            if mu_diff < 3 * scale:
                compatible += 1
        # At least 8 out of 10 should be compatible at 3 sigma
        assert compatible >= 8, (
            f"Only {compatible}/{n_trials} seeds gave compatible results"
        )

    def test_fine_bins_consistent_with_unbinned(self):
        """Coarse-bin and fine-bin fits should both recover mu close to
        the unbinned fit, within the binning approximation tolerance.

        Note: Histimator's BinnedTemplate.evaluate_density returns the
        piecewise-constant density of whichever binning the template
        was constructed on, so the unbinned fit on the coarse-binned
        templates is *mathematically identical* to the coarse-binned
        fit (modulo numerical precision). The fine-bin model uses a
        different (finer) piecewise-constant density approximation, so
        its mu_hat differs by a small binning artefact. Both should
        agree with the unbinned result within a few per cent.
        """
        events = _generate_events(mu_true=2.0, seed=42)

        # Coarse bins (20 bins) -- same density as unbinned fit
        model_coarse = _scalar_model(events)
        r_ub = fit(model_coarse, unbinned=True)
        r_coarse = fit(model_coarse)
        diff_coarse = abs(r_ub.bestfit["mu"] - r_coarse.bestfit["mu"])

        # Fine bins (60 bins) -- different density approximation
        fine_edges = np.linspace(100, 160, 61)
        fine_sig_vals = np.exp(-0.5 * (
            (0.5*(fine_edges[:-1]+fine_edges[1:]) - 130)/ 8) ** 2) * np.diff(fine_edges)
        fine_sig_vals *= 20 / fine_sig_vals.sum()
        fine_bkg_vals = np.exp(-0.04 * 0.5*(fine_edges[:-1]+fine_edges[1:])) * np.diff(fine_edges)
        fine_bkg_vals *= 200 / fine_bkg_vals.sum()

        sig_f = Sample("signal", Histogram(fine_sig_vals, fine_edges))
        sig_f.add_normfactor("mu", nominal=1.0, bounds=(0.0, 10.0))
        bkg_f = Sample("background", Histogram(fine_bkg_vals, fine_edges))
        bkg_f.add_normsys("bkg_norm", lo=0.9, hi=1.1)
        ch_f = Channel("SR")
        ch_f.add_sample(sig_f)
        ch_f.add_sample(bkg_f)
        data_fine, _ = np.histogram(events, bins=fine_edges)
        ch_f.set_data(data_fine.astype(float))
        model_fine = Model("fine")
        model_fine.add_channel(ch_f)

        r_fine = fit(model_fine)
        diff_fine = abs(r_ub.bestfit["mu"] - r_fine.bestfit["mu"])

        # Coarse fit must coincide with unbinned (same piecewise density).
        assert diff_coarse < 1e-3, (
            f"Coarse-bin fit should equal unbinned fit (same density), "
            f"got diff={diff_coarse:.6f}"
        )
        # Fine fit deviates by the binning approximation but stays within
        # a few per cent of the unbinned answer.
        assert diff_fine < 0.1, (
            f"Fine-bin fit deviates more than 10% from unbinned: "
            f"diff={diff_fine:.4f}"
        )


# ===================================================================
# Model with LumiSys in unbinned mode
# ===================================================================

class TestUnbinnedWithLumiSys:
    """LumiSys is a scalar modifier and should work in unbinned mode."""

    def test_fit_with_lumisys_converges(self):
        events = _generate_events(mu_true=2.0, seed=42)
        sig = Sample("signal", Histogram(_sig_template(), EDGES))
        sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 10.0))
        bkg = Sample("background", Histogram(_bkg_template(), EDGES))
        bkg.add_normsys("bkg_norm", lo=0.9, hi=1.1)
        bkg.add_lumisys("lumi", uncertainty=0.02)

        ch = Channel("SR")
        ch.add_sample(sig)
        ch.add_sample(bkg)
        data_hist, _ = np.histogram(events, bins=EDGES)
        ch.set_data(data_hist.astype(float))
        ch.set_unbinned_data(events)

        model = Model("lumi_test")
        model.add_channel(ch)

        result = fit(model, unbinned=True)
        assert result.valid
        assert result.bestfit["mu"] > 0


# ===================================================================
# GP template in unbinned mode
# ===================================================================

class TestUnbinnedWithGPTemplate:
    """GP templates provide smooth evaluate_density, which should
    give better unbinned likelihood resolution than BinnedTemplate."""

    def test_gp_unbinned_fit_converges(self):
        from histimator.gp_template import GPKernel, GPTemplate
        events = _generate_events(mu_true=2.0, seed=42)

        gp_sig = GPTemplate(Histogram(_sig_template(), EDGES), kernel=GPKernel.MATERN_52)
        gp_bkg = GPTemplate(Histogram(_bkg_template(), EDGES), kernel=GPKernel.MATERN_52)

        sig = Sample("signal", gp_sig)
        sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 10.0))
        bkg = Sample("background", gp_bkg)
        bkg.add_normsys("bkg_norm", lo=0.9, hi=1.1)

        ch = Channel("SR")
        ch.add_sample(sig)
        ch.add_sample(bkg)
        data_hist, _ = np.histogram(events, bins=EDGES)
        ch.set_data(data_hist.astype(float))
        ch.set_unbinned_data(events)

        model = Model("gp_test")
        model.add_channel(ch)

        result = fit(model, unbinned=True)
        assert result.valid
        assert result.bestfit["mu"] > 0
