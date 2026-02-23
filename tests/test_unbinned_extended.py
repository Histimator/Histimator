"""Tests for full unbinned support across the codebase.

Tier 1 validates that the unbinned flag threads correctly through
inference.py, scans.py, and diagnostics.py, producing results
compatible with the binned path.

Tier 2 validates the new unbinned toy generation machinery in toys.py
(rejection sampling, event generation, model construction) and the
unbinned goodness-of-fit in scans.py.
"""

import numpy as np
import pytest

from histimator.channels import Channel
from histimator.diagnostics import (
    impacts,
    nuisance_parameter_pulls,
    prefit_postfit_yields,
)
from histimator.histograms import Histogram
from histimator.inference import (
    discovery_significance,
    profile_likelihood_ratio,
    upper_limit,
)
from histimator.model import Model
from histimator.samples import Sample
from histimator.scans import (
    goodness_of_fit,
    likelihood_scan_1d,
    likelihood_scan_2d,
    scan_intervals,
)
from histimator.toys import (
    _build_unbinned_toy_model,
    _sample_from_density,
    generate_unbinned_toy_data,
    toy_cls,
)

# ===================================================================
# Shared fixtures
# ===================================================================

EDGES = np.linspace(100, 160, 21)
CENTRES = 0.5 * (EDGES[:-1] + EDGES[1:])
WIDTHS = np.diff(EDGES)


def _sig_vals(total=20.0):
    vals = np.exp(-0.5 * ((CENTRES - 130) / 8) ** 2) * WIDTHS
    vals *= total / vals.sum()
    return vals


def _bkg_vals(total=200.0):
    vals = np.exp(-0.04 * CENTRES) * WIDTHS
    vals *= total / vals.sum()
    return vals


def _generate_events(mu_true=1.0, n_bkg=200, seed=42):
    rng = np.random.default_rng(seed)
    n_sig = rng.poisson(int(mu_true * 20))
    sig_events = rng.normal(130, 8, n_sig)
    bkg_events = rng.exponential(1 / 0.04, rng.poisson(n_bkg)) + 100
    all_events = np.concatenate([sig_events, bkg_events])
    mask = (all_events >= EDGES[0]) & (all_events <= EDGES[-1])
    return all_events[mask]


def _scalar_model(events):
    """Model with scalar-only modifiers, both binned and unbinned data."""
    sig = Sample("signal", Histogram(_sig_vals(), EDGES))
    sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 10.0))
    bkg = Sample("background", Histogram(_bkg_vals(), EDGES))
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


@pytest.fixture
def events():
    return _generate_events(mu_true=1.0, seed=42)


@pytest.fixture
def model(events):
    return _scalar_model(events)


# ===================================================================
# Tier 1: inference.py unbinned plumbing
# ===================================================================


class TestInferenceUnbinned:
    """Unbinned flag threads through inference.py correctly."""

    def test_profile_likelihood_ratio_runs(self, model):
        t = profile_likelihood_ratio(model, "mu", 2.0, unbinned=True)
        assert t >= 0.0
        assert np.isfinite(t)

    def test_profile_likelihood_ratio_agrees_with_binned(self, model):
        t_binned = profile_likelihood_ratio(model, "mu", 2.0, unbinned=False)
        t_unbinned = profile_likelihood_ratio(model, "mu", 2.0, unbinned=True)
        # Both should be positive and in the same ballpark.
        # They differ because the likelihoods have different granularity.
        assert abs(t_unbinned - t_binned) < max(t_binned, 1.0)

    def test_upper_limit_runs(self, model):
        ul = upper_limit(model, "mu", unbinned=True)
        assert ul > 0.0
        assert np.isfinite(ul)

    def test_upper_limit_agrees_with_binned(self, model):
        ul_b = upper_limit(model, "mu", unbinned=False)
        ul_u = upper_limit(model, "mu", unbinned=True)
        # Should be in the same region.
        assert abs(ul_u - ul_b) / max(ul_b, 0.1) < 1.0

    def test_discovery_significance_runs(self, model):
        z, p = discovery_significance(model, "mu", unbinned=True)
        assert z >= 0.0
        assert 0.0 <= p <= 1.0

    def test_discovery_significance_agrees_with_binned(self, model):
        z_b, _ = discovery_significance(model, "mu", unbinned=False)
        z_u, _ = discovery_significance(model, "mu", unbinned=True)
        assert abs(z_u - z_b) < max(z_b, 1.0)


# ===================================================================
# Tier 1: scans.py unbinned plumbing
# ===================================================================


class TestScansUnbinned:
    """Unbinned flag threads through scans.py correctly."""

    def test_likelihood_scan_1d_runs(self, model):
        par_vals, dnll = likelihood_scan_1d(
            model, "mu", bounds=(0.0, 3.0), n_points=5, unbinned=True,
        )
        assert len(par_vals) == 5
        assert len(dnll) == 5
        assert np.all(dnll >= 0.0)

    def test_likelihood_scan_1d_minimum_near_zero(self, model):
        _, dnll = likelihood_scan_1d(
            model, "mu", bounds=(0.0, 3.0), n_points=11, unbinned=True,
        )
        assert dnll.min() < 0.5

    def test_scan_intervals_from_unbinned(self, model):
        par_vals, dnll = likelihood_scan_1d(
            model, "mu", bounds=(0.0, 5.0), n_points=21, unbinned=True,
        )
        lo, hi = scan_intervals(par_vals, dnll, level=1.0)
        assert lo < hi
        assert lo >= 0.0

    def test_likelihood_scan_2d_runs(self, model):
        xv, yv, dnll = likelihood_scan_2d(
            model, "mu", "bkg_norm",
            bounds_x=(0.5, 2.0), bounds_y=(-1.0, 1.0),
            n_points_x=3, n_points_y=3, unbinned=True,
        )
        assert xv.shape == (3,)
        assert yv.shape == (3,)
        assert dnll.shape == (3, 3)
        assert np.all(dnll >= 0.0)


# ===================================================================
# Tier 1: diagnostics.py unbinned plumbing
# ===================================================================


class TestDiagnosticsUnbinned:
    """Unbinned flag threads through diagnostics.py correctly."""

    def test_prefit_postfit_yields_runs(self, model):
        result = prefit_postfit_yields(model, unbinned=True)
        assert "SR" in result
        assert "prefit" in result["SR"]
        assert "postfit" in result["SR"]

    def test_prefit_postfit_yields_positive(self, model):
        result = prefit_postfit_yields(model, unbinned=True)
        assert np.all(result["SR"]["prefit"]["total"] >= 0)
        assert np.all(result["SR"]["postfit"]["total"] >= 0)

    def test_nuisance_parameter_pulls_runs(self, model):
        pulls = nuisance_parameter_pulls(model, unbinned=True)
        assert "bkg_norm" in pulls
        assert "pull" in pulls["bkg_norm"]
        assert "constraint" in pulls["bkg_norm"]

    def test_nuisance_parameter_pulls_bounded(self, model):
        pulls = nuisance_parameter_pulls(model, unbinned=True)
        # Pull should be small for well-behaved data.
        assert abs(pulls["bkg_norm"]["pull"]) < 5.0
        # Constraint should be between 0 and 1 (data constrains).
        assert 0.0 < pulls["bkg_norm"]["constraint"] <= 1.5

    def test_impacts_runs(self, model):
        imp = impacts(model, "mu", unbinned=True)
        assert len(imp) == 1  # only bkg_norm
        assert imp[0]["name"] == "bkg_norm"
        assert np.isfinite(imp[0]["impact_up"])
        assert np.isfinite(imp[0]["impact_down"])


# ===================================================================
# Tier 2: rejection sampling from template density
# ===================================================================


class TestSampleFromDensity:
    """_sample_from_density draws events matching the template shape."""

    def test_correct_count(self):
        template = Sample("sig", Histogram(_sig_vals(), EDGES)).template
        rng = np.random.default_rng(0)
        events = _sample_from_density(template, 500, rng)
        assert len(events) == 500

    def test_in_domain(self):
        template = Sample("sig", Histogram(_sig_vals(), EDGES)).template
        rng = np.random.default_rng(1)
        events = _sample_from_density(template, 1000, rng)
        assert np.all(events >= EDGES[0])
        assert np.all(events <= EDGES[-1])

    def test_signal_mean_near_peak(self):
        """Signal is Gaussian at 130; sampled mean should be close."""
        template = Sample("sig", Histogram(_sig_vals(), EDGES)).template
        rng = np.random.default_rng(2)
        events = _sample_from_density(template, 5000, rng)
        assert abs(events.mean() - 130.0) < 3.0

    def test_background_shape_monotone(self):
        """Background is falling exponential; lower bins should be fuller."""
        template = Sample("bkg", Histogram(_bkg_vals(), EDGES)).template
        rng = np.random.default_rng(3)
        events = _sample_from_density(template, 5000, rng)
        # More events in the lower half than upper half.
        midpoint = 0.5 * (EDGES[0] + EDGES[-1])
        n_low = np.sum(events < midpoint)
        n_high = np.sum(events >= midpoint)
        assert n_low > n_high

    def test_zero_events_returns_empty(self):
        template = Sample("sig", Histogram(_sig_vals(), EDGES)).template
        rng = np.random.default_rng(4)
        events = _sample_from_density(template, 0, rng)
        assert len(events) == 0

    def test_binned_distribution_chi2(self):
        """Sampled events should follow the template density (chi2 test)."""
        template = Sample("bkg", Histogram(_bkg_vals(total=1000.0), EDGES)).template
        rng = np.random.default_rng(5)
        events = _sample_from_density(template, 10000, rng)
        observed, _ = np.histogram(events, bins=EDGES)
        # Expected proportions from the template density.
        density_at_centres = template.evaluate_density(CENTRES)
        expected_frac = density_at_centres * WIDTHS
        expected_frac /= expected_frac.sum()
        expected_counts = expected_frac * 10000
        # Chi2 test: should not reject at p < 0.001.
        mask = expected_counts > 5
        chi2 = np.sum((observed[mask] - expected_counts[mask]) ** 2
                       / expected_counts[mask])
        ndof = mask.sum() - 1
        from scipy import stats
        p = stats.chi2.sf(chi2, ndof)
        assert p > 0.001, f"chi2={chi2:.1f}, ndof={ndof}, p={p:.4f}"


# ===================================================================
# Tier 2: unbinned toy data generation
# ===================================================================


class TestGenerateUnbinnedToyData:
    """generate_unbinned_toy_data produces per-channel event arrays."""

    def test_returns_list_per_channel(self, model):
        rng = np.random.default_rng(10)
        params = {"mu": 1.0, "bkg_norm": 0.0}
        result = generate_unbinned_toy_data(model, params, rng)
        assert isinstance(result, list)
        assert len(result) == 1  # one channel

    def test_events_in_domain(self, model):
        rng = np.random.default_rng(11)
        params = {"mu": 1.0, "bkg_norm": 0.0}
        result = generate_unbinned_toy_data(model, params, rng)
        events = result[0]
        assert np.all(events >= EDGES[0])
        assert np.all(events <= EDGES[-1])

    def test_event_count_plausible(self, model):
        """Total events should be Poisson-scattered around ~220."""
        params = {"mu": 1.0, "bkg_norm": 0.0}
        counts = []
        for seed in range(20):
            rng2 = np.random.default_rng(100 + seed)
            result = generate_unbinned_toy_data(model, params, rng2)
            counts.append(len(result[0]))
        mean_count = np.mean(counts)
        # Expected ~ 220 (20 sig + 200 bkg).
        assert 150 < mean_count < 300

    def test_zero_signal(self, model):
        """mu=0 should produce only background events."""
        rng = np.random.default_rng(13)
        params = {"mu": 0.0, "bkg_norm": 0.0}
        result = generate_unbinned_toy_data(model, params, rng)
        events = result[0]
        # Should have ~200 events, not ~220.
        assert len(events) < 250


# ===================================================================
# Tier 2: build unbinned toy model
# ===================================================================


class TestBuildUnbinnedToyModel:
    """_build_unbinned_toy_model sets both binned and unbinned data."""

    def test_has_unbinned_data(self, model):
        rng = np.random.default_rng(20)
        params = {"mu": 1.0, "bkg_norm": 0.0}
        toy_events = generate_unbinned_toy_data(model, params, rng)
        toy_model = _build_unbinned_toy_model(model, toy_events)
        assert toy_model.channels[0].unbinned_data is not None

    def test_has_binned_data(self, model):
        rng = np.random.default_rng(21)
        params = {"mu": 1.0, "bkg_norm": 0.0}
        toy_events = generate_unbinned_toy_data(model, params, rng)
        toy_model = _build_unbinned_toy_model(model, toy_events)
        assert toy_model.channels[0].data is not None

    def test_binned_unbinned_consistent(self, model):
        """Binned total should equal number of unbinned events."""
        rng = np.random.default_rng(22)
        params = {"mu": 1.0, "bkg_norm": 0.0}
        toy_events = generate_unbinned_toy_data(model, params, rng)
        toy_model = _build_unbinned_toy_model(model, toy_events)
        ch = toy_model.channels[0]
        assert ch.data.total == len(ch.unbinned_data)

    def test_preserves_samples(self, model):
        rng = np.random.default_rng(23)
        params = {"mu": 1.0, "bkg_norm": 0.0}
        toy_events = generate_unbinned_toy_data(model, params, rng)
        toy_model = _build_unbinned_toy_model(model, toy_events)
        assert len(toy_model.channels[0].samples) == 2
        names = {s.name for s in toy_model.channels[0].samples}
        assert names == {"signal", "background"}

    def test_preserves_parameters(self, model):
        rng = np.random.default_rng(24)
        params = {"mu": 1.0, "bkg_norm": 0.0}
        toy_events = generate_unbinned_toy_data(model, params, rng)
        toy_model = _build_unbinned_toy_model(model, toy_events)
        par_names = set(toy_model.parameter_names)
        assert par_names == {"mu", "bkg_norm"}


# ===================================================================
# Tier 2: unbinned toy CLs
# ===================================================================


class TestToyClsUnbinned:
    """toy_cls with unbinned=True exercises the full pipeline."""

    def test_returns_finite(self, model):
        cls_val = toy_cls(
            model, "mu", 2.0, n_toys=10, seed=50, unbinned=True,
        )
        assert np.isfinite(cls_val)
        assert 0.0 <= cls_val <= 1.0

    def test_large_mu_small_cls(self, model):
        """At large mu, CLs should be small (signal strongly excluded)."""
        cls_val = toy_cls(
            model, "mu", 5.0, n_toys=20, seed=51, unbinned=True,
        )
        # With 20 toys this is noisy, but should be < 1.
        assert cls_val < 1.0

    def test_small_mu_large_cls(self, model):
        """At small mu, CLs should be large (signal not excluded)."""
        cls_val = toy_cls(
            model, "mu", 0.1, n_toys=20, seed=52, unbinned=True,
        )
        assert cls_val > 0.0


# ===================================================================
# Tier 2: unbinned goodness-of-fit
# ===================================================================


class TestGoodnessOfFitUnbinned:
    """goodness_of_fit with unbinned=True."""

    def test_returns_three_values(self, model):
        chi2_val, ndof, p_val = goodness_of_fit(model, unbinned=True)
        assert np.isfinite(chi2_val)
        assert ndof > 0
        assert 0.0 <= p_val <= 1.0

    def test_chi2_non_negative(self, model):
        chi2_val, _, _ = goodness_of_fit(model, unbinned=True)
        assert chi2_val >= 0.0

    def test_ndof_correct(self, model):
        """ndof = nbins - n_floated_parameters."""
        _, ndof, _ = goodness_of_fit(model, unbinned=True)
        n_floated = sum(1 for p in model.parameters if not p.fixed)
        expected_ndof = model.total_nbins - n_floated
        assert ndof == expected_ndof

    def test_agrees_with_binned_approximately(self, model):
        """Unbinned and binned GoF should give similar chi2 values."""
        chi2_b, ndof_b, _ = goodness_of_fit(model, unbinned=False)
        chi2_u, ndof_u, _ = goodness_of_fit(model, unbinned=True)
        assert ndof_b == ndof_u
        # The chi2 values differ because the best-fit parameters
        # differ slightly.  They should be in the same ballpark.
        assert abs(chi2_u - chi2_b) < max(chi2_b, 5.0)

    def test_good_model_high_pvalue(self):
        """A model that matches the data well should have high p-value."""
        # Generate data exactly from the model at nominal parameters.
        rng = np.random.default_rng(70)
        sig_counts = _sig_vals(total=20.0)
        bkg_counts = _bkg_vals(total=200.0)
        expected = sig_counts + bkg_counts
        data = rng.poisson(expected).astype(float)

        sig = Sample("signal", Histogram(_sig_vals(), EDGES))
        sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 10.0))
        bkg = Sample("background", Histogram(_bkg_vals(), EDGES))
        bkg.add_normsys("bkg_norm", lo=0.9, hi=1.1)
        ch = Channel("SR")
        ch.add_sample(sig)
        ch.add_sample(bkg)
        ch.set_data(data)

        # Generate unbinned events from the same model.
        sig_events = rng.normal(130, 8, rng.poisson(20))
        bkg_events = rng.exponential(1 / 0.04, rng.poisson(200)) + 100
        all_events = np.concatenate([sig_events, bkg_events])
        mask = (all_events >= EDGES[0]) & (all_events <= EDGES[-1])
        ch.set_unbinned_data(all_events[mask])

        m = Model("good_model")
        m.add_channel(ch)

        _, _, p = goodness_of_fit(m, unbinned=True)
        # Should not be pathologically low.
        assert p > 0.001

    def test_missing_unbinned_data_raises(self):
        """Requesting unbinned GoF without unbinned data gives an error."""
        sig = Sample("signal", Histogram(_sig_vals(), EDGES))
        sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 10.0))
        bkg = Sample("background", Histogram(_bkg_vals(), EDGES))

        ch = Channel("SR")
        ch.add_sample(sig)
        ch.add_sample(bkg)
        ch.set_data(np.ones(20))
        # No set_unbinned_data call.

        m = Model("no_unbinned")
        m.add_channel(ch)

        with pytest.raises(ValueError, match="no unbinned data"):
            goodness_of_fit(m, unbinned=True)
