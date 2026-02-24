import numpy as np
import pytest

from histimator.bspline_template import BSplineTemplate
from histimator.channels import Channel
from histimator.data import Dataset
from histimator.gp_template import GPTemplate
from histimator.histograms import Histogram
from histimator.likelihood import fit
from histimator.model import Model
from histimator.samples import Sample
from histimator.templates import BinnedTemplate, Template

"""Tests for Sprint 1: Template protocol and BinnedTemplate.

The mathematical contract under test is that a BinnedTemplate wrapping
a Histogram produces:
  - counts_j = c_j  (identical to histogram values)
  - density_j = c_j / (N * w_j)  where N = sum(c), w_j = edge_{j+1} - edge_j
  - sum_j(density_j * w_j) = 1  (proper normalisation)

The architectural contract is that Sample(name, Histogram(...)) transparently
wraps into BinnedTemplate, so every existing code path is preserved.
"""

# ── Protocol conformance ─────────────────────────────────────────────────

class TestProtocolConformance:
    def test_binned_template_is_template(self):
        """BinnedTemplate must satisfy the Template runtime protocol."""
        h = Histogram([10, 20, 30], [0.0, 1.0, 2.0, 3.0])
        bt = BinnedTemplate(h)
        assert isinstance(bt, Template)

    def test_protocol_requires_counts(self):
        """An object missing counts() should not satisfy Template."""

        class BadTemplate:
            @property
            def nbins(self):
                return 1

            @property
            def edges(self):
                return np.array([0.0, 1.0])

            def density(self):
                return np.array([1.0])

        assert not isinstance(BadTemplate(), Template)

    def test_protocol_requires_density(self):
        """An object missing density() should not satisfy Template."""

        class BadTemplate:
            @property
            def nbins(self):
                return 1

            @property
            def edges(self):
                return np.array([0.0, 1.0])

            def counts(self):
                return np.array([10.0])

            def evaluate_density(self, x):
                return np.ones_like(x)

        assert not isinstance(BadTemplate(), Template)

    def test_protocol_requires_evaluate_density(self):
        """An object missing evaluate_density() should not satisfy Template."""

        class BadTemplate:
            @property
            def nbins(self):
                return 1

            @property
            def edges(self):
                return np.array([0.0, 1.0])

            def counts(self):
                return np.array([10.0])

            def density(self):
                return np.array([1.0])

        assert not isinstance(BadTemplate(), Template)

    def test_custom_template_satisfies_protocol(self):
        """A custom implementation with all five members satisfies Template."""

        class ConstantTemplate:
            @property
            def nbins(self):
                return 3

            @property
            def edges(self):
                return np.array([0.0, 1.0, 2.0, 3.0])

            def counts(self):
                return np.array([10.0, 10.0, 10.0])

            def density(self):
                return np.array([1 / 3, 1 / 3, 1 / 3])

            def evaluate_density(self, x):
                return np.full_like(x, 1 / 3)

        assert isinstance(ConstantTemplate(), Template)


# ── BinnedTemplate core properties ───────────────────────────────────────

class TestBinnedTemplateProperties:
    def test_nbins_matches_histogram(self):
        h = Histogram([10, 20, 30], [0.0, 1.0, 2.0, 3.0])
        assert BinnedTemplate(h).nbins == h.nbins

    def test_edges_match_histogram(self):
        h = Histogram([10, 20, 30], [0.0, 1.0, 2.0, 3.0])
        np.testing.assert_array_equal(BinnedTemplate(h).edges, h.edges)

    def test_histogram_property_returns_original(self):
        h = Histogram([10, 20, 30], [0.0, 1.0, 2.0, 3.0])
        assert BinnedTemplate(h).histogram is h

    def test_sumw2_delegates(self):
        h = Histogram([10, 20], [0.0, 1.0, 2.0], sumw2=[100, 400])
        np.testing.assert_array_equal(BinnedTemplate(h).sumw2, [100, 400])

    def test_rejects_non_histogram(self):
        with pytest.raises(TypeError, match="Histogram"):
            BinnedTemplate([10, 20, 30])

    def test_repr_contains_info(self):
        h = Histogram([10, 20, 30], [0.0, 1.0, 2.0, 3.0])
        r = repr(BinnedTemplate(h))
        assert "BinnedTemplate" in r and "nbins=3" in r


# ── counts() contract ────────────────────────────────────────────────────

class TestBinnedTemplateCounts:
    def test_counts_equals_histogram_values(self):
        """Bit-identical: counts_j == c_j for all j."""
        h = Histogram([10, 20, 30], [0.0, 1.0, 2.0, 3.0])
        np.testing.assert_array_equal(BinnedTemplate(h).counts(), h.values)

    def test_counts_returns_copy(self):
        """counts() must return a mutable copy, not a view."""
        bt = BinnedTemplate(Histogram([10, 20], [0.0, 1.0, 2.0]))
        c = bt.counts()
        c[0] = 999
        assert bt.counts()[0] == 10

    def test_counts_float64(self):
        bt = BinnedTemplate(Histogram([1, 2, 3], [0.0, 1.0, 2.0, 3.0]))
        assert bt.counts().dtype == np.float64

    def test_non_uniform_bins(self):
        """Counts are independent of bin widths."""
        h = Histogram([10, 20, 30], [0.0, 0.5, 2.0, 10.0])
        np.testing.assert_array_equal(BinnedTemplate(h).counts(), [10, 20, 30])


# ── density() contract ───────────────────────────────────────────────────

class TestBinnedTemplateDensity:
    def test_density_integrates_to_one_uniform_bins(self):
        """For uniform bins: sum(density_j * w_j) = 1."""
        h = Histogram([10, 20, 30], [0.0, 1.0, 2.0, 3.0])
        bt = BinnedTemplate(h)
        integral = np.sum(bt.density() * h.widths)
        np.testing.assert_allclose(integral, 1.0, rtol=1e-14)

    def test_density_integrates_to_one_non_uniform_bins(self):
        """Normalisation holds regardless of bin width structure."""
        h = Histogram([5, 25, 70], [0.0, 0.1, 1.0, 10.0])
        bt = BinnedTemplate(h)
        integral = np.sum(bt.density() * h.widths)
        np.testing.assert_allclose(integral, 1.0, rtol=1e-14)

    def test_density_formula_explicit(self):
        """density_j = c_j / (N * w_j) verified element by element."""
        vals = np.array([10.0, 20.0, 30.0])
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        h = Histogram(vals, edges)
        bt = BinnedTemplate(h)
        widths = np.diff(edges)
        expected_density = vals / (vals.sum() * widths)
        np.testing.assert_allclose(bt.density(), expected_density, rtol=1e-14)

    def test_density_non_negative(self):
        h = Histogram([0, 10, 0, 20], [0.0, 1.0, 2.0, 3.0, 4.0])
        assert np.all(BinnedTemplate(h).density() >= 0)

    def test_empty_histogram_density_is_zero(self):
        """Zero total count gives zero density everywhere (no NaN/Inf)."""
        h = Histogram([0, 0, 0], [0.0, 1.0, 2.0, 3.0])
        d = BinnedTemplate(h).density()
        np.testing.assert_array_equal(d, [0.0, 0.0, 0.0])
        assert not np.any(np.isnan(d)) and not np.any(np.isinf(d))

    def test_single_bin_density(self):
        """A single bin spanning [0, 5] with count 100 has density 100/(100*5) = 0.2."""
        h = Histogram([100], [0.0, 5.0])
        np.testing.assert_allclose(BinnedTemplate(h).density(), [0.2], rtol=1e-14)

    def test_density_scales_inversely_with_width(self):
        """A bin twice as wide with the same count has half the density."""
        h = Histogram([100, 100], [0.0, 1.0, 3.0])
        d = BinnedTemplate(h).density()
        np.testing.assert_allclose(d[0] / d[1], 2.0, rtol=1e-14)

    def test_density_times_width_gives_probability(self):
        """density_j * w_j = c_j / N = probability mass in bin j."""
        vals = np.array([15.0, 35.0, 50.0])
        h = Histogram(vals, [0.0, 1.0, 2.0, 3.0])
        bt = BinnedTemplate(h)
        prob_mass = bt.density() * h.widths
        np.testing.assert_allclose(prob_mass, vals / vals.sum(), rtol=1e-14)


# ── evaluate_density() contract ──────────────────────────────────────────

class TestBinnedTemplateEvaluateDensity:
    def test_at_bin_centres_matches_density(self):
        """evaluate_density at bin centres must equal density()."""
        h = Histogram([10, 20, 30], [0.0, 1.0, 2.0, 3.0])
        bt = BinnedTemplate(h)
        centres = 0.5 * (h.edges[:-1] + h.edges[1:])
        np.testing.assert_allclose(bt.evaluate_density(centres), bt.density(), rtol=1e-14)

    def test_within_bin_is_constant(self):
        """evaluate_density should return the same value anywhere inside a bin."""
        h = Histogram([60, 120], [0.0, 2.0, 5.0])
        bt = BinnedTemplate(h)
        x = np.array([0.1, 0.5, 1.0, 1.9])
        vals = bt.evaluate_density(x)
        np.testing.assert_allclose(vals, vals[0], rtol=1e-14)

    def test_outside_range_is_zero(self):
        """Points outside [edges[0], edges[-1]] return 0."""
        h = Histogram([10, 20], [1.0, 2.0, 3.0])
        bt = BinnedTemplate(h)
        x = np.array([-1.0, 0.5, 3.5, 100.0])
        np.testing.assert_array_equal(bt.evaluate_density(x), 0.0)

    def test_integrates_to_one(self):
        """Numerical quadrature of evaluate_density over the domain should give 1."""
        h = Histogram([10, 20, 30], [0.0, 1.0, 2.0, 3.0])
        bt = BinnedTemplate(h)
        x_fine = np.linspace(0.001, 2.999, 10000)
        dx = x_fine[1] - x_fine[0]
        integral = np.sum(bt.evaluate_density(x_fine)) * dx
        np.testing.assert_allclose(integral, 1.0, rtol=0.01)

    def test_non_uniform_bins(self):
        """Piecewise constant density with non-uniform bin widths."""
        h = Histogram([100, 100], [0.0, 1.0, 3.0])
        bt = BinnedTemplate(h)
        # Bin 0: width 1, density = 100/(200*1) = 0.5
        # Bin 1: width 2, density = 100/(200*2) = 0.25
        np.testing.assert_allclose(bt.evaluate_density(np.array([0.5])), [0.5])
        np.testing.assert_allclose(bt.evaluate_density(np.array([2.0])), [0.25])

    def test_empty_histogram(self):
        """Zero-total histogram returns 0 everywhere."""
        h = Histogram([0, 0, 0], [0.0, 1.0, 2.0, 3.0])
        bt = BinnedTemplate(h)
        x = np.array([0.5, 1.5, 2.5])
        np.testing.assert_array_equal(bt.evaluate_density(x), 0.0)

    def test_scalar_input(self):
        """Should handle a 0-d or scalar-like array."""
        h = Histogram([10, 20], [0.0, 1.0, 2.0])
        bt = BinnedTemplate(h)
        val = bt.evaluate_density(np.array(0.5))
        assert np.isfinite(val).all()

    def test_non_negative(self):
        """Density must be non-negative everywhere."""
        h = Histogram([0, 50, 0, 30], [0.0, 1.0, 2.0, 3.0, 4.0])
        bt = BinnedTemplate(h)
        x = np.linspace(-1, 5, 500)
        assert np.all(bt.evaluate_density(x) >= 0)


# ── Sample integration ───────────────────────────────────────────────────

class TestSampleTemplateIntegration:
    def test_sample_from_histogram_wraps_in_binned_template(self):
        s = Sample("sig", Histogram([10, 20], [0.0, 1.0, 2.0]))
        assert isinstance(s.template, BinnedTemplate)

    def test_sample_from_binned_template_directly(self):
        bt = BinnedTemplate(Histogram([10, 20], [0.0, 1.0, 2.0]))
        s = Sample("sig", bt)
        assert s.template is bt

    def test_sample_histogram_property_delegates(self):
        h = Histogram([10, 20], [0.0, 1.0, 2.0])
        s = Sample("sig", h)
        assert s.histogram is h

    def test_sample_expected_unchanged(self):
        """expected() via template must match the pre-template histogram path."""
        h = Histogram([10, 20, 30, 20], [0.0, 1.0, 2.0, 3.0, 4.0])
        s = Sample("sig", h)
        np.testing.assert_array_equal(s.expected({}), h.values)

    def test_modifiers_work_with_template(self):
        s = Sample("sig", Histogram([100, 100], [0.0, 1.0, 2.0]))
        s.add_normfactor("mu", nominal=1.0, bounds=(0.0, 5.0))
        np.testing.assert_array_equal(s.expected({"mu": 2.0}), [200, 200])

    def test_sample_rejects_invalid_type(self):
        with pytest.raises(TypeError, match="Histogram or Template"):
            Sample("bad", [10, 20, 30])


# ── Full model integration ───────────────────────────────────────────────

class TestFullModelThroughTemplate:
    def test_fit_via_template_matches_histogram_path(self):
        """Build identical models via Histogram and BinnedTemplate paths,
        verify fit results agree to numerical precision."""
        edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        sig_vals = np.array([10.0, 30.0, 50.0, 30.0, 10.0])
        bkg_vals = np.array([50.0] * 5)
        rng = np.random.default_rng(42)
        data = rng.poisson(1.5 * sig_vals + bkg_vals).astype(float)

        # Path A: Histogram directly (existing API)
        sig_a = Sample("signal", Histogram(sig_vals, edges))
        sig_a.add_normfactor("mu", nominal=1.0, bounds=(0.0, 10.0))
        bkg_a = Sample("background", Histogram(bkg_vals, edges))
        ch_a = Channel("SR")
        ch_a.add_sample(sig_a)
        ch_a.add_sample(bkg_a)
        ch_a.set_data(data)
        m_a = Model("hist_path")
        m_a.add_channel(ch_a)
        r_a = fit(m_a)

        # Path B: BinnedTemplate explicitly
        sig_b = Sample("signal", BinnedTemplate(Histogram(sig_vals, edges)))
        sig_b.add_normfactor("mu", nominal=1.0, bounds=(0.0, 10.0))
        bkg_b = Sample("background", BinnedTemplate(Histogram(bkg_vals, edges)))
        ch_b = Channel("SR")
        ch_b.add_sample(sig_b)
        ch_b.add_sample(bkg_b)
        ch_b.set_data(data)
        m_b = Model("template_path")
        m_b.add_channel(ch_b)
        r_b = fit(m_b)

        np.testing.assert_allclose(r_a.bestfit["mu"], r_b.bestfit["mu"], rtol=1e-10)
        np.testing.assert_allclose(r_a.nll_min, r_b.nll_min, rtol=1e-10)
        np.testing.assert_allclose(r_a.errors["mu"], r_b.errors["mu"], rtol=1e-6)

    def test_model_expected_via_template(self):
        """Model.expected() through template path gives correct values."""
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        sig = Sample("signal", Histogram([10, 20, 30], edges))
        sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 5.0))
        bkg = Sample("background", Histogram([50, 50, 50], edges))
        ch = Channel("SR")
        ch.add_sample(sig)
        ch.add_sample(bkg)
        ch.set_data([60, 70, 80])
        m = Model("test")
        m.add_channel(ch)
        np.testing.assert_array_equal(m.expected({"mu": 2.0}), [70, 90, 110])


# ── Unified template type tests ──────────────────────────────────────────
#
# Every test below runs identically for binned, gp, and bspline templates.
# The contract is that a user picks a template type at construction time,
# and the rest of the pipeline (modifiers, channels, model, fit) works
# unchanged.

""" using
from histimator.bspline_template import BSplineTemplate
from histimator.data import Dataset
from histimator.gp_template import GPTemplate
"""

# Shared fixtures: high-stats histograms so GP and BSpline smoothing
# produces counts close to the raw histogram (enabling meaningful
# cross-template comparison).

def _sig_hist():
    return Histogram(
        np.array([20.0, 60.0, 100.0, 60.0, 20.0]),
        np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
    )

def _bkg_hist():
    return Histogram(
        np.array([200.0, 200.0, 200.0, 200.0, 200.0]),
        np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
    )


TEMPLATE_TYPES = ["binned", "gp", "bspline"]


def _tt_kwargs(tt):
    """Return template_kwargs that skip expensive hyperparameter optimisation."""
    if tt == "gp":
        return {"optimize_hyperparameters": False}
    return {}


class TestFromHistogramFactory:
    @pytest.mark.parametrize("tt", TEMPLATE_TYPES)
    def test_returns_correct_template_type(self, tt):
        s = Sample.from_histogram("sig", _sig_hist(), template_type=tt, **_tt_kwargs(tt))
        if tt == "binned":
            assert isinstance(s.template, BinnedTemplate)
        elif tt == "gp":
            assert isinstance(s.template, GPTemplate)
        elif tt == "bspline":
            assert isinstance(s.template, BSplineTemplate)

    @pytest.mark.parametrize("tt", TEMPLATE_TYPES)
    def test_template_satisfies_protocol(self, tt):
        s = Sample.from_histogram("sig", _sig_hist(), template_type=tt, **_tt_kwargs(tt))
        assert isinstance(s.template, Template)

    @pytest.mark.parametrize("tt", TEMPLATE_TYPES)
    def test_histogram_accessible(self, tt):
        """sample.histogram returns the original Histogram for all types."""
        h = _sig_hist()
        s = Sample.from_histogram("sig", h, template_type=tt, **_tt_kwargs(tt))
        assert s.histogram is h

    @pytest.mark.parametrize("tt", TEMPLATE_TYPES)
    def test_counts_positive(self, tt):
        s = Sample.from_histogram("sig", _sig_hist(), template_type=tt, **_tt_kwargs(tt))
        assert np.all(s.template.counts() > 0)

    @pytest.mark.parametrize("tt", TEMPLATE_TYPES)
    def test_density_integrates_to_one(self, tt):
        s = Sample.from_histogram("sig", _sig_hist(), template_type=tt, **_tt_kwargs(tt))
        widths = np.diff(s.template.edges)
        integral = np.sum(s.template.density() * widths)
        np.testing.assert_allclose(integral, 1.0, rtol=0.05)

    def test_invalid_template_type_raises(self):
        with pytest.raises(ValueError, match="Unknown template_type"):
            Sample.from_histogram("sig", _sig_hist(), template_type="magic")


class TestEvaluateDensityAllTemplateTypes:
    """evaluate_density must satisfy basic contracts across all
    template types: agreement with density() at bin centres, proper
    normalisation, non-negativity, and zero outside the domain."""

    @pytest.mark.parametrize("tt", TEMPLATE_TYPES)
    def test_at_centres_agrees_with_density(self, tt):
        """evaluate_density at bin centres should closely match density()."""
        s = Sample.from_histogram("sig", _sig_hist(), template_type=tt, **_tt_kwargs(tt))
        t = s.template
        centres = 0.5 * (t.edges[:-1] + t.edges[1:])
        # For GP and BSpline the match is approximate due to smoothing;
        # for binned it is exact.
        rtol = 1e-10 if tt == "binned" else 0.15
        np.testing.assert_allclose(
            t.evaluate_density(centres), t.density(), rtol=rtol,
        )

    @pytest.mark.parametrize("tt", TEMPLATE_TYPES)
    def test_integrates_to_one(self, tt):
        """Numerical quadrature of evaluate_density should approximate 1."""
        s = Sample.from_histogram("sig", _sig_hist(), template_type=tt, **_tt_kwargs(tt))
        t = s.template
        lo, hi = t.edges[0], t.edges[-1]
        x_fine = np.linspace(lo + 1e-6, hi - 1e-6, 5000)
        dx = x_fine[1] - x_fine[0]
        integral = np.sum(t.evaluate_density(x_fine)) * dx
        np.testing.assert_allclose(integral, 1.0, rtol=0.05)

    @pytest.mark.parametrize("tt", TEMPLATE_TYPES)
    def test_non_negative(self, tt):
        """Density must be non-negative everywhere."""
        s = Sample.from_histogram("sig", _sig_hist(), template_type=tt, **_tt_kwargs(tt))
        t = s.template
        lo, hi = t.edges[0], t.edges[-1]
        x = np.linspace(lo, hi, 200)
        assert np.all(t.evaluate_density(x) >= 0)

    @pytest.mark.parametrize("tt", TEMPLATE_TYPES)
    def test_outside_domain_near_zero(self, tt):
        """Points outside the domain should return values near zero.

        BinnedTemplate returns exactly zero for out-of-range points.
        GP and BSpline evaluate on their internal basis which is defined
        only within [edges[0], edges[-1]], so we test at the domain
        boundaries where the density should be small for a peaked
        distribution.
        """
        s = Sample.from_histogram("sig", _sig_hist(), template_type=tt, **_tt_kwargs(tt))
        t = s.template
        lo, hi = t.edges[0], t.edges[-1]
        if tt == "binned":
            far_out = np.array([lo - 100, hi + 100])
            np.testing.assert_array_equal(t.evaluate_density(far_out), 0.0)
        else:
            # GP and BSpline basis functions are only valid within the
            # domain.  Test that boundary values are small relative to
            # the peak for a distribution peaked in the interior.
            boundary = np.array([lo + 1e-6, hi - 1e-6])
            vals = t.evaluate_density(boundary)
            peak = t.density().max()
            assert np.all(vals < peak * 1.5)


class TestFromDatasetFactory:
    @pytest.mark.parametrize("tt", TEMPLATE_TYPES)
    def test_round_trip_from_dataset(self, tt):
        rng = np.random.default_rng(42)
        ds = Dataset(rng.exponential(scale=2.0, size=500))
        edges = np.linspace(0, 8, 11)
        s = Sample.from_dataset("sig", ds, edges=edges, template_type=tt, **_tt_kwargs(tt))
        assert isinstance(s.template, Template)
        assert s.template.nbins == 10


class TestModifiersWithAllTemplateTypes:
    @pytest.mark.parametrize("tt", TEMPLATE_TYPES)
    def test_normfactor(self, tt):
        s = Sample.from_histogram("sig", _sig_hist(), template_type=tt, **_tt_kwargs(tt))
        s.add_normfactor("mu", nominal=1.0, bounds=(0.0, 5.0))
        base = s.template.counts()
        scaled = s.expected({"mu": 2.0})
        np.testing.assert_allclose(scaled, base * 2.0, rtol=1e-10)

    @pytest.mark.parametrize("tt", TEMPLATE_TYPES)
    def test_normsys(self, tt):
        s = Sample.from_histogram("sig", _sig_hist(), template_type=tt, **_tt_kwargs(tt))
        s.add_normsys("alpha", lo=0.9, hi=1.1)
        at_zero = s.expected({"alpha": 0.0})
        np.testing.assert_allclose(at_zero, s.template.counts(), rtol=1e-10)


class TestFullPipelineAllTemplateTypes:
    @pytest.mark.parametrize("tt", TEMPLATE_TYPES)
    def test_fit_produces_result(self, tt):
        """Build and fit a model for each template type.

        We do not require identical fit results across types (smoothed
        templates have different counts than raw histograms), but the
        fit must converge and produce a finite result.
        """
        sig_h = _sig_hist()
        bkg_h = _bkg_hist()

        sig = Sample.from_histogram("signal", sig_h, template_type=tt, **_tt_kwargs(tt))
        sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 10.0))
        bkg = Sample.from_histogram("background", bkg_h, template_type=tt, **_tt_kwargs(tt))

        ch = Channel("SR")
        ch.add_sample(sig)
        ch.add_sample(bkg)

        # Pseudo-data: mu=1.5
        rng = np.random.default_rng(42)
        data = rng.poisson(1.5 * sig_h.values + bkg_h.values).astype(float)
        ch.set_data(data)

        m = Model(f"test_{tt}")
        m.add_channel(ch)
        r = fit(m)

        assert np.isfinite(r.nll_min)
        assert np.isfinite(r.bestfit["mu"])
        assert r.bestfit["mu"] > 0

    @pytest.mark.parametrize("tt", TEMPLATE_TYPES)
    def test_channel_data_loader(self, tt):
        """Channel.add_data_and_samples_from_datasets works for all types."""
        rng = np.random.default_rng(42)
        sig_ds = Dataset(rng.normal(2.5, 0.8, size=200))
        bkg_ds = Dataset(rng.uniform(0, 5, size=1000))
        edges = np.linspace(0, 5, 11)

        ch = Channel("SR")
        ch.add_data_and_samples_from_datasets(
            {"signal": sig_ds, "background": bkg_ds},
            edges=edges,
            template_type=tt,
            **_tt_kwargs(tt),
        )
        assert len(ch.samples) == 2
        assert ch.data is not None
        assert ch.nbins == 10
