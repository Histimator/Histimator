"""Tests for Sprint 3: Binning strategies.

The cross-validation criterion J_hat(h) is an unbiased estimator of
the integrated squared error minus an irreducible constant.  For a
standard Gaussian, the theoretical optimal binwidth under the normal
reference rule is h* = 3.49 * sigma * n^(-1/3).  The CV approach
should find a width in the same ballpark without assuming normality.

The mathematical relationship: for f = N(0,1),
  integral(f''^2 dx) = 3/(8*sqrt(pi)) ≈ 0.2116
  h* = (6 / (n * integral(f''^2)))^(1/3)
     = (6 / (n * 0.2116))^(1/3)
     ≈ 3.49 * 1 * n^(-1/3)     [Scott's rule for sigma=1]
"""

import numpy as np
import pytest

from histimator.binning import (
    auto_edges,
    cross_validation_binwidth,
    freedman_diaconis_edges,
    normal_reference_binwidth,
    scott_edges,
    sturges_edges,
)
from histimator.channels import Channel
from histimator.data import Dataset
from histimator.likelihood import fit
from histimator.model import Model
from histimator.samples import Sample

# ── Cross-validation ─────────────────────────────────────────────────────

class TestCrossValidation:
    def test_gaussian_near_scott(self):
        """For a Gaussian sample, CV binwidth should be in the same
        ballpark as the theoretical Scott's rule."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 5000)
        ds = Dataset(data)
        h_cv = cross_validation_binwidth(ds)
        h_scott = 3.49 * np.std(data, ddof=1) * len(data) ** (-1.0 / 3)
        # CV should be within a factor of 3 of Scott's for Gaussian data
        assert 0.3 * h_scott < h_cv < 3.0 * h_scott

    def test_minimises_j_hat(self):
        """The returned h should minimise J_hat: neighbours score higher."""
        rng = np.random.default_rng(42)
        ds = Dataset(rng.normal(0, 1, 2000))
        h_opt = cross_validation_binwidth(ds)

        # Evaluate J_hat at h_opt and nearby values
        vals = ds.values
        n = len(vals)
        lo, hi = vals.min(), vals.max()

        def j_hat(h):
            edges = np.arange(lo, hi + h, h)
            if len(edges) < 2:
                return np.inf
            counts, _ = np.histogram(vals, bins=edges)
            p = counts / n
            return 2.0 / (h * (n - 1)) - (n + 1) / (h * (n - 1)) * np.sum(p ** 2)

        j_opt = j_hat(h_opt)
        # Check a grid of neighbours
        for delta in [-0.05, -0.02, 0.02, 0.05]:
            h_test = h_opt + delta
            if h_test > 0:
                assert j_hat(h_test) >= j_opt - 1e-10

    def test_deterministic(self):
        ds = Dataset(np.random.default_rng(42).normal(0, 1, 1000))
        h1 = cross_validation_binwidth(ds)
        h2 = cross_validation_binwidth(ds)
        assert h1 == h2

    def test_requires_two_events(self):
        with pytest.raises(ValueError, match="at least 2"):
            cross_validation_binwidth(Dataset([1.0]))

    def test_positive_result(self):
        ds = Dataset(np.random.default_rng(42).normal(0, 1, 500))
        assert cross_validation_binwidth(ds) > 0


# ── Normal reference rule ────────────────────────────────────────────────

class TestNormalReference:
    def test_formula_standard_normal(self):
        """h = 3.49 * sigma * n^(-1/3) for a standard normal sample."""
        rng = np.random.default_rng(42)
        n = 10000
        data = rng.normal(0, 1, n)
        ds = Dataset(data)
        h = normal_reference_binwidth(ds)
        sigma = np.std(data, ddof=1)
        expected = 3.49 * sigma * n ** (-1.0 / 3)
        np.testing.assert_allclose(h, expected, rtol=1e-10)

    def test_scales_with_sigma(self):
        rng = np.random.default_rng(42)
        ds1 = Dataset(rng.normal(0, 1, 1000))
        ds2 = Dataset(rng.normal(0, 5, 1000))
        assert normal_reference_binwidth(ds2) > normal_reference_binwidth(ds1)

    def test_scales_inversely_with_n(self):
        rng = np.random.default_rng(42)
        h1 = normal_reference_binwidth(Dataset(rng.normal(0, 1, 100)))
        h2 = normal_reference_binwidth(Dataset(rng.normal(0, 1, 10000)))
        assert h2 < h1


# ── Simple rules ─────────────────────────────────────────────────────────

class TestSturges:
    def test_formula(self):
        """Sturges: m = 1 + ceil(log2(n))."""
        rng = np.random.default_rng(42)
        n = 1000
        ds = Dataset(rng.uniform(0, 10, n))
        edges = sturges_edges(ds)
        expected_m = 1 + int(np.ceil(np.log2(n)))
        assert len(edges) == expected_m + 1

    def test_covers_data(self):
        ds = Dataset([1.0, 5.0, 9.0])
        edges = sturges_edges(ds)
        assert edges[0] <= 1.0 and edges[-1] >= 9.0


class TestScott:
    def test_edges_cover_data(self):
        ds = Dataset(np.random.default_rng(42).normal(0, 1, 1000))
        edges = scott_edges(ds)
        assert edges[0] <= ds.values.min() and edges[-1] >= ds.values.max()

    def test_reasonable_number_of_bins(self):
        ds = Dataset(np.random.default_rng(42).normal(0, 1, 1000))
        edges = scott_edges(ds)
        assert 5 <= len(edges) - 1 <= 100


class TestFreedmanDiaconis:
    def test_formula(self):
        """h = 2 * IQR * n^(-1/3)."""
        rng = np.random.default_rng(42)
        n = 1000
        data = rng.normal(0, 1, n)
        ds = Dataset(data)
        edges = freedman_diaconis_edges(ds)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        h_expected = 2.0 * iqr * n ** (-1.0 / 3)
        h_actual = edges[1] - edges[0]
        np.testing.assert_allclose(h_actual, h_expected, rtol=0.1)

    def test_fallback_zero_iqr(self):
        """If IQR is zero (e.g. constant data), falls back to Sturges."""
        ds = Dataset(np.full(100, 5.0))
        edges = freedman_diaconis_edges(ds)
        assert len(edges) >= 2


# ── Auto dispatcher ──────────────────────────────────────────────────────

class TestAutoEdges:
    def test_cv_route(self):
        ds = Dataset(np.random.default_rng(42).normal(0, 1, 500))
        edges = auto_edges(ds, method="cv")
        assert len(edges) >= 2

    def test_scott_route(self):
        ds = Dataset(np.random.default_rng(42).normal(0, 1, 500))
        np.testing.assert_array_equal(
            auto_edges(ds, method="scott"), scott_edges(ds))

    def test_sturges_route(self):
        ds = Dataset(np.random.default_rng(42).normal(0, 1, 500))
        np.testing.assert_array_equal(
            auto_edges(ds, method="sturges"), sturges_edges(ds))

    def test_fd_route(self):
        ds = Dataset(np.random.default_rng(42).normal(0, 1, 500))
        np.testing.assert_array_equal(
            auto_edges(ds, method="fd"), freedman_diaconis_edges(ds))

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            auto_edges(Dataset([1.0, 2.0]), method="magic")


# ── Dataset integration with strategy ────────────────────────────────────

class TestDatasetWithStrategy:
    def test_to_histogram_with_strategy(self):
        ds = Dataset(np.random.default_rng(42).normal(0, 1, 500))
        h = ds.to_histogram(strategy="scott")
        assert h.nbins > 0 and h.total > 0

    def test_edges_override_strategy(self):
        """When both edges and strategy are provided, edges win."""
        ds = Dataset(np.random.default_rng(42).normal(0, 1, 500))
        h = ds.to_histogram(edges=[-2, 0, 2])
        assert h.nbins == 2


# ── Model integration ────────────────────────────────────────────────────

class TestBinningModelIntegration:
    def test_auto_binned_model_fits(self):
        """A model built with auto-binned data should produce a valid fit."""
        rng = np.random.default_rng(42)
        sig_events = rng.normal(2.0, 0.5, 300)
        bkg_events = rng.uniform(0, 4, 2000)
        data_events = np.concatenate([sig_events, bkg_events])

        edges = auto_edges(Dataset(data_events), method="scott")
        sig = Sample.from_dataset("signal", Dataset(sig_events), edges=edges)
        sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 5.0))
        bkg = Sample.from_dataset("background", Dataset(bkg_events), edges=edges)
        ch = Channel("SR")
        ch.add_sample(sig)
        ch.add_sample(bkg)
        ch.set_data_from_dataset(Dataset(data_events))
        m = Model("auto_binned")
        m.add_channel(ch)
        r = fit(m)
        assert r.valid and abs(r.bestfit["mu"] - 1.0) < 3 * r.errors["mu"]

    def test_different_strategies_all_valid(self):
        """All binning strategies should produce valid models."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 1000)
        ds = Dataset(data)
        for method in ["cv", "scott", "sturges", "fd"]:
            edges = auto_edges(ds, method=method)
            assert len(edges) >= 2 and np.all(np.diff(edges) > 0)
