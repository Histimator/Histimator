"""Tests for Sprint 2: Parameter and Histogram core types."""

import numpy as np
import pytest

from histimator.histograms import Histogram
from histimator.parameters import Parameter

# ===========================================================================
# Parameter tests
# ===========================================================================

class TestParameter:
    def test_basic_construction(self):
        p = Parameter("mu", value=1.0, bounds=(0.0, 3.0))
        assert p.name == "mu"
        assert p.value == 1.0
        assert p.bounds == (0.0, 3.0)
        assert p.fixed is False

    def test_defaults(self):
        p = Parameter("x")
        assert p.value == 1.0
        assert p.bounds is None
        assert p.fixed is False

    def test_bounds_validation_inverted(self):
        with pytest.raises(ValueError, match="lower bound"):
            Parameter("bad", value=1.0, bounds=(5.0, 2.0))

    def test_bounds_validation_equal(self):
        with pytest.raises(ValueError, match="lower bound"):
            Parameter("bad", value=1.0, bounds=(2.0, 2.0))

    def test_value_outside_bounds(self):
        with pytest.raises(ValueError, match="outside bounds"):
            Parameter("bad", value=10.0, bounds=(0.0, 3.0))

    def test_clone(self):
        p = Parameter("mu", value=1.0, bounds=(0.0, 3.0))
        p2 = p.clone(value=2.0)
        assert p2.value == 2.0
        assert p2.name == "mu"
        assert p.value == 1.0  # original unchanged

    def test_to_minuit_init(self):
        p = Parameter("mu", value=1.5, bounds=(0.0, 3.0))
        d = p.to_minuit_init()
        assert d["mu"] == 1.5
        assert d["limit_mu"] == (0.0, 3.0)

    def test_to_minuit_init_fixed(self):
        p = Parameter("mu", value=1.0, fixed=True)
        d = p.to_minuit_init()
        assert d["fix_mu"] is True

    def test_to_minuit_init_unbounded(self):
        p = Parameter("mu", value=1.0)
        d = p.to_minuit_init()
        assert "limit_mu" not in d


# ===========================================================================
# Histogram tests
# ===========================================================================

class TestHistogram:
    def test_basic_construction(self):
        h = Histogram([10, 20, 30], [0.0, 1.0, 2.0, 3.0])
        assert h.nbins == 3
        assert h.total == 60.0

    def test_edges_values_mismatch(self):
        with pytest.raises(ValueError, match="edges length"):
            Histogram([10, 20], [0.0, 1.0, 2.0, 3.0])

    def test_non_monotonic_edges(self):
        with pytest.raises(ValueError, match="monotonically increasing"):
            Histogram([10, 20], [0.0, 2.0, 1.0])

    def test_1d_validation(self):
        with pytest.raises(ValueError, match="1-D"):
            Histogram([[10, 20], [30, 40]], [0.0, 1.0, 2.0])

    def test_immutability(self):
        h = Histogram([10, 20, 30], [0.0, 1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            h.values[0] = 999

    def test_centres(self):
        h = Histogram([10, 20], [0.0, 2.0, 4.0])
        np.testing.assert_array_equal(h.centres, [1.0, 3.0])

    def test_widths(self):
        h = Histogram([10, 20], [0.0, 1.0, 3.0])
        np.testing.assert_array_equal(h.widths, [1.0, 2.0])

    def test_density(self):
        h = Histogram([10, 20], [0.0, 1.0, 3.0])
        np.testing.assert_array_almost_equal(h.density, [10.0, 10.0])

    def test_integration_full_range(self):
        h = Histogram([10, 20, 30], [0.0, 1.0, 2.0, 3.0])
        assert h.integrate() == pytest.approx(60.0)

    def test_integration_partial_bin(self):
        h = Histogram([10, 20], [0.0, 1.0, 2.0])
        # integrate from 0.5 to 1.5: half of bin0 (5) + half of bin1 (10)
        assert h.integrate(0.5, 1.5) == pytest.approx(15.0)

    def test_integration_single_bin(self):
        h = Histogram([10, 20], [0.0, 1.0, 2.0])
        assert h.integrate(0.0, 1.0) == pytest.approx(10.0)

    def test_integration_out_of_range(self):
        h = Histogram([10, 20], [0.0, 1.0, 2.0])
        assert h.integrate(5.0, 10.0) == pytest.approx(0.0)

    def test_from_numpy(self):
        data = np.random.default_rng(42).normal(0, 1, 1000)
        counts, edges = np.histogram(data, bins=20)
        h = Histogram.from_numpy((counts, edges))
        assert h.nbins == 20
        assert h.total == 1000.0

    def test_addition(self):
        h1 = Histogram([10, 20], [0.0, 1.0, 2.0])
        h2 = Histogram([5, 15], [0.0, 1.0, 2.0])
        h3 = h1 + h2
        np.testing.assert_array_equal(h3.values, [15, 35])

    def test_addition_incompatible(self):
        h1 = Histogram([10, 20], [0.0, 1.0, 2.0])
        h2 = Histogram([5, 15], [0.0, 1.0, 3.0])
        with pytest.raises(ValueError, match="identical bin edges"):
            h1 + h2

    def test_scalar_multiply(self):
        h = Histogram([10, 20], [0.0, 1.0, 2.0])
        h2 = h * 2.0
        np.testing.assert_array_equal(h2.values, [20, 40])

    def test_rmul(self):
        h = Histogram([10, 20], [0.0, 1.0, 2.0])
        h2 = 3.0 * h
        np.testing.assert_array_equal(h2.values, [30, 60])

    def test_division_by_scalar(self):
        h = Histogram([10, 20], [0.0, 1.0, 2.0])
        h2 = h / 2.0
        np.testing.assert_array_almost_equal(h2.values, [5, 10])

    def test_division_by_histogram(self):
        h1 = Histogram([10, 20], [0.0, 1.0, 2.0])
        h2 = Histogram([5, 10], [0.0, 1.0, 2.0])
        ratio = h1 / h2
        np.testing.assert_array_almost_equal(ratio.values, [2.0, 2.0])

    def test_division_by_zero_histogram(self):
        h1 = Histogram([10, 20], [0.0, 1.0, 2.0])
        h2 = Histogram([0, 10], [0.0, 1.0, 2.0])
        ratio = h1 / h2
        assert ratio.values[0] == 0.0  # safe division
        assert ratio.values[1] == 2.0

    def test_equality(self):
        h1 = Histogram([10, 20], [0.0, 1.0, 2.0])
        h2 = Histogram([10, 20], [0.0, 1.0, 2.0])
        assert h1 == h2

    def test_inequality(self):
        h1 = Histogram([10, 20], [0.0, 1.0, 2.0])
        h2 = Histogram([10, 25], [0.0, 1.0, 2.0])
        assert h1 != h2

    def test_repr(self):
        h = Histogram([10, 20, 30], [0.0, 1.0, 2.0, 3.0])
        r = repr(h)
        assert "nbins=3" in r
        assert "total=60" in r

    def test_sumw2_custom(self):
        h = Histogram([10, 20], [0.0, 1.0, 2.0], sumw2=[100, 400])
        np.testing.assert_array_equal(h.sumw2, [100, 400])

    def test_sumw2_default_poisson(self):
        h = Histogram([10, 20], [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(h.sumw2, [10, 20])
