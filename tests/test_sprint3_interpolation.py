"""Tests for Sprint 3: Interpolation schemes."""

import numpy as np
import pytest

from histimator.interpolation import InterpolationCode, interpolate

# All schemes must satisfy these fundamental boundary conditions.
SCHEMES = list(InterpolationCode)

NOM = np.array([100.0, 200.0, 50.0])
UP = np.array([110.0, 220.0, 55.0])    # +10%
DOWN = np.array([90.0, 180.0, 45.0])   # -10%


class TestBoundaryConditions:
    """Every scheme must return nominal at alpha=0, up at alpha=+1, down at alpha=-1."""

    @pytest.mark.parametrize("code", SCHEMES)
    def test_alpha_zero_returns_nominal(self, code):
        result = interpolate(0.0, NOM, DOWN, UP, code)
        np.testing.assert_allclose(result, NOM, rtol=1e-12)

    @pytest.mark.parametrize("code", SCHEMES)
    def test_alpha_plus_one_returns_up(self, code):
        result = interpolate(1.0, NOM, DOWN, UP, code)
        np.testing.assert_allclose(result, UP, rtol=1e-10)

    @pytest.mark.parametrize("code", SCHEMES)
    def test_alpha_minus_one_returns_down(self, code):
        result = interpolate(-1.0, NOM, DOWN, UP, code)
        np.testing.assert_allclose(result, DOWN, rtol=1e-10)


class TestPiecewiseLinear:
    """Scheme 0 specific tests."""

    def test_extrapolation_positive(self):
        result = interpolate(2.0, NOM, DOWN, UP, InterpolationCode.PIECEWISE_LINEAR)
        # At alpha=2: nom + 2*(up - nom) = 100 + 2*10 = 120 etc.
        np.testing.assert_allclose(result, [120.0, 240.0, 60.0])

    def test_extrapolation_negative(self):
        result = interpolate(-2.0, NOM, DOWN, UP, InterpolationCode.PIECEWISE_LINEAR)
        # At alpha=-2: nom + 2*(nom - down) = 100 + 2*10 = 80... wait
        # nom - (-2)*(down - nom) = nom + 2*(down - nom) = 100 + 2*(-10) = 80
        np.testing.assert_allclose(result, [80.0, 160.0, 40.0])

    def test_scalar_input(self):
        result = interpolate(0.5, 100.0, 90.0, 110.0, InterpolationCode.PIECEWISE_LINEAR)
        np.testing.assert_allclose(result, 105.0)


class TestPiecewiseExponential:
    """Scheme 1 specific tests."""

    def test_always_positive(self):
        """Exponential interpolation should never go negative."""
        result = interpolate(
            -5.0, NOM, DOWN, UP, InterpolationCode.PIECEWISE_EXPONENTIAL
        )
        assert np.all(result > 0)

    def test_symmetric_10pct(self):
        """For symmetric +/-10% at alpha=+/-1, check alpha=0.5 is geometric mean."""
        result = interpolate(
            0.5, 100.0, 90.0, 110.0, InterpolationCode.PIECEWISE_EXPONENTIAL
        )
        expected = 100.0 * (110.0 / 100.0) ** 0.5
        np.testing.assert_allclose(result, expected, rtol=1e-12)


class TestQuadraticLinear:
    """Scheme 2 specific tests."""

    def test_smooth_at_zero(self):
        """The quadratic interpolation should be smooth (no kink) at alpha=0.

        We check that a small positive and small negative alpha give
        nearly equal absolute deviations from nominal.
        """
        eps = 1e-4
        r_plus = interpolate(eps, NOM, DOWN, UP, InterpolationCode.QUADRATIC_LINEAR)
        r_minus = interpolate(-eps, NOM, DOWN, UP, InterpolationCode.QUADRATIC_LINEAR)
        # For symmetric variations, the quadratic term dominates near 0,
        # so the deviations should be nearly equal (both positive for parabola)
        diff_plus = np.abs(r_plus - NOM)
        diff_minus = np.abs(r_minus - NOM)
        np.testing.assert_allclose(diff_plus, diff_minus, atol=1e-6)

    def test_linear_extrapolation_above_1(self):
        """For alpha > 1, the function should be linear."""
        r2 = interpolate(2.0, NOM, DOWN, UP, InterpolationCode.QUADRATIC_LINEAR)
        r3 = interpolate(3.0, NOM, DOWN, UP, InterpolationCode.QUADRATIC_LINEAR)
        r4 = interpolate(4.0, NOM, DOWN, UP, InterpolationCode.QUADRATIC_LINEAR)
        # Check constant slope: (r3 - r2) == (r4 - r3)
        np.testing.assert_allclose(r3 - r2, r4 - r3, rtol=1e-12)


class TestPolyExponential:
    """Scheme 3 specific tests."""

    def test_continuity_at_boundary(self):
        """The polynomial must match the exponential at alpha = +/-1."""
        # Test at a point just inside and just outside
        r_inside = interpolate(0.999, NOM, DOWN, UP, InterpolationCode.POLY_EXPONENTIAL)
        r_boundary = interpolate(1.0, NOM, DOWN, UP, InterpolationCode.POLY_EXPONENTIAL)
        r_outside = interpolate(1.001, NOM, DOWN, UP, InterpolationCode.POLY_EXPONENTIAL)
        # Should be nearly continuous
        np.testing.assert_allclose(r_inside, r_boundary, rtol=1e-3)
        np.testing.assert_allclose(r_outside, r_boundary, rtol=1e-3)

    def test_always_positive(self):
        result = interpolate(
            -3.0, NOM, DOWN, UP, InterpolationCode.POLY_EXPONENTIAL
        )
        assert np.all(result > 0)


class TestEdgeCases:
    """Edge cases all schemes should handle."""

    @pytest.mark.parametrize("code", SCHEMES)
    def test_zero_variation(self, code):
        """If up == down == nominal, result should be nominal for any alpha."""
        result = interpolate(2.5, NOM, NOM, NOM, code)
        np.testing.assert_allclose(result, NOM, rtol=1e-10)

    @pytest.mark.parametrize("code", SCHEMES)
    def test_scalar_nominal(self, code):
        """Works with scalar inputs."""
        result = interpolate(0.0, 100.0, 90.0, 110.0, code)
        np.testing.assert_allclose(result, 100.0, rtol=1e-10)
