"""Interpolation schemes for systematic uncertainty morphing.

These implement the standard HistFactory interpolation codes that map a
nuisance parameter *alpha* (centred at 0) and up/down scale factors into
a multiplicative modifier.  All functions are vectorised over the bin
axis so they work with full histogram arrays.

Interpolation codes
-------------------
0 : Piecewise linear
    Linear in alpha, discontinuous first derivative at alpha=0.
1 : Piecewise exponential  (default)
    Exponential in alpha, smooth everywhere, always positive.
2 : Quadratic interpolation / linear extrapolation
    Parabolic near alpha=0 for smoothness, linear for |alpha|>1.
3 : Polynomial interpolation / exponential extrapolation  (6th order)
    A degree-6 polynomial for |alpha| <= 1, exponential outside.
    This is the InterpCode=4 from ROOT HistFactory.

References
----------
Cranmer, K., Lewis, G., Moneta, L., Shibata, A., & Verkerke, W. (2012).
HistFactory: A tool for creating statistical models for use with RooFit
and RooStats. (CERN-OPEN-2012-016)
"""

from __future__ import annotations

import enum

import numpy as np
from numpy.typing import ArrayLike


class InterpolationCode(enum.IntEnum):
    """Available interpolation schemes."""

    PIECEWISE_LINEAR = 0
    PIECEWISE_EXPONENTIAL = 1
    QUADRATIC_LINEAR = 2
    POLY_EXPONENTIAL = 3


def interpolate(
    alpha: float,
    nominal: ArrayLike,
    down: ArrayLike,
    up: ArrayLike,
    code: InterpolationCode = InterpolationCode.PIECEWISE_EXPONENTIAL,
) -> np.ndarray:
    """Compute the interpolated scale factor(s).

    Parameters
    ----------
    alpha : float
        The nuisance parameter value. 0 = nominal, +1 = up variation,
        -1 = down variation.
    nominal : array_like
        Nominal values (scalar or per-bin array).
    down : array_like
        Down-variation values (at alpha = -1).
    up : array_like
        Up-variation values (at alpha = +1).
    code : InterpolationCode
        Which interpolation scheme to use.

    Returns
    -------
    numpy.ndarray
        Interpolated values, same shape as *nominal*.
    """
    nom = np.asarray(nominal, dtype=np.float64)
    d = np.asarray(down, dtype=np.float64)
    u = np.asarray(up, dtype=np.float64)

    code = InterpolationCode(code)

    if code == InterpolationCode.PIECEWISE_LINEAR:
        return _piecewise_linear(alpha, nom, d, u)
    elif code == InterpolationCode.PIECEWISE_EXPONENTIAL:
        return _piecewise_exponential(alpha, nom, d, u)
    elif code == InterpolationCode.QUADRATIC_LINEAR:
        return _quadratic_linear(alpha, nom, d, u)
    elif code == InterpolationCode.POLY_EXPONENTIAL:
        return _poly_exponential(alpha, nom, d, u)
    else:
        raise ValueError(f"Unknown interpolation code: {code}")


# ---------------------------------------------------------------------------
# Scheme 0: piecewise linear
# ---------------------------------------------------------------------------

def _piecewise_linear(
    alpha: float, nom: np.ndarray, down: np.ndarray, up: np.ndarray
) -> np.ndarray:
    if alpha >= 0:
        return nom + alpha * (up - nom)
    else:
        return nom - alpha * (down - nom)


# ---------------------------------------------------------------------------
# Scheme 1: piecewise exponential
# ---------------------------------------------------------------------------

def _piecewise_exponential(
    alpha: float, nom: np.ndarray, down: np.ndarray, up: np.ndarray
) -> np.ndarray:
    # Protect against non-positive values by falling back to linear
    safe_up = np.where((up > 0) & (nom > 0), up, nom)
    safe_down = np.where((down > 0) & (nom > 0), down, nom)

    if alpha >= 0:
        return nom * np.power(safe_up / nom, alpha)
    else:
        return nom * np.power(safe_down / nom, -alpha)


# ---------------------------------------------------------------------------
# Scheme 2: quadratic interpolation / linear extrapolation
# ---------------------------------------------------------------------------

def _quadratic_linear(
    alpha: float, nom: np.ndarray, down: np.ndarray, up: np.ndarray
) -> np.ndarray:
    a = 0.5 * (up + down) - nom
    b = 0.5 * (up - down)

    if alpha > 1.0:
        return nom + (b + 2.0 * a) * (alpha - 1.0) + (up - nom)
    elif alpha < -1.0:
        return nom + (b - 2.0 * a) * (alpha + 1.0) + (down - nom)
    else:
        return nom + a * alpha * alpha + b * alpha


# ---------------------------------------------------------------------------
# Scheme 3: 6th-order polynomial interpolation / exponential extrapolation
# ---------------------------------------------------------------------------

def _poly_exponential(
    alpha: float, nom: np.ndarray, down: np.ndarray, up: np.ndarray
) -> np.ndarray:
    if alpha >= 1.0:
        return nom * np.power(np.where(up > 0, up, nom) / nom, alpha)
    elif alpha <= -1.0:
        return nom * np.power(np.where(down > 0, down, nom) / nom, -alpha)
    else:
        return _poly6_interp(alpha, nom, down, up)


def _poly6_interp(
    alpha: float, nom: np.ndarray, down: np.ndarray, up: np.ndarray
) -> np.ndarray:
    """Degree-6 polynomial matching exponential behaviour at alpha = +/-1.

    This provides C2-smooth interpolation in [-1, 1] that matches the
    exponential extrapolation and its first two derivatives at the
    boundaries.
    """
    # Compute logarithmic derivatives, protecting against non-positive values
    with np.errstate(divide="ignore", invalid="ignore"):
        log_up = np.where((up > 0) & (nom > 0), np.log(up / nom), 0.0)
        log_down = np.where((down > 0) & (nom > 0), np.log(down / nom), 0.0)

    # Symmetric / antisymmetric combinations (at alpha_0 = 1)
    # Value
    s0 = 0.5 * (up / nom + down / nom)
    a0 = 0.5 * (up / nom - down / nom)
    # First derivative
    s1 = 0.5 * (up / nom * log_up + down / nom * log_down)
    a1 = 0.5 * (up / nom * log_up - down / nom * log_down)
    # Second derivative
    s2 = 0.5 * (up / nom * log_up**2 + down / nom * log_down**2)
    a2 = 0.5 * (up / nom * log_up**2 - down / nom * log_down**2)

    # Coefficients of the degree-6 polynomial f(alpha) such that
    # result = nom * f(alpha), where f(0) = 1
    a = (1.0 / 8.0) * (15 * a0 - 7 * s1 + a2)
    b = (1.0 / 8.0) * (-24 + 24 * s0 - 9 * a1 + s2)
    c = (1.0 / 4.0) * (-5 * a0 + 5 * s1 - a2)
    d = (1.0 / 4.0) * (12 - 12 * s0 + 7 * a1 - s2)
    e = (1.0 / 8.0) * (3 * a0 - 3 * s1 + a2)
    f = (1.0 / 8.0) * (-8 + 8 * s0 - 5 * a1 + s2)

    # Evaluate using Horner's method
    x = alpha
    poly = 1.0 + x * (a + x * (b + x * (c + x * (d + x * (e + x * f)))))
    return nom * poly
