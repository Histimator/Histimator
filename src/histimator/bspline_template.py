"""B-spline template: smooth density estimation via P-splines.

Implements the penalized B-spline approach of Eilers and Marx (1996)
for estimating a smooth rate function from binned count data.  The
log-rate is expanded in a cubic B-spline basis with equally spaced
knots, and a second-order difference penalty on the coefficients
controls roughness.

The model for bin j with centre x_j and width w_j is:

    n_j ~ Poisson(mu_j)
    mu_j = N * f(x_j) * w_j
    log f(x_j) = sum_k B_k(x_j) * theta_k

where N is total expected events, B_k are cubic B-splines, and
theta_k are the coefficients to be estimated.

The penalized log-likelihood is:

    ell(theta) = sum_j [n_j * eta_j - exp(eta_j)] - (lambda/2) * theta^T P theta

where eta_j = sum_k B_k(x_j) * theta_k and P = D_d^T D_d is the
penalty matrix (D_d is the d-th order difference matrix).

The smoothing parameter lambda is selected by minimizing AIC:

    AIC(lambda) = -2 * ell(theta_hat) + 2 * tr(H)

where H is the hat matrix at the penalized MLE.

No dependencies beyond numpy and scipy.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import solve

from histimator.histograms import Histogram


def _bspline_basis(x: np.ndarray, knots: np.ndarray, degree: int = 3) -> np.ndarray:
    """Evaluate B-spline basis matrix at points x.

    Returns matrix B of shape (len(x), n_basis) where
    n_basis = len(knots) - degree - 1.
    """
    n_basis = len(knots) - degree - 1
    b = np.zeros((len(x), n_basis))
    for k in range(n_basis):
        coeffs = np.zeros(n_basis)
        coeffs[k] = 1.0
        spline = BSpline(knots, coeffs, degree, extrapolate=False)
        vals = spline(x)
        vals[np.isnan(vals)] = 0.0
        b[:, k] = vals
    return b


def _difference_matrix(n: int, order: int = 2) -> np.ndarray:
    """Construct the d-th order difference matrix of size (n-d) x n."""
    d = np.eye(n)
    for _ in range(order):
        d = np.diff(d, axis=0)
    return d


def _fit_pspline(
    counts: np.ndarray,
    basis: np.ndarray,
    penalty_matrix: np.ndarray,
    lam: float,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> np.ndarray:
    """Fit P-spline via IRLS for Poisson penalized likelihood.

    Parameters
    ----------
    counts : (m,) array
        Observed bin counts.
    basis : (m, p) array
        B-spline basis evaluated at bin centres.
    penalty_matrix : (p, p) array
        D^T D penalty matrix.
    lam : float
        Smoothing parameter.
    max_iter : int
        Maximum IRLS iterations.
    tol : float
        Convergence tolerance on coefficient change.

    Returns
    -------
    theta : (p,) array
        Estimated B-spline coefficients.
    """
    m, p = basis.shape
    # Initialize at log of mean count (safe starting point)
    mean_count = max(counts.mean(), 1e-10)
    theta = np.full(p, np.log(mean_count) / max(basis.sum(axis=1).mean(), 1.0))

    for _iteration in range(max_iter):
        eta = basis @ theta
        mu = np.exp(np.clip(eta, -20, 20))

        # Working response and weights for IRLS
        w = mu
        z = eta + (counts - mu) / np.maximum(mu, 1e-10)

        # Penalized normal equations: (B^T W B + lam*P) theta = B^T W z
        wdi = np.diag(w)
        lhs = basis.T @ wdi @ basis + lam * penalty_matrix
        rhs = basis.T @ (w * z)

        theta_new = solve(lhs, rhs, assume_a="pos")

        if np.max(np.abs(theta_new - theta)) < tol:
            theta = theta_new
            break
        theta = theta_new

    return theta


def _aic(counts: np.ndarray, basis: np.ndarray, theta: np.ndarray,
         penalty_matrix: np.ndarray, lam: float) -> float:
    """Compute AIC for P-spline model.

    AIC = -2 * log-likelihood + 2 * effective_df
    effective_df = trace of hat matrix H = B (B^T W B + lam*P)^{-1} B^T W
    """
    eta = basis @ theta
    mu = np.exp(np.clip(eta, -20, 20))

    # Poisson log-likelihood (up to constant)
    ll = np.sum(counts * eta - mu)

    # Effective degrees of freedom
    wdi = np.diag(mu)
    btwb = basis.T @ wdi @ basis
    h_inner = solve(btwb + lam * penalty_matrix, btwb, assume_a="pos")
    edf = np.trace(h_inner)

    return -2.0 * ll + 2.0 * edf


class BSplineTemplate:
    """Smooth template using penalized B-spline density estimation.

    Fits a smooth rate function to histogram bin counts using the
    P-spline approach (Eilers & Marx 1996).  The smoothing parameter
    is selected by AIC over a logarithmic grid.

    Parameters
    ----------
    histogram : Histogram
        Binned data to smooth.
    n_knots : int
        Number of interior knots for the B-spline basis.
        More knots allow more flexibility; the penalty prevents
        overfitting regardless of knot count.
    penalty_order : int
        Order of the difference penalty (2 = roughness penalty on
        second differences, the standard choice).
    lam : float or None
        Fixed smoothing parameter.  If None, selected by AIC.
    """

    def __init__(
        self,
        histogram: Histogram,
        n_knots: int = 20,
        penalty_order: int = 2,
        lam: float | None = None,
    ) -> None:
        if not isinstance(histogram, Histogram):
            raise TypeError(
                f"BSplineTemplate requires a Histogram, got {type(histogram).__name__}"
            )
        self._histogram = histogram
        self._edges = histogram.edges.copy()
        self._nbins = histogram.nbins
        self._total = histogram.total

        centres = 0.5 * (self._edges[:-1] + self._edges[1:])
        widths = np.diff(self._edges)
        counts = histogram.values.copy()

        # Construct B-spline basis
        lo, hi = self._edges[0], self._edges[-1]
        degree = 3
        interior_knots = np.linspace(lo, hi, n_knots + 2)[1:-1]
        self._knots = np.concatenate([
            np.full(degree + 1, lo),
            interior_knots,
            np.full(degree + 1, hi),
        ])
        n_basis = len(self._knots) - degree - 1

        basis = _bspline_basis(centres, self._knots, degree)
        self._basis = basis

        # Difference penalty matrix
        d = _difference_matrix(n_basis, penalty_order)
        p = d.T @ d
        self._penalty_matrix = p

        # Select lambda by AIC if not provided
        if lam is None:
            lam_grid = np.logspace(-2, 8, 30)
            best_aic = np.inf
            best_lam = 1.0
            for trial_lam in lam_grid:
                try:
                    theta_trial = _fit_pspline(counts, basis, p, trial_lam)
                    aic_val = _aic(counts, basis, theta_trial, p, trial_lam)
                    if aic_val < best_aic:
                        best_aic = aic_val
                        best_lam = trial_lam
                except (np.linalg.LinAlgError, FloatingPointError):
                    continue
            lam = best_lam

        self._lam = lam
        self._theta = _fit_pspline(counts, basis, p, lam)

        # Compute fitted values at bin centres
        eta = basis @ self._theta
        self._fitted_rate = np.exp(np.clip(eta, -20, 20))

        # Normalize: fitted rate * widths should sum to total
        raw_integral = np.sum(self._fitted_rate * widths)
        if raw_integral > 0:
            self._scale = self._total / raw_integral
        else:
            self._scale = 1.0

        self._fitted_counts = self._fitted_rate * self._scale * widths
        self._fitted_density = (
            self._fitted_rate * self._scale / self._total
            if self._total > 0 else np.zeros(self._nbins)
            )

        # Store sumw2 from original histogram for uncertainty propagation
        self._sumw2 = histogram.sumw2.copy()

    @property
    def histogram(self) -> Histogram:
        """Access to the underlying Histogram (pre-smoothing)."""
        return self._histogram

    @property
    def edges(self) -> np.ndarray:
        return self._edges

    @property
    def nbins(self) -> int:
        return self._nbins

    @property
    def sumw2(self) -> np.ndarray:
        return self._sumw2

    @property
    def smoothing_parameter(self) -> float:
        return self._lam

    @property
    def coefficients(self) -> np.ndarray:
        return self._theta.copy()

    def counts(self) -> np.ndarray:
        """Smoothed expected counts per bin."""
        return self._fitted_counts.copy()

    def density(self) -> np.ndarray:
        """Smooth probability density at bin centres.

        Integrates to 1: sum(density * widths) = 1.
        """
        if self._total == 0:
            return np.zeros(self._nbins)
        widths = np.diff(self._edges)
        d = self._fitted_counts / (self._total * widths)
        return d

    def evaluate_density(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the smooth density at arbitrary points.

        Unlike BinnedTemplate's piecewise constant evaluation, the
        B-spline density is smooth and differentiable everywhere
        within the domain.
        """
        x = np.asarray(x, dtype=np.float64)
        basis_x = _bspline_basis(x, self._knots, 3)
        eta = basis_x @ self._theta
        rate = np.exp(np.clip(eta, -20, 20))
        if self._total > 0:
            return rate * self._scale / self._total
        return np.zeros_like(x)

    def __repr__(self) -> str:
        return (
            f"BSplineTemplate(nbins={self._nbins}, "
            f"lambda={self._lam:.2g}, "
            f"total={self._total:.6g})"
        )
