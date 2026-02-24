"""Observation model abstraction for the GP Laplace inference core.

This module extracts the observation-model-specific computation from
the Newton solver in ``gp_template._laplace_mode``, so that the same
solver can drive both the existing binned Poisson path and the new
hybrid Cox point-process path.

Architecture
------------
``ObservationModel`` is a runtime protocol with three methods that
describe how the data contribute to the posterior:

- ``gradient(f)``           data-term gradient of log p(data | f)
- ``hessian_diagonal(f)``   diagonal of the negative Hessian of log p(data | f)
- ``log_likelihood(f)``     log p(data | f), evaluated at the current f

``_cox_laplace_mode`` is a Newton solver that accepts any object
satisfying that protocol, combining the data gradient with the GP prior
gradient at each step.  The existing ``_laplace_mode`` in
``gp_template.py`` is not touched; ``PoissonCountModel`` re-implements
its logic under the new interface and is validated against that function
in the test suite.

``HybridCoxModel`` implements the Cox point-process likelihood, where
the GP is evaluated at N exact event positions (sum term) and M
quadrature points (integral term).  This is the theoretically superior
approach described in the sprint plan: event positions supply
fine-grained positional information that grid-based binning discards.

Gradient and Hessian for the hybrid Cox likelihood
--------------------------------------------------
The log-likelihood is

    L(f) = sum_{i=1}^N f(x_i) - sum_{j=1}^M w_j exp(f(s_j))

where x_i are event positions and (s_j, w_j) are quadrature nodes and
weights approximating the integral of the rate over the domain.

Gradient dL/df_k:
    k in event block (0..N-1):     +1
    k in quadrature block (N..N+M-1): -w_j exp(f(s_j))

Hessian diagonal -d^2L/df_k^2:
    k in event block:              0
    k in quadrature block:         w_j exp(f(s_j))

The zero diagonal in the event block means the Newton matrix
B = K^{-1} + W has curvature from the GP prior K^{-1} alone at
event locations, which is correct: the GP interpolates smoothly
between event positions without a local Poisson constraint there.

Quadrature
----------
Gauss-Legendre nodes and weights are computed by
``numpy.polynomial.legendre.leggauss`` on [-1, 1] and mapped to
[x_min, x_max].  M = 30 to 60 points suffice for smooth rate
functions with lengthscales that are O(domain / 10) or larger.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class ObservationModel(Protocol):
    """Runtime-checkable protocol for GP observation models.

    Any object implementing these three methods can be passed to
    ``_cox_laplace_mode`` and ``_cox_laplace_log_marginal``.
    """

    def gradient(self, f: np.ndarray) -> np.ndarray:
        """Data-term gradient of log p(data | f).

        Parameters
        ----------
        f : (n,) array
            Current log-rate values at training locations.

        Returns
        -------
        grad : (n,) array
            d/df log p(data | f).
        """
        ...

    def hessian_diagonal(self, f: np.ndarray) -> np.ndarray:
        """Diagonal of the *negative* Hessian of log p(data | f).

        The Newton matrix is B = K^{-1} + W where W = diag(returned
        values).  All returned values must be non-negative.

        Parameters
        ----------
        f : (n,) array
            Current log-rate values.

        Returns
        -------
        w_diag : (n,) array
            Non-negative diagonal entries of -d^2/df^2 log p(data | f).
        """
        ...

    def log_likelihood(self, f: np.ndarray) -> float:
        """Evaluate log p(data | f) at the given f.

        Used to compute the data term in the Laplace approximation to
        the log marginal likelihood.

        Parameters
        ----------
        f : (n,) array
            Log-rate values (typically the posterior mode f_hat).

        Returns
        -------
        float
        """
        ...


# ---------------------------------------------------------------------------
# Concrete observation models
# ---------------------------------------------------------------------------

class PoissonCountModel:
    """Poisson observation model for binned count data.

    Implements the same logic as the existing ``_laplace_mode`` in
    ``gp_template.py``, re-expressed under the ``ObservationModel``
    interface so that the Newton solver is observation-model-agnostic.

    Parameters
    ----------
    counts : (m,) array
        Observed bin counts.
    widths : (m,) array
        Bin widths (each bin j contributes mu_j = exp(f_j) * w_j).
    """

    def __init__(self, counts: np.ndarray, widths: np.ndarray) -> None:
        self._counts = np.asarray(counts, dtype=np.float64)
        self._widths = np.asarray(widths, dtype=np.float64)

    def _mu(self, f: np.ndarray) -> np.ndarray:
        return np.exp(np.clip(f, -20, 20)) * self._widths

    def gradient(self, f: np.ndarray) -> np.ndarray:
        return self._counts - self._mu(f)

    def hessian_diagonal(self, f: np.ndarray) -> np.ndarray:
        return self._mu(f)

    def log_likelihood(self, f: np.ndarray) -> float:
        f_clipped = np.clip(f, -20, 20)
        mu = np.exp(f_clipped) * self._widths
        ll = np.sum(self._counts * f_clipped - mu)
        return float(ll)


class HybridCoxModel:
    """Hybrid Cox point-process observation model.

    Evaluates the log-Gaussian Cox process likelihood using exact event
    positions for the sum term and quadrature for the integral term.

    The augmented training location vector expected by ``_cox_laplace_mode``
    is X_aug = [x_1, ..., x_N, s_1, ..., s_M], where x_i are the N event
    positions and (s_j, w_j) are M quadrature nodes and weights.  All
    methods expect f to be evaluated on this augmented vector.

    Parameters
    ----------
    n_events : int
        Number of observed events N.  The first n_events entries of f
        correspond to event positions.
    quad_weights : (M,) array
        Quadrature weights w_j.  The last M entries of f correspond to
        quadrature nodes.
    """

    def __init__(self, n_events: int, quad_weights: np.ndarray) -> None:
        self._n = n_events
        self._qw = np.asarray(quad_weights, dtype=np.float64)

    def _rate_at_quad(self, f: np.ndarray) -> np.ndarray:
        """exp(f) at quadrature nodes, clipped for numerical safety."""
        return np.exp(np.clip(f[self._n:], -20, 20))

    def gradient(self, f: np.ndarray) -> np.ndarray:
        grad = np.empty_like(f)
        grad[: self._n] = 1.0
        grad[self._n:] = -self._qw * self._rate_at_quad(f)
        return grad

    def hessian_diagonal(self, f: np.ndarray) -> np.ndarray:
        w_diag = np.zeros_like(f)
        w_diag[self._n:] = self._qw * self._rate_at_quad(f)
        return w_diag

    def log_likelihood(self, f: np.ndarray) -> float:
        sum_term = float(np.sum(f[: self._n]))
        integral_term = float(np.sum(self._qw * self._rate_at_quad(f)))
        return sum_term - integral_term


# ---------------------------------------------------------------------------
# Quadrature helper
# ---------------------------------------------------------------------------

def gauss_legendre_quadrature(
    x_min: float, x_max: float, n_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights mapped to [x_min, x_max].

    Parameters
    ----------
    x_min, x_max : float
        Domain boundaries.
    n_points : int
        Number of quadrature points.

    Returns
    -------
    nodes : (n_points,) array
        Quadrature nodes in [x_min, x_max].
    weights : (n_points,) array
        Corresponding weights (already scaled by (x_max - x_min) / 2).
    """
    t, w = np.polynomial.legendre.leggauss(n_points)
    half_range = 0.5 * (x_max - x_min)
    mid = 0.5 * (x_min + x_max)
    nodes = mid + half_range * t
    weights = half_range * w
    return nodes, weights


# ---------------------------------------------------------------------------
# Newton solver
# ---------------------------------------------------------------------------

def _cox_laplace_mode(
    obs_model: ObservationModel,
    k: np.ndarray,
    mean_values: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-6,
    jitter: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the posterior mode via Newton's method for any observation model.

    Solves for the mode of log p(f | data) = log p(data | f) + log p(f)
    using Newton iterations, where log p(f) = N(m, K).

    The Newton update at each step solves

        B @ delta = grad

    where grad = d/df [log p(data | f) - 0.5 (f-m)^T K^{-1} (f-m)]
               = obs_model.gradient(f) - K^{-1} (f - m)

    and B = K^{-1} + diag(obs_model.hessian_diagonal(f)).

    Parameters
    ----------
    obs_model : ObservationModel
        Object supplying gradient and Hessian diagonal of log p(data | f).
    k : (n, n) array
        Prior covariance matrix (possibly augmented with mean function
        uncertainty via K_eff = K + H B H^T).  Must be positive definite.
    mean_values : (n,) array
        Prior mean at all training locations.
    max_iter : int
        Maximum Newton iterations.
    tol : float
        Convergence tolerance on the infinity norm of the Newton step.
    jitter : float
        Small constant added to the diagonal of k for numerical
        stability.  Should match the jitter used when constructing k.

    Returns
    -------
    f_hat : (n,) array
        Posterior mode.
    w_hat : (n,) array
        Hessian diagonal at the mode (obs_model.hessian_diagonal(f_hat)).
        Returned so that callers can compute B = K^{-1} + diag(w_hat)
        without a redundant model evaluation.
    """
    n = len(mean_values)
    k_reg = k + jitter * np.eye(n)

    try:
        l_k = cho_factor(k_reg)
        k_inv = cho_solve(l_k, np.eye(n))
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "_cox_laplace_mode: kernel matrix is not positive definite "
            "after jitter; check hyperparameters."
        ) from exc

    f = mean_values.copy()

    for _iteration in range(max_iter):
        obs_grad = obs_model.gradient(f)
        w_diag = obs_model.hessian_diagonal(f)

        prior_grad = k_inv @ (f - mean_values)
        grad = obs_grad - prior_grad

        b = k_inv + np.diag(w_diag)
        b += 1e-8 * np.eye(len(b))

        try:
            delta = solve(b, grad, assume_a="pos")
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(b, grad, rcond=None)[0]

        f = f + delta

        if np.max(np.abs(delta)) < tol:
            break

    w_hat = obs_model.hessian_diagonal(f)
    return f, w_hat


# ---------------------------------------------------------------------------
# Laplace log marginal likelihood
# ---------------------------------------------------------------------------

def _cox_laplace_log_marginal(
    obs_model: ObservationModel,
    k: np.ndarray,
    mean_values: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-6,
    jitter: float = 1e-6,
) -> float:
    """Laplace approximation to the log marginal likelihood.

    Computes log p(data | theta) approximately as

        log p(data | f_hat) + log p(f_hat) - 0.5 * log |B|

    where B = K^{-1} + W is the Hessian of the negative log posterior
    and log p(f_hat) = -0.5 (f_hat - m)^T K^{-1} (f_hat - m) - 0.5 log|K|.

    Unlike the existing ``_laplace_log_marginal`` in gp_template.py,
    this function computes log|B| directly via Cholesky of B rather
    than via the W^{1/2} K W^{1/2} trick, because the HybridCoxModel
    has zeros in the event block of W, making the square-root trick
    ill-conditioned.

    This function is the foundation for hyperparameter optimization in
    the hybrid Cox path.  Gradients with respect to kernel hyperparameters
    flow through dK/d(theta), which is independent of the observation
    model.

    Parameters
    ----------
    obs_model : ObservationModel
        Observation model providing gradient, Hessian diagonal, and
        log-likelihood.
    k : (n, n) array
        Prior covariance matrix (possibly augmented).
    mean_values : (n,) array
        Prior mean at all training locations.
    max_iter, tol, jitter :
        Passed through to ``_cox_laplace_mode``.

    Returns
    -------
    float
        Laplace approximation to log p(data | theta), or -inf if any
        linear algebra step fails.
    """
    n = len(mean_values)
    k_reg = k + jitter * np.eye(n)

    try:
        f_hat, w_hat = _cox_laplace_mode(obs_model, k_reg, mean_values,
                                         max_iter=max_iter, tol=tol,
                                         jitter=0.0)
    except np.linalg.LinAlgError:
        return -np.inf

    try:
        l_k = cho_factor(k_reg)
        alpha = cho_solve(l_k, f_hat - mean_values)
        log_det_k = 2.0 * np.sum(np.log(np.diag(l_k[0])))
    except np.linalg.LinAlgError:
        return -np.inf

    log_lik = obs_model.log_likelihood(f_hat)
    log_prior = -0.5 * float((f_hat - mean_values) @ alpha) - 0.5 * log_det_k

    k_inv = cho_solve(l_k, np.eye(n))
    b = k_inv + np.diag(w_hat)

    try:
        l_b = cho_factor(b)
        log_det_b = 2.0 * np.sum(np.log(np.diag(l_b[0])))
    except np.linalg.LinAlgError:
        return -np.inf

    return float(log_lik + log_prior - 0.5 * log_det_b)
