"""Gaussian process template: smooth density estimation via log-GP.

Implements a log-Gaussian Cox process model for binned count data,
following the framework of Moller, Syversveen and Waagepetersen (1998).

The generative model for bin j with centre x_j and width w_j is:

    n_j ~ Poisson(mu_j)
    mu_j = exp(f(x_j)) * w_j
    f ~ GP(m(x), k(x, x'))

where f is the log-rate function modeled as a Gaussian process with
mean function m and covariance function k.

Mean function treatment
-----------------------
The default mean function uses a B-spline basis in normalised
coordinates: m(x) = H(x)^T beta where H contains cubic B-spline
evaluations.  We place a broad Gaussian prior beta ~ N(beta_hat, B)
and absorb it into the kernel:

    K_eff = K + H B H^T

This augmented kernel propagates uncertainty in the mean function
parameters into the posterior covariance (Rasmussen & Williams 2006,
Sec. 2.7).  B-splines form a partition of unity and can represent
arbitrary smooth shapes, making the GP shape-agnostic: falling
spectra, quadratic rises, broad peaks, and BDT discriminants are
all handled without manual tuning.

Polynomial mean functions (degree 0, 1, 2, ...) are also supported
for backward compatibility.

Inference uses the Laplace approximation (Rasmussen & Williams 2006,
Chapter 3):

1. Find the mode f_hat of the posterior p(f|n) by Newton's method.
2. Approximate p(f|n) as N(f_hat, (K_eff^{-1} + W)^{-1}) where
   W = diag(mu_j) is the Hessian of the negative log-likelihood.
3. Optimize kernel hyperparameters by maximizing the Laplace
   approximation to the log marginal likelihood.

Supported kernels:
- Squared Exponential: k(r) = sigma^2 * exp(-r^2 / (2*ell^2))
- Matern 3/2: k(r) = sigma^2 * (1 + sqrt(3)*r/ell) * exp(-sqrt(3)*r/ell)
- Matern 5/2: k(r) = sigma^2 * (1 + sqrt(5)*r/ell + 5*r^2/(3*ell^2)) * exp(-sqrt(5)*r/ell)

No dependencies beyond numpy and scipy.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve
from scipy.optimize import minimize

from histimator.histograms import Histogram


class GPKernel(Enum):
    """Available GP covariance functions."""
    SQUARED_EXPONENTIAL = "squared_exponential"
    MATERN_32 = "matern_32"
    MATERN_52 = "matern_52"


def _kernel_matrix(
    x1: np.ndarray, x2: np.ndarray,
    kernel: GPKernel,
    log_sigma: float, log_ell: float,
) -> np.ndarray:
    """Compute covariance matrix K(x1, x2).

    Parameters
    ----------
    x1, x2 : (n,) and (m,) arrays
    kernel : GPKernel enum
    log_sigma, log_ell : log-scale hyperparameters

    Returns
    -------
    K : (n, m) array
    """
    sigma2 = np.exp(2 * log_sigma)
    ell = np.exp(log_ell)

    r = np.abs(x1[:, None] - x2[None, :])
    s = r / ell

    if kernel == GPKernel.SQUARED_EXPONENTIAL:
        return sigma2 * np.exp(-0.5 * s**2)
    elif kernel == GPKernel.MATERN_32:
        sqrt3_s = np.sqrt(3.0) * s
        return sigma2 * (1.0 + sqrt3_s) * np.exp(-sqrt3_s)
    elif kernel == GPKernel.MATERN_52:
        sqrt5_s = np.sqrt(5.0) * s
        return sigma2 * (1.0 + sqrt5_s + sqrt5_s**2 / 3.0) * np.exp(-sqrt5_s)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")


def _build_design_matrix(centres: np.ndarray, degree: int,
                         x_min: float, x_max: float) -> np.ndarray:
    """Build design matrix H for polynomial mean in log-rate space.

    Coordinates are normalised to [-1, 1] for numerical stability.

    Parameters
    ----------
    centres : (n,) array of bin centres
    degree : polynomial degree (0 = constant, 1 = linear, ...)
    x_min, x_max : range for normalisation (from edges)

    Returns
    -------
    H : (n, degree+1) array
    """
    span = x_max - x_min
    x_norm = np.zeros_like(centres) if span < 1e-10 else 2.0 * (centres - x_min) / span - 1.0
    return np.column_stack([x_norm**p for p in range(degree + 1)])


def _build_bspline_basis(
    centres: np.ndarray,
    x_min: float,
    x_max: float,
    n_internal: int = 6,
    degree: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Build B-spline design matrix for mean function in log-rate space.

    Uses the R&W Section 2.7 framework: the mean function is
    m(x) = H(x)^T beta where H contains B-spline basis evaluations.
    Unlike polynomials, B-splines form a partition of unity and can
    represent arbitrary smooth shapes with enough knots.

    Parameters
    ----------
    centres : (n,) array of bin centres
    x_min, x_max : domain boundaries (from histogram edges)
    n_internal : number of interior knots (default 6)
    degree : B-spline degree (default 3 = cubic)

    Returns
    -------
    H : (n, n_basis) array, design matrix
    knots : full knot vector (for prediction at new points)
    """
    from scipy.interpolate import BSpline

    internal_knots = np.linspace(x_min, x_max, n_internal + 2)[1:-1]
    knots = np.concatenate([
        np.full(degree + 1, x_min),
        internal_knots,
        np.full(degree + 1, x_max),
    ])
    h = BSpline.design_matrix(centres, knots, degree).toarray()
    return h, knots


def _fit_mean_coefficients(
    counts: np.ndarray,
    widths: np.ndarray,
    h: np.ndarray,
) -> np.ndarray:
    """Estimate mean function coefficients by weighted least squares.

    Fits beta in m(x) = H beta to log(counts/widths), weighting by
    counts so that high-statistics bins dominate.

    Returns
    -------
    beta_hat : (p,) coefficient vector
    """
    safe_counts = np.maximum(counts, 0.5)
    y = np.log(safe_counts / widths)
    w = np.sqrt(safe_counts)
    hw = h * w[:, None]
    yw = y * w
    beta_hat, _, _, _ = np.linalg.lstsq(hw, yw, rcond=None)
    return beta_hat


def _augment_kernel(
    k: np.ndarray,
    h: np.ndarray,
    beta_prior_variance: float,
) -> np.ndarray:
    """Augment kernel with mean function uncertainty: K_eff = K + H B H^T.

    This ensures the posterior covariance accounts for uncertainty in
    the trend, not just uncertainty around the trend (R&W Sec. 2.7).
    """
    p = h.shape[1]
    b = beta_prior_variance * np.eye(p)
    return k + h @ b @ h.T


def _laplace_mode(
    counts: np.ndarray,
    widths: np.ndarray,
    k: np.ndarray,
    mean_values: np.ndarray,
    sumw2: np.ndarray | None = None,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the Laplace mode f_hat of the GP posterior.

    Solves for the mode of log p(f|n) = log p(n|f) + log p(f) + const
    using Newton's method, where p(f) = N(m, K).

    The observation model depends on whether weighted events are present:
    - Unweighted (sumw2 is None): Poisson(n_j | mu_j)
    - Weighted (sumw2 provided): Gaussian(y_j | mu_j, sumw2_j)

    Parameters
    ----------
    counts : (m,) observed bin counts (or weighted sums)
    widths : (m,) bin widths
    K : (m, m) prior covariance (possibly augmented)
    mean_values : (m,) prior mean at bin centres
    sumw2 : (m,) or None
        Sum of squared weights per bin.  When provided, switches from
        Poisson to Gaussian observation model with variance sumw2_j.

    Returns
    -------
    f_hat : (m,) posterior mode of log-rate
    mu_hat : (m,) exp(f_hat) * widths
    """

    f = mean_values.copy()
    weighted = sumw2 is not None

    for _iteration in range(max_iter):
        mu = np.exp(np.clip(f, -20, 20)) * widths

        if weighted:
            # Gaussian observation model: p(y|f) = N(y | mu, sumw2)
            # grad of log p(y|f) w.r.t. f = (y - mu) * mu / sumw2
            # W = -d^2/df^2 log p = mu^2 / sumw2  (Fisher information)
            sigma2 = np.maximum(sumw2, 1e-10)
            w = mu**2 / sigma2
            obs_grad = (counts - mu) * mu / sigma2
        else:
            # Poisson observation model: p(n|f) = Poisson(n | mu)
            w = mu
            obs_grad = counts - mu

        grad = obs_grad - solve(k, f - mean_values, assume_a="pos")

        b = np.diag(w) + np.linalg.inv(k)
        try:
            delta = solve(b, grad, assume_a="pos")
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(b, grad, rcond=None)[0]

        f_new = f + delta
        if np.max(np.abs(delta)) < tol:
            f = f_new
            break
        f = f_new

    mu_final = np.exp(np.clip(f, -20, 20)) * widths
    return f, mu_final


def _laplace_log_marginal(
    counts: np.ndarray,
    widths: np.ndarray,
    kernel: GPKernel,
    log_sigma: float, log_ell: float,
    mean_values: np.ndarray,
    centres: np.ndarray,
    h: np.ndarray | None,
    beta_prior_variance: float,
    sumw2: np.ndarray | None = None,
) -> float:
    """Laplace approximation to log marginal likelihood."""
    m = len(counts)
    k = _kernel_matrix(centres, centres, kernel, log_sigma, log_ell)
    k_eff = _augment_kernel(k, h, beta_prior_variance) if h is not None else k.copy()
    k_eff += 1e-6 * np.eye(m)

    try:
        f_hat, mu_hat = _laplace_mode(counts, widths, k_eff, mean_values,
                                       sumw2=sumw2)
    except (np.linalg.LinAlgError, FloatingPointError):
        return -np.inf

    f_hat_clipped = np.clip(f_hat, -20, 20)

    if sumw2 is not None:
        # Gaussian log-likelihood: -0.5 * (y - mu)^2 / sigma^2
        sigma2 = np.maximum(sumw2, 1e-10)
        ll = -0.5 * np.sum((counts - mu_hat)**2 / sigma2)
        w = mu_hat**2 / sigma2
    else:
        # Poisson log-likelihood
        ll = np.sum(counts * f_hat_clipped
                    + counts * np.log(np.maximum(widths, 1e-300))
                    - np.exp(f_hat_clipped) * widths)
        w = mu_hat

    try:
        l_k = cho_factor(k_eff)
        alpha = cho_solve(l_k, f_hat - mean_values)
        log_prior = (-0.5 * (f_hat - mean_values) @ alpha
                     - 0.5 * np.sum(np.log(np.diag(l_k[0]))) * 2)
    except np.linalg.LinAlgError:
        return -np.inf

    w_sqrt = np.sqrt(np.maximum(w, 1e-10))
    b_mat = np.eye(m) + (w_sqrt[:, None] * k_eff * w_sqrt[None, :])
    try:
        l_b = cho_factor(b_mat)
        log_det_b = np.sum(np.log(np.diag(l_b[0]))) * 2
    except np.linalg.LinAlgError:
        return -np.inf

    return float(ll + log_prior - 0.5 * log_det_b)


def _select_mean_degree(
    counts: np.ndarray,
    widths: np.ndarray,
    centres: np.ndarray,
    edges: np.ndarray,
    kernel: GPKernel,
    beta_prior_variance: float = 100.0,
    min_degree: int = 1,
    max_degree: int = 3,
    sumw2: np.ndarray | None = None,
) -> int:
    """Select polynomial mean degree by Laplace marginal likelihood.

    Compares log marginal likelihood across degrees min_degree..max_degree,
    each with a grid of hyperparameter initialisations, and returns
    the degree achieving the highest value.  The default min_degree=1
    avoids constant means, which cause boundary artifacts for any
    non-uniform distribution.
    """
    x_min, x_max = float(edges[0]), float(edges[-1])
    data_range = x_max - x_min
    best_lml = -np.inf
    best_deg = min_degree

    for deg in range(min_degree, max_degree + 1):
        h = _build_design_matrix(centres, deg, x_min, x_max)
        if np.sum(counts) > 0:
            beta = _fit_mean_coefficients(counts, widths, h)
        else:
            beta = np.zeros(h.shape[1])
        mean_vals = h @ beta

        best_local = -np.inf
        for init_le in [np.log(data_range / 10),
                        np.log(data_range / 5),
                        np.log(data_range / 3)]:
            for init_ls in [-0.5, 0.0, 0.5]:
                lml = _laplace_log_marginal(
                    counts, widths, kernel, init_ls, init_le,
                    mean_vals, centres, h, beta_prior_variance,
                    sumw2=sumw2,
                )
                if lml > best_local:
                    best_local = lml

        if best_local > best_lml:
            best_lml = best_local
            best_deg = deg

    return best_deg


class GPTemplate:
    """Smooth template using Gaussian process density estimation.

    Fits a smooth rate function to histogram bin counts using a
    log-Gaussian Cox process model with Laplace inference.

    Parameters
    ----------
    histogram : Histogram
        Binned data to smooth.
    kernel : GPKernel
        Covariance function type.
    optimize_hyperparameters : bool
        If True, optimize lengthscale and amplitude by marginal
        likelihood.  If False, uses provided values.
    log_sigma : float
        Log amplitude of the GP prior.
    log_ell : float
        Log lengthscale of the GP prior.
    mean_degree : int, str
        Mean function in log-rate space.
        "bspline" (default): B-spline basis with automatic knot count,
        making the GP fully shape-agnostic via R&W 2006 Sec. 2.7.
        "data": uses smoothed observed log-counts as the mean.
        "auto": selects polynomial degree by marginal likelihood.
        An integer (0, 1, 2, ...) specifies a polynomial degree.
        For "bspline" and integer modes, uncertainty in the
        coefficients is propagated into the posterior covariance
        via kernel augmentation K_eff = K + H B H^T.
    beta_prior_variance : float
        Prior variance on each mean function coefficient.  Should be
        large enough that the data dominate (default 100).
    """

    def __init__(
        self,
        histogram: Histogram,
        kernel: GPKernel = GPKernel.MATERN_32,
        optimize_hyperparameters: bool = True,
        log_sigma: float = 0.0,
        log_ell: float = 0.0,
        mean_degree: int | str = "bspline",
        beta_prior_variance: float = 100.0,
    ) -> None:
        if not isinstance(histogram, Histogram):
            raise TypeError(
                f"GPTemplate requires a Histogram, got {type(histogram).__name__}"
            )
        self._histogram = histogram
        self._edges = histogram.edges.copy()
        self._nbins = histogram.nbins
        self._total = histogram.total
        self._kernel_type = kernel
        self._beta_prior_variance = beta_prior_variance

        centres = 0.5 * (self._edges[:-1] + self._edges[1:])
        widths = np.diff(self._edges)
        counts = histogram.values.copy()
        self._centres = centres
        self._widths = widths

        # Detect weighted events: sumw2 != values means non-trivial weights
        raw_sumw2 = histogram.sumw2
        if np.allclose(raw_sumw2, counts):
            self._obs_sumw2 = None  # unweighted: use Poisson model
        else:
            self._obs_sumw2 = raw_sumw2.copy()  # weighted: use Gaussian model

        # Resolve mean_degree
        self._use_data_mean = (mean_degree == "data")
        self._use_bspline = (mean_degree == "bspline")
        self._bspline_knots = None
        self._bspline_degree = 3

        if self._use_bspline:
            # B-spline basis: shape-agnostic via R&W 2006 Sec. 2.7.
            # Number of internal knots scales with number of bins:
            # enough flexibility for arbitrary shapes, but fewer than
            # the number of bins so the GP still does useful smoothing.
            n_internal = max(self._nbins // 3, 3)
            h, knots = _build_bspline_basis(
                centres, float(self._edges[0]), float(self._edges[-1]),
                n_internal=n_internal, degree=self._bspline_degree,
            )
            self._h = h
            self._bspline_knots = knots
            self._mean_degree = -2  # sentinel for bspline

            if self._total > 0:
                beta_hat = _fit_mean_coefficients(counts, widths, h)
            else:
                beta_hat = np.zeros(h.shape[1])
            self._beta_hat = beta_hat
            self._mean_values = h @ beta_hat

        elif self._use_data_mean:
            # Nonparametric: smooth the observed counts in rate space
            # first, then take log.  This avoids log(0) pathology for
            # bins near the edge of support.  The GP then models
            # residuals around this smoothed trend.
            safe_counts = np.maximum(counts, 0.5)
            rate = safe_counts / widths
            # Gaussian kernel smooth in bin-index space
            sigma_smooth = 2.0  # bins
            idx = np.arange(self._nbins, dtype=float)
            weights_matrix = np.exp(
                -0.5 * ((idx[:, None] - idx[None, :]) / sigma_smooth) ** 2
            )
            # Weight by counts so high-stats bins dominate
            w_smooth = weights_matrix * safe_counts[None, :]
            w_smooth /= w_smooth.sum(axis=1, keepdims=True)
            smooth_rate = w_smooth @ rate
            smooth_rate = np.maximum(smooth_rate, 0.5 / widths.max())
            self._mean_values = np.log(smooth_rate)
            self._mean_degree = -1  # sentinel
            self._h = None
            self._beta_hat = None
        else:
            if mean_degree == "auto" and self._total > 0:
                mean_degree = _select_mean_degree(
                    counts, widths, centres, self._edges,
                    kernel, beta_prior_variance,
                    sumw2=self._obs_sumw2,
                )
            elif mean_degree == "auto":
                mean_degree = 0
            self._mean_degree = int(mean_degree)

            # Build design matrix and fit mean function
            x_min, x_max = float(self._edges[0]), float(self._edges[-1])
            h = _build_design_matrix(centres, mean_degree, x_min, x_max)
            self._h = h
            self._x_min = x_min
            self._x_max = x_max

            if self._total > 0:
                beta_hat = _fit_mean_coefficients(counts, widths, h)
            else:
                beta_hat = np.zeros(h.shape[1])
            self._beta_hat = beta_hat
            self._mean_values = h @ beta_hat

        # Always store edge range for evaluate_density
        self._x_min = float(self._edges[0])
        self._x_max = float(self._edges[-1])

        # Optimize or use fixed hyperparameters
        if optimize_hyperparameters and self._total > 0:
            data_range = self._edges[-1] - self._edges[0]
            init_log_ell = np.log(max(data_range / 5.0, widths.mean()))

            if self._use_data_mean:
                # With data mean, GP handles small residuals:
                # smaller amplitude, wider prior to let marginal
                # likelihood find the right smoothing scale.
                prior_mean_le = init_log_ell
                prior_width_le = 2.0
                prior_mean_ls = -1.0  # smaller initial amplitude
                prior_width_ls = 2.0
            else:
                prior_mean_le = init_log_ell
                prior_width_le = 1.5
                prior_mean_ls = 0.0
                prior_width_ls = 2.0

            def neg_lml(params):
                ls, le = params
                val = _laplace_log_marginal(
                    counts, widths, kernel, ls, le,
                    self._mean_values, centres,
                    h=self._h,
                    beta_prior_variance=beta_prior_variance,
                    sumw2=self._obs_sumw2,
                )
                penalty = (0.5 * ((le - prior_mean_le) / prior_width_le) ** 2
                           + 0.5 * ((ls - prior_mean_ls) / prior_width_ls) ** 2)
                return (-val + penalty) if np.isfinite(val) else 1e10

            best_nll = np.inf
            best_params = [log_sigma, init_log_ell]
            for init_le in [np.log(data_range / 10),
                            np.log(data_range / 5),
                            np.log(data_range / 3)]:
                for init_ls in ([-2.0, -1.0, 0.0] if self._use_data_mean
                                else [-0.5, 0.0, 0.5]):
                    try:
                        result = minimize(
                            neg_lml,
                            x0=[init_ls, init_le],
                            method="Nelder-Mead",
                            options={"maxiter": 200, "xatol": 0.01, "fatol": 0.1},
                        )
                        if result.fun < best_nll:
                            best_nll = result.fun
                            best_params = [float(result.x[0]), float(result.x[1])]
                    except Exception:
                        pass

            self._log_sigma = best_params[0]
            self._log_ell = best_params[1]
        else:
            self._log_sigma = log_sigma
            self._log_ell = log_ell

        # Build kernel and compute posterior
        k = _kernel_matrix(centres, centres, kernel,
                           self._log_sigma, self._log_ell)
        if self._use_data_mean:
            k_eff = k + 1e-6 * np.eye(self._nbins)
        else:
            # Both polynomial and B-spline use R&W 2.7 augmentation
            k_eff = _augment_kernel(k, self._h, beta_prior_variance)
            k_eff += 1e-6 * np.eye(self._nbins)
        self._k = k_eff

        if self._total > 0:
            self._f_hat, self._mu_hat = _laplace_mode(
                counts, widths, k_eff, self._mean_values,
                sumw2=self._obs_sumw2,
            )
            # Full posterior covariance via Woodbury
            if self._obs_sumw2 is not None:
                sigma2 = np.maximum(self._obs_sumw2, 1e-10)
                w = self._mu_hat**2 / sigma2
            else:
                w = self._mu_hat
            w_sqrt = np.sqrt(np.maximum(w, 1e-10))
            b = np.eye(self._nbins) + (
                w_sqrt[:, None] * k_eff * w_sqrt[None, :]
            )
            try:
                l_b = cho_factor(b)
                b_inv = cho_solve(l_b, np.eye(self._nbins))
                self._post_cov = k_eff - k_eff @ (
                    w_sqrt[:, None] * b_inv * w_sqrt[None, :]
                ) @ k_eff
                self._post_var = np.diag(self._post_cov)
            except np.linalg.LinAlgError:
                self._post_cov = k_eff.copy()
                self._post_var = np.diag(k_eff)
        else:
            self._f_hat = self._mean_values.copy()
            self._mu_hat = np.zeros(self._nbins)
            self._post_cov = k_eff.copy()
            self._post_var = np.diag(k_eff)

        # Compute fitted counts and renormalise
        fitted_rate = np.exp(np.clip(self._f_hat, -20, 20))
        self._fitted_counts_raw = fitted_rate * widths

        raw_total = np.sum(self._fitted_counts_raw)
        if raw_total > 0 and self._total > 0:
            self._scale = self._total / raw_total
        else:
            self._scale = 1.0
        self._fitted_counts = self._fitted_counts_raw * self._scale

        self._sumw2 = histogram.sumw2.copy()

    @property
    def histogram(self) -> Histogram:
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
    def kernel_type(self) -> GPKernel:
        return self._kernel_type

    @property
    def lengthscale(self) -> float:
        return float(np.exp(self._log_ell))

    @property
    def amplitude(self) -> float:
        return float(np.exp(self._log_sigma))

    @property
    def mean_degree(self) -> int | str:
        if self._use_data_mean:
            return "data"
        if self._use_bspline:
            return "bspline"
        return self._mean_degree

    @property
    def mean_coefficients(self) -> np.ndarray | None:
        """Estimated mean function coefficients (None for data mean)."""
        if self._beta_hat is None:
            return None
        return self._beta_hat.copy()

    @property
    def posterior_variance(self) -> np.ndarray:
        """Posterior variance of log-rate at bin centres."""
        return self._post_var.copy()

    @property
    def posterior_covariance(self) -> np.ndarray:
        """Full posterior covariance of log-rate at bin centres.

        For B-spline or polynomial mean: includes mean function
        uncertainty from kernel augmentation K_eff = K + H B H^T
        (Rasmussen & Williams 2006, Sec. 2.7).
        For data mean: just the kernel posterior covariance.
        """
        return self._post_cov.copy()

    @property
    def log_rate(self) -> np.ndarray:
        """Posterior mode of the log-rate function at bin centres."""
        return self._f_hat.copy()

    def counts(self) -> np.ndarray:
        """Smoothed expected counts per bin."""
        return self._fitted_counts.copy()

    def density(self) -> np.ndarray:
        """Smooth probability density at bin centres."""
        if self._total == 0:
            return np.zeros(self._nbins)
        return self._fitted_counts / (self._total * self._widths)

    def evaluate_density(self, x: np.ndarray) -> np.ndarray:
        """Evaluate smooth density at arbitrary points via GP prediction."""
        from scipy.interpolate import BSpline

        x = np.asarray(x, dtype=np.float64)
        k_star = _kernel_matrix(x, self._centres, self._kernel_type,
                                self._log_sigma, self._log_ell)

        if self._use_data_mean:
            # Interpolate data mean to new points
            mean_star = np.interp(x, self._centres, self._mean_values)
            k_star_eff = k_star
        elif self._use_bspline:
            h_star = BSpline.design_matrix(
                x, self._bspline_knots, self._bspline_degree,
            ).toarray()
            mean_star = h_star @ self._beta_hat
            k_star_eff = k_star + self._beta_prior_variance * (
                h_star @ self._h.T
            )
        else:
            h_star = _build_design_matrix(x, self._mean_degree,
                                          self._x_min, self._x_max)
            mean_star = h_star @ self._beta_hat
            k_star_eff = k_star + self._beta_prior_variance * (
                h_star @ self._h.T
            )

        alpha = solve(self._k, self._f_hat - self._mean_values,
                      assume_a="pos")
        f_pred = mean_star + k_star_eff @ alpha
        rate = np.exp(np.clip(f_pred, -20, 20))
        if self._total > 0:
            return rate * self._scale / self._total
        return np.zeros_like(x)

    def __repr__(self) -> str:
        if self._use_data_mean:
            md = "data"
        elif self._use_bspline:
            md = f"bspline(k={len(self._bspline_knots) - self._bspline_degree - 1})"
        else:
            md = str(self._mean_degree)
        return (
            f"GPTemplate(nbins={self._nbins}, "
            f"kernel={self._kernel_type.value}, "
            f"ell={self.lengthscale:.3g}, "
            f"sigma={self.amplitude:.3g}, "
            f"mean={md}, "
            f"total={self._total:.6g})"
        )
