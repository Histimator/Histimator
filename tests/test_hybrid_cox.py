"""Tests for the hybrid Cox point-process Laplace inference.

Design principles
-----------------
All tests use small event counts (10 to 50) so the test suite runs
quickly on CI.  The underlying mathematics is validated against
analytical results where possible, and against the existing
``_laplace_mode`` function in ``gp_template.py`` for the Poisson path.

The test domain is [0, 10] throughout, with M = 20 Gauss-Legendre
quadrature points.  A squared-exponential kernel with fixed
hyperparameters (log_sigma = 0, log_ell = log(3)) is used everywhere
so that convergence behaviour is predictable.

Test structure
--------------
Tier 1: Protocol and model objects
    Verifies that PoissonCountModel and HybridCoxModel satisfy the
    ObservationModel protocol and that their gradient and Hessian
    entries have the correct signs and shapes.

Tier 2: Newton solver convergence
    Verifies that _cox_laplace_mode converges in finite iterations,
    returns finite values, and that the gradient at the mode is
    sufficiently small.

Tier 3: Log marginal likelihood
    Verifies that _cox_laplace_log_marginal returns a finite value
    and is higher at reasonable hyperparameters than at extreme ones.

Tier 4: PoissonCountModel regression
    Verifies that PoissonCountModel fed through _cox_laplace_mode
    recovers the same posterior mode as the existing _laplace_mode
    in gp_template.py.  This is the contract that guarantees the
    abstraction does not change any existing behaviour.

Tier 5: Density quality
    Verifies that the hybrid Cox path recovers a density closer to the
    truth than a coarsely binned Poisson path on the same events,
    measured by integrated squared error against the known generating
    density.  Uses 50 events for statistical power, still fast enough
    for CI.
"""

import numpy as np
import pytest

from histimator.cox_model import (
    HybridCoxModel,
    ObservationModel,
    PoissonCountModel,
    _cox_laplace_log_marginal,
    _cox_laplace_mode,
    gauss_legendre_quadrature,
)

# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

X_MIN = 0.0
X_MAX = 10.0
DOMAIN = X_MAX - X_MIN
M_QUAD = 20          # quadrature points
LOG_SIGMA = 0.0      # GP amplitude: exp(0) = 1
LOG_ELL = np.log(3)  # GP lengthscale: 3 units


def _sq_exp_kernel(
    x1: np.ndarray, x2: np.ndarray, log_sigma: float, log_ell: float
) -> np.ndarray:
    """Squared-exponential covariance matrix."""
    sigma2 = np.exp(2.0 * log_sigma)
    ell2 = np.exp(2.0 * log_ell)
    diff = x1[:, None] - x2[None, :]
    return sigma2 * np.exp(-0.5 * diff**2 / ell2)


def _make_augmented_kernel(
    event_positions: np.ndarray,
    quad_nodes: np.ndarray,
    log_sigma: float = LOG_SIGMA,
    log_ell: float = LOG_ELL,
) -> np.ndarray:
    """Kernel matrix on the augmented location set [events, quad_nodes]."""
    x_aug = np.concatenate([event_positions, quad_nodes])
    return _sq_exp_kernel(x_aug, x_aug, log_sigma, log_ell)


def _generate_uniform_events(n: int, seed: int) -> np.ndarray:
    """Uniform events on [X_MIN, X_MAX] with no edge clipping needed."""
    rng = np.random.default_rng(seed)
    return rng.uniform(X_MIN, X_MAX, n)


def _make_binned_histogram(
    events: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin events and return (counts, centres, widths)."""
    edges = np.linspace(X_MIN, X_MAX, n_bins + 1)
    counts, _ = np.histogram(events, bins=edges)
    widths = np.diff(edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return counts.astype(float), centres, widths


# ---------------------------------------------------------------------------
# Tier 1: Protocol and model objects
# ---------------------------------------------------------------------------

class TestObservationModelProtocol:
    """PoissonCountModel and HybridCoxModel satisfy the protocol."""

    def test_poisson_satisfies_protocol(self):
        counts = np.array([5.0, 10.0, 3.0])
        widths = np.ones(3)
        model = PoissonCountModel(counts, widths)
        assert isinstance(model, ObservationModel)

    def test_hybrid_satisfies_protocol(self):
        _, qw = gauss_legendre_quadrature(X_MIN, X_MAX, M_QUAD)
        model = HybridCoxModel(n_events=10, quad_weights=qw)
        assert isinstance(model, ObservationModel)

    def test_poisson_gradient_shape(self):
        counts = np.array([5.0, 10.0, 3.0])
        widths = np.ones(3)
        model = PoissonCountModel(counts, widths)
        f = np.zeros(3)
        assert model.gradient(f).shape == (3,)

    def test_poisson_hessian_non_negative(self):
        counts = np.array([5.0, 10.0, 3.0])
        widths = np.ones(3)
        model = PoissonCountModel(counts, widths)
        f = np.zeros(3)
        assert np.all(model.hessian_diagonal(f) >= 0)

    def test_hybrid_gradient_shape(self):
        n_ev, n_q = 10, M_QUAD
        _, qw = gauss_legendre_quadrature(X_MIN, X_MAX, n_q)
        model = HybridCoxModel(n_events=n_ev, quad_weights=qw)
        f = np.zeros(n_ev + n_q)
        assert model.gradient(f).shape == (n_ev + n_q,)

    def test_hybrid_gradient_event_block_is_one(self):
        """Event block gradient is always +1, independent of f."""
        n_ev = 15
        _, qw = gauss_legendre_quadrature(X_MIN, X_MAX, M_QUAD)
        model = HybridCoxModel(n_events=n_ev, quad_weights=qw)
        f = np.random.default_rng(0).normal(size=n_ev + M_QUAD)
        grad = model.gradient(f)
        np.testing.assert_array_equal(grad[:n_ev], 1.0)

    def test_hybrid_gradient_quad_block_is_negative(self):
        """Quadrature block gradient is negative (rate pushes down)."""
        n_ev = 10
        _, qw = gauss_legendre_quadrature(X_MIN, X_MAX, M_QUAD)
        model = HybridCoxModel(n_events=n_ev, quad_weights=qw)
        f = np.zeros(n_ev + M_QUAD)
        grad = model.gradient(f)
        assert np.all(grad[n_ev:] < 0)

    def test_hybrid_hessian_event_block_is_zero(self):
        """Event block Hessian diagonal is exactly zero."""
        n_ev = 10
        _, qw = gauss_legendre_quadrature(X_MIN, X_MAX, M_QUAD)
        model = HybridCoxModel(n_events=n_ev, quad_weights=qw)
        f = np.zeros(n_ev + M_QUAD)
        w = model.hessian_diagonal(f)
        np.testing.assert_array_equal(w[:n_ev], 0.0)

    def test_hybrid_hessian_quad_block_is_positive(self):
        """Quadrature block Hessian diagonal is strictly positive."""
        n_ev = 10
        _, qw = gauss_legendre_quadrature(X_MIN, X_MAX, M_QUAD)
        model = HybridCoxModel(n_events=n_ev, quad_weights=qw)
        f = np.zeros(n_ev + M_QUAD)
        w = model.hessian_diagonal(f)
        assert np.all(w[n_ev:] > 0)

    def test_hybrid_log_likelihood_flat_rate(self):
        """With flat f=0, log-lik = N - integral(exp(0)*dx).

        For f=0 everywhere, sum_i f(x_i) = 0 and
        sum_j w_j exp(0) = sum_j w_j = (X_MAX - X_MIN) = 10.
        So log-lik = 0 - 10 = -10.
        """
        n_ev = 8
        qn, qw = gauss_legendre_quadrature(X_MIN, X_MAX, M_QUAD)
        model = HybridCoxModel(n_events=n_ev, quad_weights=qw)
        f = np.zeros(n_ev + M_QUAD)
        ll = model.log_likelihood(f)
        expected = 0.0 - float(np.sum(qw))  # sum qw = X_MAX - X_MIN = 10
        assert abs(ll - expected) < 1e-10

    def test_poisson_log_likelihood_known_value(self):
        """With f=0 and width=1: log-lik = sum(n*0 - exp(0)*1) = -m."""
        m = 5
        counts = np.ones(m)
        widths = np.ones(m)
        model = PoissonCountModel(counts, widths)
        f = np.zeros(m)
        # log-lik = sum_j [n_j * f_j - exp(f_j)*w_j] = sum_j [0 - 1] = -m
        assert abs(model.log_likelihood(f) - (-m)) < 1e-10


# ---------------------------------------------------------------------------
# Tier 2: Newton solver convergence
# ---------------------------------------------------------------------------

class TestCoxLaplaceMode:
    """_cox_laplace_mode converges correctly."""

    def test_uniform_events_converges(self):
        """20 uniform events: mode is finite and has correct shape."""
        events = _generate_uniform_events(20, seed=0)
        qn, qw = gauss_legendre_quadrature(X_MIN, X_MAX, M_QUAD)
        k = _make_augmented_kernel(events, qn)
        mean = np.zeros(len(events) + M_QUAD)
        obs = HybridCoxModel(n_events=len(events), quad_weights=qw)

        f_hat, w_hat = _cox_laplace_mode(obs, k, mean)

        assert f_hat.shape == mean.shape
        assert np.all(np.isfinite(f_hat))
        assert w_hat.shape == mean.shape

    def test_gradient_near_zero_at_mode(self):
        """The total gradient at f_hat should be small (convergence check).

        The total gradient is obs_grad - K^{-1}(f - m).  At the mode
        this should be near zero by construction of the Newton solver.
        """
        events = _generate_uniform_events(20, seed=1)
        qn, qw = gauss_legendre_quadrature(X_MIN, X_MAX, M_QUAD)
        k = _make_augmented_kernel(events, qn)
        jitter = 1e-6
        k_reg = k + jitter * np.eye(len(events) + M_QUAD)
        mean = np.zeros(len(events) + M_QUAD)
        obs = HybridCoxModel(n_events=len(events), quad_weights=qw)

        f_hat, _ = _cox_laplace_mode(obs, k_reg, mean, jitter=0.0)

        from scipy.linalg import solve as sp_solve
        k_inv_fm = sp_solve(k_reg, f_hat - mean, assume_a="pos")
        total_grad = obs.gradient(f_hat) - k_inv_fm
        assert np.max(np.abs(total_grad)) < 1e-4

    def test_rate_integral_approximately_n(self):
        """The quadrature integral of exp(f) at the mode should be ~N.

        The Cox process mode satisfies: for the optimal f, the expected
        total number of events (the integral of the rate) is close to
        the observed N.  This is the GP analogue of the Poisson
        maximum likelihood condition.  It holds approximately, not
        exactly, because the GP prior regularises the solution.
        """
        n_events = 20
        events = _generate_uniform_events(n_events, seed=2)
        qn, qw = gauss_legendre_quadrature(X_MIN, X_MAX, M_QUAD)
        k = _make_augmented_kernel(events, qn)
        mean = np.zeros(n_events + M_QUAD)
        obs = HybridCoxModel(n_events=n_events, quad_weights=qw)

        f_hat, _ = _cox_laplace_mode(obs, k, mean)

        # Integral of rate over domain via quadrature at the mode
        rate_at_quad = np.exp(np.clip(f_hat[n_events:], -20, 20))
        total_rate = float(np.sum(qw * rate_at_quad))

        # Should be within a factor of 2 of N for any reasonable kernel.
        # Exact equality does not hold because the GP prior pulls toward
        # the mean function.
        assert total_rate > 0.5 * n_events
        assert total_rate < 2.0 * n_events

    def test_gaussian_bump_events_recovers_peak_direction(self):
        """Events concentrated near x=7 should push f_hat higher near x=7.

        This validates that the event-position sum term is driving
        the posterior in the right direction.
        """
        rng = np.random.default_rng(3)
        events = rng.normal(loc=7.0, scale=0.8, size=15)
        events = events[(events >= X_MIN) & (events <= X_MAX)]

        qn, qw = gauss_legendre_quadrature(X_MIN, X_MAX, M_QUAD)
        k = _make_augmented_kernel(events, qn)
        mean = np.zeros(len(events) + M_QUAD)
        obs = HybridCoxModel(n_events=len(events), quad_weights=qw)

        f_hat, _ = _cox_laplace_mode(obs, k, mean)

        # f_hat at quadrature points near x=7 should exceed f_hat near x=2
        near_peak = np.abs(qn - 7.0) < 1.5
        near_left = np.abs(qn - 2.0) < 1.5
        mean_near_peak = float(np.mean(f_hat[len(events):][near_peak]))
        mean_near_left = float(np.mean(f_hat[len(events):][near_left]))
        assert mean_near_peak > mean_near_left

    def test_different_seeds_give_different_modes(self):
        """Two different datasets produce different posterior modes."""
        events_a = _generate_uniform_events(20, seed=10)
        events_b = _generate_uniform_events(20, seed=11)
        qn, qw = gauss_legendre_quadrature(X_MIN, X_MAX, M_QUAD)

        def run(events):
            k = _make_augmented_kernel(events, qn)
            mean = np.zeros(len(events) + M_QUAD)
            obs = HybridCoxModel(n_events=len(events), quad_weights=qw)
            f_hat, _ = _cox_laplace_mode(obs, k, mean)
            return f_hat[len(events):]  # mode at quadrature points

        f_a = run(events_a)
        f_b = run(events_b)
        assert not np.allclose(f_a, f_b, atol=1e-3)

    def test_poisson_mode_shape_and_finiteness(self):
        """PoissonCountModel through _cox_laplace_mode returns finite mode."""
        counts, centres, widths = _make_binned_histogram(
            _generate_uniform_events(50, seed=4), n_bins=10
        )
        k = _sq_exp_kernel(centres, centres, LOG_SIGMA, LOG_ELL)
        k += 1e-6 * np.eye(len(centres))
        mean = np.zeros(len(centres))
        obs = PoissonCountModel(counts, widths)

        f_hat, w_hat = _cox_laplace_mode(obs, k, mean)

        assert f_hat.shape == (10,)
        assert np.all(np.isfinite(f_hat))


# ---------------------------------------------------------------------------
# Tier 3: Log marginal likelihood
# ---------------------------------------------------------------------------

class TestCoxLaplaceLogMarginal:
    """_cox_laplace_log_marginal behaves correctly."""

    def test_finite_for_uniform_events(self):
        events = _generate_uniform_events(15, seed=5)
        qn, qw = gauss_legendre_quadrature(X_MIN, X_MAX, M_QUAD)
        k = _make_augmented_kernel(events, qn)
        mean = np.zeros(len(events) + M_QUAD)
        obs = HybridCoxModel(n_events=len(events), quad_weights=qw)

        lml = _cox_laplace_log_marginal(obs, k, mean)

        assert np.isfinite(lml)

    def test_moderate_lengthscale_beats_very_long_on_structured_data(self):
        """Marginal likelihood at ell=2 exceeds that at ell=40 for bump data.

        When events are concentrated near x=5 on a [0, 10] domain, a
        very long lengthscale (ell=40, nearly constant GP) cannot fit
        the bump and should lose to a moderate lengthscale (ell=2) that
        can resolve it.

        The comparison against very short lengthscales (ell << 1) is not
        made because the Laplace approximation is unreliable there for
        point process data: a short-lengthscale kernel can place spikes
        exactly at event locations, making the event sum term large while
        the quadrature misses the narrow peaks, inflating the apparent
        marginal likelihood.  The GPTemplate code addresses this with a
        Gaussian prior on the lengthscale; that is out of scope here.
        """
        rng = np.random.default_rng(42)
        # Concentrated bump: most events near x=5
        events = rng.normal(loc=5.0, scale=1.0, size=30)
        events = events[(events >= X_MIN + 0.5) & (events <= X_MAX - 0.5)]

        qn, qw = gauss_legendre_quadrature(X_MIN, X_MAX, M_QUAD)
        obs = HybridCoxModel(n_events=len(events), quad_weights=qw)
        mean = np.zeros(len(events) + M_QUAD)

        def lml_at_ell(log_ell):
            k = _make_augmented_kernel(events, qn, log_ell=log_ell)
            return _cox_laplace_log_marginal(obs, k, mean)

        lml_moderate = lml_at_ell(np.log(2.0))
        lml_very_long = lml_at_ell(np.log(40.0))

        assert lml_moderate > lml_very_long

    def test_more_events_raises_marginal_likelihood(self):
        """Adding more events from the same process increases the log marginal lik.

        With a fixed kernel and more data, the model is better supported.
        """
        qn, qw = gauss_legendre_quadrature(X_MIN, X_MAX, M_QUAD)

        def lml_for_n(n, seed):
            events = _generate_uniform_events(n, seed=seed)
            obs = HybridCoxModel(n_events=n, quad_weights=qw)
            k = _make_augmented_kernel(events, qn)
            mean = np.zeros(n + M_QUAD)
            return _cox_laplace_log_marginal(obs, k, mean)

        # Average over several seeds to reduce variance
        lmls_small = [lml_for_n(5, s) for s in range(5)]
        lmls_large = [lml_for_n(30, s) for s in range(5)]
        assert np.mean(lmls_large) > np.mean(lmls_small)


# ---------------------------------------------------------------------------
# Tier 4: PoissonCountModel regression against existing _laplace_mode
# ---------------------------------------------------------------------------

class TestPoissonCountModelRegression:
    """PoissonCountModel matches the existing _laplace_mode in gp_template."""

    def test_mode_matches_existing_implementation(self):
        """The protocol path reproduces the existing Laplace mode to tolerance.

        This is the contract that guarantees the abstraction does not
        silently change any existing result.
        """
        try:
            from histimator.gp_template import _laplace_mode as legacy_laplace_mode
        except ImportError:
            pytest.skip(
                "gp_template not importable from test environment; "
                "run this test from the repository root with "
                "PYTHONPATH=src."
            )

        counts, centres, widths = _make_binned_histogram(
            _generate_uniform_events(80, seed=7), n_bins=10
        )
        k = _sq_exp_kernel(centres, centres, LOG_SIGMA, LOG_ELL)
        k += 1e-6 * np.eye(10)
        mean = np.zeros(10)

        f_legacy, _ = legacy_laplace_mode(counts, widths, k, mean)

        obs = PoissonCountModel(counts, widths)
        f_new, _ = _cox_laplace_mode(obs, k, mean, jitter=0.0)

        np.testing.assert_allclose(f_new, f_legacy, rtol=1e-5, atol=1e-6)

    def test_log_likelihood_matches_poisson_formula(self):
        """PoissonCountModel.log_likelihood matches the manual Poisson formula."""
        counts = np.array([3.0, 7.0, 2.0, 5.0])
        widths = np.array([1.0, 1.5, 0.5, 2.0])
        model = PoissonCountModel(counts, widths)
        f = np.log(counts / widths)  # MLE for a single bin
        ll = model.log_likelihood(f)

        f_clipped = np.clip(f, -20, 20)
        mu = np.exp(f_clipped) * widths
        expected = float(np.sum(counts * f_clipped - mu))
        assert abs(ll - expected) < 1e-10


# ---------------------------------------------------------------------------
# Tier 5: Density quality — hybrid Cox vs coarse binning
# ---------------------------------------------------------------------------

class TestDensityQuality:
    """Hybrid Cox density is closer to truth than coarse-binned Poisson GP.

    This is the acceptance test from the sprint description, scaled down
    to a sample size that is manageable on CI.  The true density is
    uniform on [0, 10].  We compare:

    - Hybrid Cox: GP fitted on 50 exact event positions + 20 quadrature pts
    - Binned Poisson: GP fitted on a 5-bin histogram of the same events

    The hybrid path should have lower integrated squared error because
    it retains exact event positions rather than smearing them across
    bin widths of width 2.
    """

    def _ise_hybrid(self, events: np.ndarray) -> float:
        """ISE of the hybrid Cox density against Uniform(0, 10)."""
        qn, qw = gauss_legendre_quadrature(X_MIN, X_MAX, M_QUAD)
        n_ev = len(events)
        k = _make_augmented_kernel(events, qn)
        mean = np.zeros(n_ev + M_QUAD)
        obs = HybridCoxModel(n_events=n_ev, quad_weights=qw)

        f_hat, _ = _cox_laplace_mode(obs, k, mean)

        # Rate at quadrature points (the mode values there)
        rate_at_quad = np.exp(np.clip(f_hat[n_ev:], -20, 20))
        total_rate = float(np.sum(qw * rate_at_quad))

        # Density at quadrature points: rate / total_rate
        density_at_quad = rate_at_quad / max(total_rate, 1e-10)
        true_density = 1.0 / DOMAIN  # Uniform on [0, 10]

        # ISE via quadrature
        ise = float(np.sum(qw * (density_at_quad - true_density) ** 2))
        return ise

    def _ise_binned_poisson(self, events: np.ndarray, n_bins: int) -> float:
        """ISE of a coarsely binned Poisson GP against Uniform(0, 10).

        Runs _cox_laplace_mode with PoissonCountModel on a coarse
        histogram of the same events.
        """
        counts, centres, widths = _make_binned_histogram(events, n_bins)
        k = _sq_exp_kernel(centres, centres, LOG_SIGMA, LOG_ELL)
        k += 1e-6 * np.eye(n_bins)
        mean = np.zeros(n_bins)
        obs = PoissonCountModel(counts, widths)

        f_hat, _ = _cox_laplace_mode(obs, k, mean, jitter=0.0)

        # Evaluate density at bin centres: exp(f) / total
        rate = np.exp(np.clip(f_hat, -20, 20))
        total_rate = float(np.sum(rate * widths))
        density = rate / max(total_rate, 1e-10)

        true_density = 1.0 / DOMAIN
        # ISE at bin centres, weighted by bin width (midpoint rule)
        ise = float(np.sum(widths * (density - true_density) ** 2))
        return ise

    def test_hybrid_beats_coarse_binning_on_average(self):
        """Averaged over seeds, hybrid ISE < coarsely binned Poisson ISE.

        5-bin histogram has bin widths of 2 units against a lengthscale
        of 3, so binning introduces substantial smearing.  The hybrid
        path avoids this.
        """
        n_events = 50
        n_seeds = 6  # enough for statistical stability, fast enough for CI

        ise_hybrid_vals = []
        ise_binned_vals = []

        for seed in range(n_seeds):
            events = _generate_uniform_events(n_events, seed=seed + 20)
            ise_hybrid_vals.append(self._ise_hybrid(events))
            ise_binned_vals.append(self._ise_binned_poisson(events, n_bins=5))

        mean_hybrid = float(np.mean(ise_hybrid_vals))
        mean_binned = float(np.mean(ise_binned_vals))

        assert mean_hybrid < mean_binned, (
            f"Hybrid ISE {mean_hybrid:.4f} should be less than "
            f"coarse-binned ISE {mean_binned:.4f}"
        )

    def test_hybrid_density_integrates_to_one(self):
        """The normalised hybrid density integrates to 1 over the domain."""
        events = _generate_uniform_events(30, seed=99)
        qn, qw = gauss_legendre_quadrature(X_MIN, X_MAX, M_QUAD)
        n_ev = len(events)
        k = _make_augmented_kernel(events, qn)
        mean = np.zeros(n_ev + M_QUAD)
        obs = HybridCoxModel(n_events=n_ev, quad_weights=qw)

        f_hat, _ = _cox_laplace_mode(obs, k, mean)

        rate_at_quad = np.exp(np.clip(f_hat[n_ev:], -20, 20))
        total_rate = float(np.sum(qw * rate_at_quad))
        integral_of_density = float(np.sum(qw * (rate_at_quad / total_rate)))

        assert abs(integral_of_density - 1.0) < 0.01
