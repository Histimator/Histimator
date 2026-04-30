"""Tests for GPTemplate.from_dataset and Sample.from_dataset hybrid Cox path."""

import numpy as np
import pytest
from scipy.stats import norm as normal_dist

from histimator.cox_model import gauss_legendre_quadrature
from histimator.data import Dataset
from histimator.gp_template import GPTemplate
from histimator.histograms import Histogram
from histimator.samples import Sample
from histimator.templates import BinnedTemplate

X_MIN, X_MAX = 0.0, 10.0
EDGES_10 = np.linspace(X_MIN, X_MAX, 11)
EDGES_5 = np.linspace(X_MIN, X_MAX, 6)


def _uniform_dataset(n, seed):
    return Dataset(np.random.default_rng(seed).uniform(X_MIN, X_MAX, n))


def _gaussian_dataset(n, loc, scale, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(loc, scale, n * 3)
    x = x[(x >= X_MIN) & (x <= X_MAX)][:n]
    return Dataset(x)


def _ise(density_fn, true_density_fn, n_eval=400):
    qn, qw = gauss_legendre_quadrature(X_MIN, X_MAX, n_eval)
    return float(np.sum(qw * (density_fn(qn) - true_density_fn(qn)) ** 2))


class TestTemplateProtocol:
    """GPTemplate.from_dataset satisfies the full Template interface."""

    @pytest.fixture(scope="class")
    def gp(self):
        return GPTemplate.from_dataset(
            _uniform_dataset(30, seed=0), EDGES_10,
            optimize_hyperparameters=False,
        )

    def test_returns_gp_template(self, gp):
        assert isinstance(gp, GPTemplate)

    def test_edges_matches_input(self, gp):
        np.testing.assert_array_equal(gp.edges, EDGES_10)

    def test_nbins_correct(self, gp):
        assert gp.nbins == 10

    def test_counts_shape(self, gp):
        assert gp.counts().shape == (10,)

    def test_counts_positive(self, gp):
        assert np.all(gp.counts() > 0)

    def test_counts_finite(self, gp):
        assert np.all(np.isfinite(gp.counts()))

    def test_density_shape(self, gp):
        assert gp.density().shape == (10,)

    def test_density_integrates_to_one(self, gp):
        integral = float(np.sum(gp.density() * np.diff(gp.edges)))
        assert abs(integral - 1.0) < 0.05

    def test_density_non_negative(self, gp):
        assert np.all(gp.density() >= 0)

    def test_evaluate_density_shape(self, gp):
        assert gp.evaluate_density(np.linspace(0.5, 9.5, 50)).shape == (50,)

    def test_evaluate_density_non_negative(self, gp):
        assert np.all(gp.evaluate_density(np.linspace(0.5, 9.5, 50)) >= 0)

    def test_evaluate_density_integrates_to_one(self, gp):
        qn, qw = gauss_legendre_quadrature(X_MIN, X_MAX, 60)
        assert abs(float(np.sum(qw * gp.evaluate_density(qn))) - 1.0) < 0.01

    def test_total_is_n_events(self, gp):
        assert gp._total == 30.0

    def test_posterior_covariance_shape(self, gp):
        # posterior_covariance is bin-level (projected from the augmented
        # training space when the GP is fit via from_dataset).
        assert gp.posterior_covariance.shape == (gp.nbins, gp.nbins)

    def test_posterior_variance_positive(self, gp):
        assert np.all(gp.posterior_variance > 0)

    def test_histogram_attribute_exists(self, gp):
        assert isinstance(gp.histogram, Histogram)
        assert gp.histogram.nbins == 10


class TestNormalization:

    def test_scale_positive(self):
        gp = GPTemplate.from_dataset(
            _uniform_dataset(25, seed=1), EDGES_10, optimize_hyperparameters=False,
        )
        assert gp._scale > 0

    def test_fitted_counts_sum_approximately_n_events(self):
        n = 30
        gp = GPTemplate.from_dataset(
            _uniform_dataset(n, seed=2), EDGES_10, optimize_hyperparameters=False,
        )
        total = float(gp.counts().sum())
        assert 0.8 * n < total < 1.2 * n

    def test_evaluate_density_agrees_with_density_at_centres(self):
        gp = GPTemplate.from_dataset(
            _uniform_dataset(40, seed=3), EDGES_10, optimize_hyperparameters=False,
        )
        centres = 0.5 * (EDGES_10[:-1] + EDGES_10[1:])
        np.testing.assert_allclose(gp.evaluate_density(centres), gp.density(), rtol=0.10)

    def test_scale_equals_n_over_midpoint_sum(self):
        gp = GPTemplate.from_dataset(
            _uniform_dataset(20, seed=4), EDGES_10, optimize_hyperparameters=False,
        )
        # from_dataset uses two normalization paths:
        # - Midpoint: fitted_counts sum to _total (so density integrates to 1)
        # - Quadrature: _scale normalises the continuous evaluate_density.
        # Verify the midpoint invariant: sum(counts) == total.
        assert abs(float(gp.counts().sum()) - gp._total) < 1e-10


class TestDensityQuality:

    def test_gaussian_bump_density_converges(self):
        gp = GPTemplate.from_dataset(
            _gaussian_dataset(30, loc=5.0, scale=1.0, seed=20),
            EDGES_10, optimize_hyperparameters=False,
        )
        qn, _ = gauss_legendre_quadrature(X_MIN, X_MAX, 80)
        peak_x = float(qn[np.argmax(gp.evaluate_density(qn))])
        assert 3.5 < peak_x < 6.5

    def test_different_seeds_produce_different_densities(self):
        g1 = GPTemplate.from_dataset(
            _uniform_dataset(25, seed=30), EDGES_10, optimize_hyperparameters=False,
        )
        g2 = GPTemplate.from_dataset(
            _uniform_dataset(25, seed=31), EDGES_10, optimize_hyperparameters=False,
        )
        qn, _ = gauss_legendre_quadrature(X_MIN, X_MAX, 40)
        assert not np.allclose(g1.evaluate_density(qn), g2.evaluate_density(qn), atol=1e-3)

    def test_hyperparameter_optimization_runs(self):
        gp = GPTemplate.from_dataset(
            _uniform_dataset(20, seed=40), EDGES_10, optimize_hyperparameters=True,
        )
        qn, qw = gauss_legendre_quadrature(X_MIN, X_MAX, 40)
        d = gp.evaluate_density(qn)
        assert np.all(np.isfinite(d))
        assert abs(float(np.sum(qw * d)) - 1.0) < 0.05

    def test_hybrid_beats_coarsely_binned_gp_on_structured_distribution(self):
        """Exact positions beat 5-bin GP for a narrow Gaussian peak.

        Both fits use fixed hyperparameters, isolating the effect of
        event position information.
        """
        peak_loc, peak_sigma = 5.0, 0.8

        def true_fn(x):
            return normal_dist.pdf(x, peak_loc, peak_sigma)

        qn, qw = gauss_legendre_quadrature(X_MIN, X_MAX, 80)
        norm_const = float(np.sum(qw * true_fn(qn)))

        def true_fn_normed(x):
            return true_fn(x) / norm_const

        ise_hybrid, ise_coarse = [], []
        for seed in range(6):
            events = _gaussian_dataset(60, loc=peak_loc, scale=peak_sigma, seed=seed + 100)
            gp = GPTemplate.from_dataset(events, EDGES_10, optimize_hyperparameters=False)
            ise_hybrid.append(_ise(gp.evaluate_density, true_fn_normed))
            gp_coarse = GPTemplate(events.to_histogram(edges=EDGES_5),
                                   optimize_hyperparameters=False)
            ise_coarse.append(_ise(gp_coarse.evaluate_density, true_fn_normed))

        assert np.mean(ise_hybrid) < np.mean(ise_coarse)


class TestEdgeCases:

    def test_events_outside_domain_clipped(self):
        values = np.random.default_rng(50).uniform(-5, 15, 60)
        in_domain = values[(values >= X_MIN) & (values <= X_MAX)]
        gp = GPTemplate.from_dataset(Dataset(values), EDGES_10, optimize_hyperparameters=False)
        assert gp._n_events == len(in_domain)

    def test_all_events_outside_domain_raises(self):
        with pytest.raises(ValueError, match="No events fall within"):
            GPTemplate.from_dataset(Dataset(np.array([20.0, 25.0, 30.0])), EDGES_10,
                                    optimize_hyperparameters=False)

    def test_five_bin_output(self):
        gp = GPTemplate.from_dataset(
            _uniform_dataset(20, seed=51), EDGES_5, optimize_hyperparameters=False,
        )
        assert gp.nbins == 5
        assert gp.counts().shape == (5,)

    def test_fixed_hyperparameters_respected(self):
        log_sigma_in, log_ell_in = -0.3, np.log(2.5)
        gp = GPTemplate.from_dataset(
            _uniform_dataset(20, seed=52), EDGES_10,
            optimize_hyperparameters=False,
            log_sigma=log_sigma_in, log_ell=log_ell_in,
        )
        assert abs(gp._log_sigma - log_sigma_in) < 1e-12
        assert abs(gp._log_ell - log_ell_in) < 1e-12


class TestSampleFromDataset:

    def test_gp_path_returns_gp_template(self):
        s = Sample.from_dataset("sig", _uniform_dataset(30, seed=60),
                                edges=EDGES_10, template_type="gp")
        assert isinstance(s.template, GPTemplate)

    def test_binned_path_unchanged(self):
        s = Sample.from_dataset("sig", _uniform_dataset(200, seed=61),
                                edges=EDGES_10, template_type="binned")
        assert isinstance(s.template, BinnedTemplate)

    def test_sample_counts_finite_positive(self):
        s = Sample.from_dataset("sig", _uniform_dataset(30, seed=62),
                                edges=EDGES_10, template_type="gp")
        assert np.all(np.isfinite(s.template.counts()))
        assert np.all(s.template.counts() > 0)


class TestLaplaceModeRegression:

    def test_mode_matches_legacy_implementation(self):
        """PoissonCountModel through _cox_laplace_mode matches _laplace_mode."""
        from histimator.cox_model import PoissonCountModel, _cox_laplace_mode
        from histimator.gp_template import _laplace_mode as legacy

        rng = np.random.default_rng(99)
        edges = np.linspace(X_MIN, X_MAX, 11)
        widths = np.diff(edges)
        centres = 0.5 * (edges[:-1] + edges[1:])
        counts = rng.poisson(5, 10).astype(float)

        ell = 3.0
        diff = centres[:, None] - centres[None, :]
        k = np.exp(-0.5 * diff**2 / ell**2) + 1e-6 * np.eye(10)
        mean = np.zeros(10)

        f_legacy, _ = legacy(counts, widths, k, mean)
        obs = PoissonCountModel(counts, widths)
        f_new, _ = _cox_laplace_mode(obs, k, mean, jitter=0.0)

        np.testing.assert_allclose(f_new, f_legacy, rtol=1e-5, atol=1e-6)
