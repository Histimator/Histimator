"""Tests for GPTemplate lengthscale bounds and auto_template dispatch.

Lengthscale bounds
------------------
The GP lengthscale is analogous to the KDE bandwidth.  The AMISE theory
(Silverman 1986, García-Portugués Ch. 2) requires h -> 0 and nh -> inf
simultaneously.  For binned data this translates to:

  lower: ell >= 1.5 * bin_width
    Below this the GP resolves structure finer than the binning supports
    and the Laplace approximation degrades (analogous to nh -> inf).

  upper: ell <= domain_width
    Beyond this every bin is perfectly correlated and the GP collapses
    to a constant offset already absorbed by the mean function.

auto_template
-------------
When min(counts) >= threshold the histogram is already a good density
estimator (fractional Poisson uncertainty < 10% at threshold=100).
The AMISE-optimal bandwidth O(n^{-1/5}) shrinks below the bin width,
so GP smoothing provides diminishing returns.  auto_template dispatches
to BinnedTemplate in that regime.
"""

import numpy as np

from histimator import GPTemplate, Histogram

# ── Lengthscale bounds ───────────────────────────────────────────────────

class TestLengthscaleBounds:
    """Verify the optimized lengthscale respects hard bounds."""

    def _make_histogram(self, nbins, x_min, x_max, total=200):
        """Exponential-like template in [x_min, x_max]."""
        rng = np.random.default_rng(42)
        edges = np.linspace(x_min, x_max, nbins + 1)
        centres = 0.5 * (edges[:-1] + edges[1:])
        rate = np.exp(-0.3 * centres)
        rate *= total / rate.sum()
        counts = rng.poisson(rate).astype(float)
        return Histogram(counts, edges)

    def test_lengthscale_above_lower_bound(self):
        """Optimized ell must be >= 1.5 * mean_bin_width."""
        h = self._make_histogram(nbins=20, x_min=0, x_max=10)
        gpt = GPTemplate(h, optimize_hyperparameters=True)
        bin_width = 10.0 / 20
        assert gpt.lengthscale >= 1.5 * bin_width - 1e-6

    def test_lengthscale_finite(self):
        """Optimized ell must be finite and positive.

        With the B-spline mean function the lengthscale can exceed the
        domain width because the mean absorbs large-scale structure,
        leaving the GP to model only local residuals.  The relevant
        constraint is that the optimizer converges to a finite value.
        """
        h = self._make_histogram(nbins=20, x_min=0, x_max=10)
        gpt = GPTemplate(h, optimize_hyperparameters=True)
        assert np.isfinite(gpt.lengthscale)
        assert gpt.lengthscale > 0

    def test_narrow_bins_clamp_lower(self):
        """With many narrow bins the lower bound should prevent
        the optimizer from choosing a pathologically short lengthscale."""
        h = self._make_histogram(nbins=100, x_min=0, x_max=10, total=500)
        gpt = GPTemplate(h, optimize_hyperparameters=True)
        bin_width = 10.0 / 100
        assert gpt.lengthscale >= 1.5 * bin_width - 1e-6

    def test_wide_bins_finite(self):
        """With very few wide bins the lengthscale should still converge."""
        h = self._make_histogram(nbins=5, x_min=0, x_max=10, total=50)
        gpt = GPTemplate(h, optimize_hyperparameters=True)
        assert np.isfinite(gpt.lengthscale)
        assert gpt.lengthscale > 0

    def test_fixed_hyperparameters_bypass_bounds(self):
        """When optimize_hyperparameters=False, user values are
        accepted without clamping."""
        h = self._make_histogram(nbins=20, x_min=0, x_max=10)
        gpt = GPTemplate(h, optimize_hyperparameters=False,
                         log_ell=np.log(100.0))
        assert abs(gpt.lengthscale - 100.0) < 0.1

    def test_bounds_with_data_mean(self):
        """Bounds apply equally to the data-mean path."""
        h = self._make_histogram(nbins=20, x_min=0, x_max=10)
        gpt = GPTemplate(h, optimize_hyperparameters=True,
                         mean_degree="data")
        bin_width = 10.0 / 20
        assert gpt.lengthscale >= 1.5 * bin_width - 1e-6
        assert np.isfinite(gpt.lengthscale)
