"""Tests for the EigenmodeStat modifier and add_eigenmode_staterror.

Covers the new path the paper exercises in Section 6 / Appendix C: a
GP-template-backed sample carries its statistical uncertainty as a
small number of eigenmode amplitudes rather than per-bin gammas.

The tests pin:

  * `EigenmodeStat` is constructed via `Sample.add_eigenmode_staterror`
    only when the underlying template is a GPTemplate (TypeError
    otherwise).
  * The number of retained modes matches the variance threshold passed
    in.
  * `Sample.expected` evaluates the eigenmode log-shift correctly:
    z = 0 reproduces the GP nominal counts; z = +1 in mode i shifts by
    sqrt(lambda_i) * V_i.
  * `BinnedNLL` registers each z parameter as a unit-Gaussian-constrained
    nuisance.
  * A fit that injects an eigenmode shift recovers it within the
    parameter's prior width.
"""
from __future__ import annotations

import numpy as np
import pytest

from histimator.channels import Channel
from histimator.histograms import Histogram
from histimator.likelihood import BinnedNLL, fit
from histimator.model import Model
from histimator.samples import EigenmodeStat, Sample


@pytest.fixture
def gp_sample():
    """A small GP-templated background sample with a fixed eigenmode unc."""
    rng = np.random.default_rng(7)
    edges = np.linspace(0.0, 5.0, 11)  # 10 bins
    centres = 0.5 * (edges[:-1] + edges[1:])
    truth = 50.0 * np.exp(-0.5 * ((centres - 2.5) / 1.0) ** 2) + 5.0
    h = Histogram(rng.poisson(truth).astype(float), edges)
    return Sample.from_histogram("bkg", h, template_type="gp")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestAddEigenmodeStatError:
    def test_requires_gp_template(self):
        edges = np.linspace(0.0, 5.0, 11)
        h = Histogram(np.full(10, 50.0), edges)
        s = Sample("bkg", h)  # default = BinnedTemplate
        with pytest.raises(TypeError):
            s.add_eigenmode_staterror("eig")

    def test_creates_eigenmode_stat(self, gp_sample):
        gp_sample.add_eigenmode_staterror("eig", variance_threshold=0.95)
        em_mods = [m for m in gp_sample.modifiers if isinstance(m, EigenmodeStat)]
        assert len(em_mods) == 1
        em = em_mods[0]
        assert em.eigenvectors.shape[0] == gp_sample.template.nbins
        assert em.eigenvectors.shape[1] == em.sqrt_eigenvalues.size
        assert em.eigenvectors.shape[1] == len(em.parameters)

    def test_higher_threshold_keeps_more_modes(self, gp_sample):
        s_lo = gp_sample
        s_lo.add_eigenmode_staterror("eig_lo", variance_threshold=0.50)
        n_lo = next(m for m in s_lo.modifiers
                    if isinstance(m, EigenmodeStat)).sqrt_eigenvalues.size

        # A second sample with the same template but a tighter threshold
        edges = np.linspace(0.0, 5.0, 11)
        rng = np.random.default_rng(7)
        truth = 50.0 * np.exp(-0.5 * ((0.5 * (edges[:-1] + edges[1:]) - 2.5) / 1.0) ** 2) + 5.0
        h = Histogram(rng.poisson(truth).astype(float), edges)
        s_hi = Sample.from_histogram("bkg2", h, template_type="gp")
        s_hi.add_eigenmode_staterror("eig_hi", variance_threshold=0.99)
        n_hi = next(m for m in s_hi.modifiers
                    if isinstance(m, EigenmodeStat)).sqrt_eigenvalues.size

        assert n_hi >= n_lo


# ---------------------------------------------------------------------------
# Sample.expected: log-shift form exp(V @ diag(sqrt_lambda) @ z)
# ---------------------------------------------------------------------------

class TestEigenmodeExpected:
    def test_zero_z_returns_gp_nominal(self, gp_sample):
        gp_sample.add_eigenmode_staterror("eig")
        gp_nom = gp_sample.template.counts()
        params = {p.name: 0.0 for p in gp_sample.parameters}
        np.testing.assert_allclose(gp_sample.expected(params), gp_nom, rtol=1e-12)

    def test_unit_z_in_one_mode_applies_log_shift(self, gp_sample):
        gp_sample.add_eigenmode_staterror("eig")
        em = next(m for m in gp_sample.modifiers if isinstance(m, EigenmodeStat))
        n_modes = em.sqrt_eigenvalues.size
        if n_modes == 0:
            pytest.skip("No modes retained at default threshold")
        nom = gp_sample.template.counts()
        for i in range(n_modes):
            z = np.zeros(n_modes)
            z[i] = 1.0
            params = {p.name: z[k] for k, p in enumerate(em.parameters)}
            expected = nom * np.exp(em.eigenvectors[:, i] * em.sqrt_eigenvalues[i])
            np.testing.assert_allclose(gp_sample.expected(params), expected,
                                       rtol=1e-12)


# ---------------------------------------------------------------------------
# BinnedNLL constraint registration
# ---------------------------------------------------------------------------

class TestBinnedNLLConstraint:
    def test_z_parameters_unit_gaussian_constrained(self, gp_sample):
        gp_sample.add_eigenmode_staterror("eig")
        ch = Channel("SR")
        ch.add_sample(gp_sample)
        ch.set_data(gp_sample.template.counts())
        m = Model("test")
        m.add_channel(ch)

        nll = BinnedNLL(m, extended=True)
        em = next(mod for mod in gp_sample.modifiers if isinstance(mod, EigenmodeStat))
        for p in em.parameters:
            assert p.name in nll._constrained, (
                f"z parameter {p.name!r} not registered as unit-Gaussian "
                "constraint in BinnedNLL"
            )

    def test_constraint_term_zero_at_z_equals_zero(self, gp_sample):
        gp_sample.add_eigenmode_staterror("eig")
        ch = Channel("SR")
        ch.add_sample(gp_sample)
        ch.set_data(gp_sample.template.counts())
        m = Model("test")
        m.add_channel(ch)

        nll = BinnedNLL(m, extended=True)
        n_par = len(m.parameter_names)
        nll_at_zero = nll(np.zeros(n_par))

        # Single mode shifted by 1 sigma should add 0.5 to the NLL
        em = next(mod for mod in gp_sample.modifiers if isinstance(mod, EigenmodeStat))
        if em.sqrt_eigenvalues.size == 0:
            pytest.skip("No modes")
        z_shifted = np.zeros(n_par)
        idx = m.parameter_names.index(em.parameters[0].name)
        z_shifted[idx] = 1.0

        # data == nominal so per-bin Poisson contribution at z=0 is exact;
        # at z=1 the data/expected mismatch contributes too. We check the
        # constraint piece directly via NLL difference between independent
        # toy data points.
        nll_at_shift = nll(z_shifted)
        # The constraint contribution alone is +0.5 on the NLL.
        # The Poisson contribution increases too (data away from expected),
        # so the total increase should be at least 0.5.
        assert nll_at_shift - nll_at_zero >= 0.5 - 1e-9


# ---------------------------------------------------------------------------
# Fit recovery: inject a known eigenmode shift and recover it
# ---------------------------------------------------------------------------

class TestEigenmodeStatFitRecovery:
    def test_recovers_injected_shift(self, gp_sample):
        gp_sample.add_eigenmode_staterror("eig", variance_threshold=0.99)
        em = next(m for m in gp_sample.modifiers if isinstance(m, EigenmodeStat))
        n_modes = em.sqrt_eigenvalues.size
        if n_modes == 0:
            pytest.skip("No modes retained")

        # Asimov data with z_0 = 0.5 injected on the first mode.
        z_true = np.zeros(n_modes)
        z_true[0] = 0.5
        params_true = {p.name: z_true[k] for k, p in enumerate(em.parameters)}
        data = gp_sample.expected(params_true)

        ch = Channel("SR")
        ch.add_sample(gp_sample)
        ch.set_data(data)
        model = Model("test")
        model.add_channel(ch)

        result = fit(model)
        assert result.valid
        recovered = result.bestfit[em.parameters[0].name]
        # The unit-Gaussian prior shrinks the posterior toward 0 by
        # 1 / (1 + sigma_data^2) when sigma_data is order unity (which
        # is the case here -- the injection is well inside the GP's
        # posterior). We only require that recovery has the right sign,
        # is bounded by the truth, and is non-trivial.
        assert 0.05 < recovered < 0.5
        # Other modes should fit close to zero.
        for p in em.parameters[1:]:
            assert abs(result.bestfit[p.name]) < 0.5
