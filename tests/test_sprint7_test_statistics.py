"""Tests for Sprint 7: One-sided test statistics and asymptotic p-values.

Validation strategy
-------------------
The CCGV paper (arXiv:1007.1727) defines three test statistics with
known boundary behaviour and known asymptotic distributions.  We test
each by constructing models that force mu_hat into specific regimes
(positive, negative, above/below the test value) and checking that the
boundary conditions hold exactly.  For the asymptotic p-value functions
we use closed-form results from the paper: the single-bin counting
experiment (eq 97) and the known half-chi2 limiting cases.

All models in this file use fixed seeds so results are reproducible.
"""

import numpy as np
import pytest
from scipy import stats

from histimator.channels import Channel
from histimator.histograms import Histogram
from histimator.model import Model
from histimator.samples import Sample
from histimator.test_statistics import (
    compute_q0,
    compute_qmu,
    compute_qtilde_mu,
    pvalue_q0,
    pvalues_qmu,
    pvalues_qtilde_mu,
)

EDGES = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])


# ===================================================================
# Model builders
# ===================================================================

def _strong_signal_model(seed=42):
    """Model where the data contains a clear signal excess.

    Signal template peaks at 40, background is flat 100/bin.
    Data is drawn from Poisson(2*sig + bkg), so mu_hat ~ 2.
    Because mu_hat > 0, q_0 should be positive.
    Because mu_hat ~ 2, q_mu(5) > 0 but q_mu(0.5) = 0.
    """
    rng = np.random.default_rng(seed)
    sig_t = np.array([5.0, 20.0, 40.0, 20.0, 5.0])
    bkg_t = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    data = rng.poisson(2.0 * sig_t + bkg_t).astype(float)

    sig = Sample("signal", Histogram(sig_t, EDGES))
    sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 20.0))
    bkg = Sample("background", Histogram(bkg_t, EDGES))

    ch = Channel("SR")
    ch.add_sample(sig)
    ch.add_sample(bkg)
    ch.set_data(data)

    m = Model("strong_sig")
    m.add_channel(ch)
    return m


def _deficit_model():
    """Model where the data is below the background expectation.

    Data is set to bkg - 0.4*sig (deterministic, no Poisson), so the
    unconstrained MLE wants mu_hat < 0.  This exercises the boundary
    conditions: q_0 should be 0, and q_tilde_mu should use the
    mu=0 denominator instead of the unconstrained one.
    """
    sig_t = np.array([10.0, 20.0, 30.0, 20.0, 10.0])
    bkg_t = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    data = bkg_t - 0.4 * sig_t  # deficit: data < bkg everywhere

    sig = Sample("signal", Histogram(sig_t, EDGES))
    sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 20.0))
    bkg = Sample("background", Histogram(bkg_t, EDGES))

    ch = Channel("SR")
    ch.add_sample(sig)
    ch.add_sample(bkg)
    ch.set_data(data)

    m = Model("deficit")
    m.add_channel(ch)
    return m


def _counting_model(s, b, n):
    """Single-bin counting experiment for closed-form checks.

    With one bin, the test statistics have exact analytic expressions:
        mu_hat = (n - b) / s
        q_0 = 2*(n*ln(n/b) - (n - b))   when n > b, else 0
    """
    edges = np.array([0.0, 1.0])
    sig = Sample("signal", Histogram([s], edges))
    sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 100.0))
    bkg = Sample("background", Histogram([b], edges))

    ch = Channel("count")
    ch.add_sample(sig)
    ch.add_sample(bkg)
    ch.set_data([n])

    m = Model("counting")
    m.add_channel(ch)
    return m


# ===================================================================
# Tests: compute_q0  (discovery test statistic)
# ===================================================================

class TestQ0:
    """q_0 = 2[NLL(mu=0) - NLL(mu_hat)] when mu_hat >= 0, else 0.

    The purpose of q_0 is discovery: it tests the background-only
    hypothesis.  A downward fluctuation (mu_hat < 0) should not count
    as evidence for signal, hence the boundary at zero.
    """

    def test_positive_when_signal_present(self):
        """With clear signal in data, mu_hat > 0 so q_0 > 0."""
        model = _strong_signal_model()
        q0, mu_hat = compute_q0(model, "mu")
        assert mu_hat > 0.5, "mu_hat should be positive with signal"
        assert q0 > 1.0, "q_0 should be significantly positive"

    def test_zero_when_deficit(self):
        """With data below background, mu_hat < 0 so q_0 = 0 exactly."""
        model = _deficit_model()
        q0, mu_hat = compute_q0(model, "mu")
        assert mu_hat < 0, "mu_hat should be negative for deficit data"
        assert q0 == 0.0, "q_0 must be exactly 0 when mu_hat < 0"

    def test_non_negative(self):
        """q_0 should never be negative regardless of data."""
        for seed in [1, 2, 3, 4, 5]:
            model = _strong_signal_model(seed=seed)
            q0, _ = compute_q0(model, "mu")
            assert q0 >= 0.0

    def test_counting_experiment_closed_form(self):
        """Single bin: q_0 = 2*(n*ln(n/b) - (n-b)) when n > b.

        This is the exact result for a counting experiment with no
        nuisance parameters, derived by substituting mu_hat = (n-b)/s
        into the Poisson NLL.
        """
        s, b, n = 10.0, 50.0, 65.0
        model = _counting_model(s, b, n)
        q0, mu_hat = compute_q0(model, "mu")

        expected_mu_hat = (n - b) / s
        expected_q0 = 2.0 * (n * np.log(n / b) - (n - b))

        np.testing.assert_allclose(mu_hat, expected_mu_hat, rtol=0.02)
        np.testing.assert_allclose(q0, expected_q0, rtol=0.02)

    def test_counting_experiment_deficit(self):
        """Single bin with n < b gives q_0 = 0."""
        s, b, n = 10.0, 50.0, 40.0
        model = _counting_model(s, b, n)
        q0, mu_hat = compute_q0(model, "mu")
        assert mu_hat < 0
        assert q0 == 0.0


# ===================================================================
# Tests: compute_qmu  (one-sided upper limit test statistic)
# ===================================================================

class TestQmu:
    """q_mu = 2[NLL(mu) - NLL(mu_hat)] when mu_hat <= mu, else 0.

    The purpose of q_mu is exclusion: it tests whether a signal of
    strength mu is compatible with the data.  If the data actually
    prefers more signal than mu (mu_hat > mu), that should not count
    as evidence against mu, hence the boundary at zero.
    """

    def test_positive_when_mu_test_above_bestfit(self):
        """Testing mu=5 when mu_hat ~ 2 should give q_mu > 0."""
        model = _strong_signal_model()
        qmu, mu_hat = compute_qmu(model, "mu", 5.0)
        assert mu_hat < 5.0
        assert qmu > 0.0

    def test_zero_when_mu_test_below_bestfit(self):
        """Testing mu=0.5 when mu_hat ~ 2 should give q_mu = 0."""
        model = _strong_signal_model()
        qmu, mu_hat = compute_qmu(model, "mu", 0.5)
        assert mu_hat > 0.5
        assert qmu == 0.0

    def test_non_negative(self):
        """q_mu should never be negative."""
        model = _strong_signal_model()
        for mu_test in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
            qmu, _ = compute_qmu(model, "mu", mu_test)
            assert qmu >= 0.0

    def test_monotonic_above_bestfit(self):
        """q_mu should increase as mu moves further above mu_hat.

        If mu_hat ~ 2, then q_mu(3) < q_mu(5) < q_mu(8) because
        each value is further from the best fit.
        """
        model = _strong_signal_model()
        vals = []
        for mu_test in [3.0, 5.0, 8.0]:
            qmu, _ = compute_qmu(model, "mu", mu_test)
            vals.append(qmu)
        assert vals[0] < vals[1] < vals[2]

    def test_equals_two_sided_in_correct_regime(self):
        """When mu_hat < mu_test, q_mu should equal the two-sided PLR
        computed with the same likelihood convention.

        We verify this by computing the two-sided statistic directly:
        2 * [NLL(mu_test) - NLL(mu_hat)], using the same internal
        functions that compute_qmu uses.
        """
        from histimator.test_statistics import _fit_unconstrained, _profile_nll

        model = _strong_signal_model()
        mu_test = 6.0
        qmu, mu_hat = compute_qmu(model, "mu", mu_test)

        # Compute two-sided statistic directly
        nll_uncond, _, _ = _fit_unconstrained(model, "mu")
        nll_cond = _profile_nll(model, "mu", mu_test)
        two_sided = 2.0 * (nll_cond - nll_uncond)

        assert mu_hat < mu_test
        np.testing.assert_allclose(qmu, two_sided, rtol=1e-6)


# ===================================================================
# Tests: compute_qtilde_mu  (bounded mu_hat >= 0)
# ===================================================================

class TestQtildeMu:
    """q_tilde_mu adds the physical boundary mu_hat >= 0 to q_mu.

    When mu_hat >= 0, q_tilde_mu equals q_mu.  When mu_hat < 0,
    the denominator switches from NLL(mu_hat) to NLL(0), which
    gives a different (larger) test statistic than would be obtained
    by simply setting mu_hat = 0.
    """

    def test_equals_qmu_when_muhat_positive(self):
        """When mu_hat > 0, q_tilde_mu should equal q_mu."""
        model = _strong_signal_model()
        mu_test = 5.0
        qtilde, mu_hat = compute_qtilde_mu(model, "mu", mu_test)
        qmu, _ = compute_qmu(model, "mu", mu_test)

        assert mu_hat > 0
        np.testing.assert_allclose(qtilde, qmu, rtol=1e-4)

    def test_zero_when_muhat_above_mu(self):
        """When mu_hat > mu_test, q_tilde_mu = 0, same as q_mu."""
        model = _strong_signal_model()
        qtilde, mu_hat = compute_qtilde_mu(model, "mu", 0.5)
        assert mu_hat > 0.5
        assert qtilde == 0.0

    def test_positive_when_deficit(self):
        """With deficit data (mu_hat < 0), q_tilde_mu should still be
        positive for any mu_test > 0 because the data disfavours signal.

        The denominator uses NLL(0) instead of NLL(mu_hat), so the
        statistic is 2[NLL(mu) - NLL(0)] which is positive because
        mu > 0 is further from the data than mu = 0.
        """
        model = _deficit_model()
        qtilde, mu_hat = compute_qtilde_mu(model, "mu", 2.0)
        assert mu_hat < 0
        assert qtilde > 0.0

    def test_deficit_less_than_qmu(self):
        """When mu_hat < 0, q_tilde_mu < q_mu because q_tilde_mu uses
        NLL(0) as denominator instead of the smaller NLL(mu_hat).

        NLL(mu_hat) <= NLL(0) since mu_hat is the global minimum, so
        q_mu = 2[NLL(mu) - NLL(mu_hat)] >= 2[NLL(mu) - NLL(0)] = q_tilde_mu.
        """
        model = _deficit_model()
        mu_test = 2.0
        qmu, mu_hat_1 = compute_qmu(model, "mu", mu_test)
        qtilde, mu_hat_2 = compute_qtilde_mu(model, "mu", mu_test)

        assert mu_hat_1 < 0
        assert qmu > qtilde, (
            "q_mu should be larger because it uses NLL(mu_hat) < NLL(0)"
        )

    def test_non_negative(self):
        """q_tilde_mu should never be negative."""
        model = _deficit_model()
        for mu_test in [0.1, 0.5, 1.0, 3.0, 5.0]:
            qtilde, _ = compute_qtilde_mu(model, "mu", mu_test)
            assert qtilde >= 0.0


# ===================================================================
# Tests: pvalue_q0  (asymptotic p-value for discovery)
# ===================================================================

class TestPvalueQ0:
    """p_0 = 1 - Phi(sqrt(q_0)) for q_0 > 0; p_0 = 0.5 for q_0 = 0.

    The asymptotic distribution of q_0 under mu=0 is a half-chi2(1):
    half the time mu_hat < 0 and q_0 = 0, half the time q_0 ~ chi2(1).
    """

    def test_zero_gives_half(self):
        """q_0 = 0 means no evidence for signal, p-value = 0.5."""
        assert pvalue_q0(0.0) == 0.5

    def test_large_q0_small_pvalue(self):
        """q_0 = 25 corresponds to a 5-sigma discovery (p ~ 2.87e-7)."""
        p = pvalue_q0(25.0)
        z = np.sqrt(25.0)
        expected_p = stats.norm.sf(z)
        np.testing.assert_allclose(p, expected_p, rtol=1e-10)

    def test_moderate_q0(self):
        """q_0 = 4 corresponds to 2-sigma (p ~ 0.0228)."""
        p = pvalue_q0(4.0)
        np.testing.assert_allclose(p, stats.norm.sf(2.0), rtol=1e-10)

    def test_consistency_with_z(self):
        """p = 1 - Phi(sqrt(q_0)) for several values."""
        for q0 in [0.5, 1.0, 2.0, 9.0, 16.0]:
            p = pvalue_q0(q0)
            expected = stats.norm.sf(np.sqrt(q0))
            np.testing.assert_allclose(p, expected, rtol=1e-10)


# ===================================================================
# Tests: pvalues_qmu  (asymptotic p-values for q_mu)
# ===================================================================

class TestPvaluesQmu:
    """Returns (CLsb, CLb) for the q_mu test statistic.

    CLsb = 1 - Phi(sqrt(q_mu))
    CLb  = 1 - Phi(sqrt(q_mu) - mu/sigma)

    where sigma is the standard deviation of mu_hat.
    """

    def test_zero_qmu(self):
        """q_mu = 0 means mu_hat > mu: data prefers more signal.

        CLsb should be 0.5 (not excluded).  CLb depends on mu/sigma.
        """
        clsb, clb = pvalues_qmu(0.0, mu_test=1.0, sigma=1.0)
        assert clsb == 0.5

    def test_known_values(self):
        """Check against hand-computed values.

        q_mu = 4.0, mu = 2.0, sigma = 1.0:
        CLsb = 1 - Phi(2) = 0.02275
        CLb = 1 - Phi(2 - 2) = 0.5
        CLs = CLsb / CLb = 0.04550
        """
        clsb, clb = pvalues_qmu(4.0, mu_test=2.0, sigma=1.0)
        np.testing.assert_allclose(clsb, stats.norm.sf(2.0), rtol=1e-10)
        np.testing.assert_allclose(clb, 0.5, rtol=1e-10)

    def test_large_mu_over_sigma_gives_clb_near_one(self):
        """When mu/sigma is large, CLb approaches 1.

        This corresponds to a scenario with high sensitivity: under
        background-only, the test statistic would be much larger than
        observed, so nearly all background toys would exceed it.
        """
        _, clb = pvalues_qmu(1.0, mu_test=5.0, sigma=1.0)
        assert clb > 0.99

    def test_cls_ratio(self):
        """CLs = CLsb / CLb should decrease as q_mu increases."""
        cls_values = []
        for qmu in [1.0, 4.0, 9.0]:
            clsb, clb = pvalues_qmu(qmu, mu_test=2.0, sigma=1.0)
            cls_values.append(clsb / clb)
        assert cls_values[0] > cls_values[1] > cls_values[2]


# ===================================================================
# Tests: pvalues_qtilde_mu  (asymptotic p-values for q_tilde_mu)
# ===================================================================

class TestPvaluesQtildeMu:
    """Returns (CLsb, CLb) for the q_tilde_mu test statistic.

    When mu_hat >= 0, the formulas are the same as for q_mu.
    When mu_hat < 0, the test statistic has a different form and the
    p-value computation uses different expressions from CCGV eqs 64-66.
    """

    def test_matches_qmu_when_muhat_positive(self):
        """When mu_hat > 0, q_tilde_mu p-values should equal q_mu p-values."""
        mu_hat = 1.0
        qtilde = 4.0
        mu_test = 3.0
        sigma = 1.0

        clsb_t, clb_t = pvalues_qtilde_mu(
            qtilde, mu_test, mu_hat, sigma
        )
        clsb_q, clb_q = pvalues_qmu(qtilde, mu_test, sigma)

        np.testing.assert_allclose(clsb_t, clsb_q, rtol=1e-10)
        np.testing.assert_allclose(clb_t, clb_q, rtol=1e-10)

    def test_muhat_negative_clsb(self):
        """When mu_hat < 0, the CLsb formula changes per CCGV eq 66.

        CLsb = 1 - Phi((q_tilde - mu^2/sigma^2) / (2*mu/sigma))

        This is equivalent to Phi(-sqrt(q_tilde) + mu/sigma) when
        expressed in terms of sqrt(q_tilde).
        """
        mu_hat = -1.0
        mu_test = 2.0
        sigma = 1.0
        qtilde = mu_test**2 / sigma**2 - 2 * mu_test * mu_hat / sigma**2

        clsb, clb = pvalues_qtilde_mu(qtilde, mu_test, mu_hat, sigma)

        # CLsb should be a valid probability
        assert 0.0 < clsb < 1.0

    def test_muhat_negative_clb_valid(self):
        """When mu_hat < 0, CLb should be a valid probability."""
        mu_hat = -1.0
        mu_test = 2.0
        sigma = 1.0
        qtilde = mu_test**2 / sigma**2 - 2 * mu_test * mu_hat / sigma**2

        _, clb = pvalues_qtilde_mu(qtilde, mu_test, mu_hat, sigma)
        assert 0.0 < clb < 1.0

    def test_cls_well_defined(self):
        """CLs = CLsb / CLb should be between 0 and 1 for all regimes."""
        cases = [
            (1.0, 4.0, 3.0, 1.0),   # mu_hat > 0
            (-1.0, 8.0, 2.0, 1.0),  # mu_hat < 0
            (0.0, 4.0, 2.0, 1.0),   # mu_hat = 0 boundary
        ]
        for mu_hat, qtilde, mu_test, sigma in cases:
            clsb, clb = pvalues_qtilde_mu(qtilde, mu_test, mu_hat, sigma)
            if clb > 0:
                cls = clsb / clb
                assert 0.0 <= cls <= 1.0 + 1e-10


# ===================================================================
# Integration test: q_0 + pvalue_q0 end-to-end
# ===================================================================

class TestQ0EndToEnd:
    """Check that compute_q0 and pvalue_q0 are mutually consistent:
    the p-value and z-value should satisfy the standard relationship
    p = 1 - Phi(z) where z = sqrt(q_0).
    """

    def test_z_p_consistency(self):
        model = _strong_signal_model(seed=77)
        q0, mu_hat = compute_q0(model, "mu")
        p0 = pvalue_q0(q0)

        assert mu_hat > 0, "This model should have positive mu_hat"
        assert q0 > 0, "q_0 should be positive for this model"

        z = np.sqrt(q0)
        expected_p = stats.norm.sf(z)
        np.testing.assert_allclose(p0, expected_p, rtol=1e-10)

    def test_deficit_gives_half(self):
        """Deficit model should give q_0 = 0 and p = 0.5."""
        model = _deficit_model()
        q0, _ = compute_q0(model, "mu")
        p0 = pvalue_q0(q0)

        assert q0 == 0.0
        assert p0 == 0.5


# ===================================================================
# Tests: Asimov significance closed form (CCGV eq 97)
# ===================================================================

class TestAsimovSignificance:
    """For a counting experiment with Asimov data (n = s + b exactly),
    the discovery significance has a known closed form:

        Z_A = sqrt(2 * ((s+b)*ln(1 + s/b) - s))

    This tests compute_q0 on deterministic Asimov data against the
    analytic result.
    """

    def test_moderate_signal(self):
        """s=20, b=100: Z_A ~ 1.956."""
        s, b = 20.0, 100.0
        n = s + b  # Asimov data

        model = _counting_model(s, b, n)
        q0, mu_hat = compute_q0(model, "mu")

        expected_z = np.sqrt(2.0 * ((s + b) * np.log(1 + s / b) - s))
        expected_q0 = expected_z**2

        np.testing.assert_allclose(q0, expected_q0, rtol=0.01)

    def test_strong_signal(self):
        """s=50, b=50: Z_A ~ 5.057."""
        s, b = 50.0, 50.0
        n = s + b

        model = _counting_model(s, b, n)
        q0, _ = compute_q0(model, "mu")

        expected_z = np.sqrt(2.0 * ((s + b) * np.log(1 + s / b) - s))
        np.testing.assert_allclose(np.sqrt(q0), expected_z, rtol=0.01)

    def test_weak_signal(self):
        """s=5, b=200: Z_A ~ 0.352."""
        s, b = 5.0, 200.0
        n = s + b

        model = _counting_model(s, b, n)
        q0, _ = compute_q0(model, "mu")

        expected_z = np.sqrt(2.0 * ((s + b) * np.log(1 + s / b) - s))
        np.testing.assert_allclose(np.sqrt(q0), expected_z, rtol=0.02)
