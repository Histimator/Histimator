"""Performance benchmarks inspired by rootbench RooFitBinnedBenchmarks.

These benchmark the three scaling axes that matter for real-world physics
analyses: number of channels, number of bins per channel, and number of
systematic (nuisance) parameters per sample.

Run with:
    pytest benchmarks/ --benchmark-enable -v
"""

import numpy as np
import pytest

from histimator.histograms import Histogram
from histimator.samples import Sample
from histimator.channels import Channel
from histimator.model import Model
from histimator.likelihood import BinnedNLL, fit


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def build_model(n_channels=1, n_bins=10, n_systematics=0, seed=42):
    """Construct a model of configurable complexity.

    Each channel gets:
    - one signal sample with a shared normfactor "mu"
    - one background sample with per-channel norm systematics
    - Poisson pseudo-data generated at mu=1
    """
    rng = np.random.default_rng(seed)
    edges = np.linspace(0.0, 10.0, n_bins + 1)

    model = Model("benchmark")

    for ich in range(n_channels):
        # Signal: Gaussian-like peak
        centres = 0.5 * (edges[:-1] + edges[1:])
        sig_vals = 50.0 * np.exp(-0.5 * ((centres - 5.0) / 1.5) ** 2)
        bkg_vals = np.full(n_bins, 100.0)

        sig = Sample(f"signal_ch{ich}", Histogram(sig_vals, edges))
        sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 10.0))

        bkg = Sample(f"background_ch{ich}", Histogram(bkg_vals, edges))
        for isys in range(n_systematics):
            bkg.add_normsys(
                f"syst_ch{ich}_s{isys}",
                lo=0.95,
                hi=1.05,
            )

        expected = sig_vals + bkg_vals
        data = rng.poisson(expected).astype(float)

        ch = Channel(f"ch{ich}")
        ch.add_sample(sig)
        ch.add_sample(bkg)
        ch.set_data(data)
        model.add_channel(ch)

    return model


# ---------------------------------------------------------------------------
# Benchmark: NLL evaluation speed
# ---------------------------------------------------------------------------

class TestNLLEvaluationBenchmarks:
    """Benchmark raw NLL evaluation (no minimisation)."""

    @pytest.mark.benchmark(group="nll_eval")
    def test_nll_1ch_10bins(self, benchmark):
        model = build_model(n_channels=1, n_bins=10)
        nll = BinnedNLL(model)
        vals = np.array([1.0])
        benchmark(nll, vals)

    @pytest.mark.benchmark(group="nll_eval")
    def test_nll_1ch_100bins(self, benchmark):
        model = build_model(n_channels=1, n_bins=100)
        nll = BinnedNLL(model)
        vals = np.array([1.0])
        benchmark(nll, vals)

    @pytest.mark.benchmark(group="nll_eval")
    def test_nll_1ch_1000bins(self, benchmark):
        model = build_model(n_channels=1, n_bins=1000)
        nll = BinnedNLL(model)
        vals = np.array([1.0])
        benchmark(nll, vals)

    @pytest.mark.benchmark(group="nll_eval")
    def test_nll_10ch_10bins(self, benchmark):
        model = build_model(n_channels=10, n_bins=10)
        nll = BinnedNLL(model)
        vals = np.array([1.0])
        benchmark(nll, vals)


# ---------------------------------------------------------------------------
# Benchmark: full fit (migrad + hesse)
# ---------------------------------------------------------------------------

class TestFitBenchmarks:
    """Benchmark the full minimisation cycle."""

    @pytest.mark.benchmark(group="fit")
    def test_fit_1ch_10bins_0sys(self, benchmark):
        model = build_model(n_channels=1, n_bins=10, n_systematics=0)
        benchmark(fit, model)

    @pytest.mark.benchmark(group="fit")
    def test_fit_1ch_50bins_0sys(self, benchmark):
        model = build_model(n_channels=1, n_bins=50, n_systematics=0)
        benchmark(fit, model)

    @pytest.mark.benchmark(group="fit")
    def test_fit_1ch_10bins_5sys(self, benchmark):
        model = build_model(n_channels=1, n_bins=10, n_systematics=5)
        benchmark(fit, model)

    @pytest.mark.benchmark(group="fit")
    def test_fit_3ch_10bins_2sys(self, benchmark):
        model = build_model(n_channels=3, n_bins=10, n_systematics=2)
        benchmark(fit, model)

    @pytest.mark.benchmark(group="fit")
    def test_fit_5ch_20bins_3sys(self, benchmark):
        model = build_model(n_channels=5, n_bins=20, n_systematics=3)
        benchmark(fit, model)
