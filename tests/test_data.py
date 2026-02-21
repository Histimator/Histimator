"""Tests for Sprint 2: Dataset container.

The mathematical contract under test is that to_histogram correctly computes:
  - unweighted: counts_j = number of events in bin j, sumw2_j = counts_j
  - weighted: counts_j = sum(w_i for i in bin j), sumw2_j = sum(w_i^2 for i in bin j)

The sumw2 computation is critical for Barlow-Beeston.  The effective number
of events is n_eff = (sum w)^2 / sumw2, which must equal n for unit weights.
"""

import numpy as np
import pytest

from histimator.channels import Channel
from histimator.data import Dataset
from histimator.histograms import Histogram
from histimator.likelihood import fit
from histimator.model import Model
from histimator.samples import Sample


# ── Construction and basic properties ────────────────────────────────────

class TestDatasetConstruction:
    def test_basic(self):
        ds = Dataset([1.0, 2.0, 3.0])
        assert len(ds) == 3 and not ds.is_weighted

    def test_weighted(self):
        ds = Dataset([1.0, 2.0], weights=[0.5, 1.5])
        assert ds.is_weighted and len(ds) == 2

    def test_empty(self):
        ds = Dataset([])
        assert len(ds) == 0

    def test_single_event(self):
        ds = Dataset([5.0])
        assert len(ds) == 1

    def test_weight_length_mismatch(self):
        with pytest.raises(ValueError, match="weights length"):
            Dataset([1.0, 2.0, 3.0], weights=[0.5, 1.0])

    def test_values_are_float64(self):
        ds = Dataset([1, 2, 3])
        assert ds.values.dtype == np.float64

    def test_repr(self):
        assert "unweighted" in repr(Dataset([1.0, 2.0]))
        assert "weighted" in repr(Dataset([1.0, 2.0], weights=[1.0, 1.0]))


# ── Unweighted histogramming ─────────────────────────────────────────────

class TestUnweightedHistogramming:
    def test_matches_numpy_histogram(self):
        """to_histogram must produce identical counts to numpy.histogram."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 1000)
        edges = np.linspace(-3, 3, 21)
        ds = Dataset(data)
        h = ds.to_histogram(edges)
        np_counts, _ = np.histogram(data, bins=edges)
        np.testing.assert_array_equal(h.values, np_counts)

    def test_sumw2_equals_counts_unweighted(self):
        """For unweighted data, sumw2_j = n_j (Poisson variance)."""
        rng = np.random.default_rng(42)
        ds = Dataset(rng.normal(0, 1, 500))
        h = ds.to_histogram(np.linspace(-3, 3, 11))
        np.testing.assert_array_equal(h.sumw2, h.values)

    def test_events_outside_range_excluded(self):
        """Events outside bin edges are silently excluded (numpy convention)."""
        ds = Dataset([0.5, 1.5, 2.5, 10.0, -5.0])
        h = ds.to_histogram([0.0, 1.0, 2.0, 3.0])
        assert h.total == 3.0  # only 3 events inside [0, 3)

    def test_total_matches_in_range_events(self):
        rng = np.random.default_rng(42)
        data = rng.uniform(-1, 1, 200)
        edges = np.array([-0.5, 0.0, 0.5])
        ds = Dataset(data)
        h = ds.to_histogram(edges)
        expected = np.sum((data >= -0.5) & (data < 0.5))
        # numpy histogram includes the last bin's right edge
        np.testing.assert_allclose(h.total, expected, atol=1)

    def test_empty_dataset_gives_zero_histogram(self):
        ds = Dataset([])
        h = ds.to_histogram([0.0, 1.0, 2.0])
        np.testing.assert_array_equal(h.values, [0.0, 0.0])

    def test_all_events_in_one_bin(self):
        ds = Dataset([0.5, 0.5, 0.5])
        h = ds.to_histogram([0.0, 1.0, 2.0])
        assert h.values[0] == 3.0 and h.values[1] == 0.0


# ── Weighted histogramming ───────────────────────────────────────────────

class TestWeightedHistogramming:
    def test_sumw_computed_correctly(self):
        """counts_j = sum(w_i for i in bin j)."""
        # 3 events: {x=0.5, w=2}, {x=0.5, w=3}, {x=1.5, w=5}
        ds = Dataset([0.5, 0.5, 1.5], weights=[2.0, 3.0, 5.0])
        h = ds.to_histogram([0.0, 1.0, 2.0])
        np.testing.assert_allclose(h.values, [5.0, 5.0])

    def test_sumw2_computed_correctly(self):
        """sumw2_j = sum(w_i^2 for i in bin j).

        Hand-calculated: bin 0 has w=2,3 so sumw2 = 4+9 = 13.
                         bin 1 has w=5 so sumw2 = 25.
        """
        ds = Dataset([0.5, 0.5, 1.5], weights=[2.0, 3.0, 5.0])
        h = ds.to_histogram([0.0, 1.0, 2.0])
        np.testing.assert_allclose(h.sumw2, [13.0, 25.0])

    def test_effective_events(self):
        """n_eff = (sum w)^2 / sumw2 must equal n for unit weights."""
        ds = Dataset([0.1, 0.2, 0.3, 0.4], weights=[1.0, 1.0, 1.0, 1.0])
        h = ds.to_histogram([0.0, 0.5])
        n_eff = h.values[0] ** 2 / h.sumw2[0]
        np.testing.assert_allclose(n_eff, 4.0)

    def test_weighted_events_outside_range(self):
        ds = Dataset([0.5, 5.0], weights=[3.0, 7.0])
        h = ds.to_histogram([0.0, 1.0])
        np.testing.assert_allclose(h.values, [3.0])
        np.testing.assert_allclose(h.sumw2, [9.0])

    def test_empty_bins_have_zero_sumw2(self):
        ds = Dataset([0.5], weights=[2.0])
        h = ds.to_histogram([0.0, 1.0, 2.0])
        assert h.values[1] == 0.0 and h.sumw2[1] == 0.0

    def test_large_weighted_sample(self):
        """Statistical test: sumw2 for uniform weights should equal n * w^2."""
        rng = np.random.default_rng(42)
        n = 5000
        data = rng.uniform(0, 10, n)
        w = 2.5
        ds = Dataset(data, weights=np.full(n, w))
        h = ds.to_histogram(np.linspace(0, 10, 11))
        # Each bin has ~500 events, sumw2 ~ 500 * 6.25 = 3125
        for j in range(h.nbins):
            nj = int(round(h.values[j] / w))
            np.testing.assert_allclose(h.sumw2[j], nj * w**2, rtol=0.01)


# ── Bootstrap resampling ─────────────────────────────────────────────────

class TestBootstrapResample:
    def test_same_length(self):
        ds = Dataset(np.arange(100, dtype=float))
        bs = ds.bootstrap_resample(np.random.default_rng(42))
        assert len(bs) == len(ds)

    def test_different_from_original(self):
        ds = Dataset(np.arange(100, dtype=float))
        bs = ds.bootstrap_resample(np.random.default_rng(42))
        assert not np.array_equal(bs.values, ds.values)

    def test_preserves_weights(self):
        ds = Dataset([1.0, 2.0, 3.0], weights=[0.5, 1.0, 1.5])
        bs = ds.bootstrap_resample(np.random.default_rng(42))
        assert bs.is_weighted and len(bs) == 3

    def test_unweighted_stays_unweighted(self):
        ds = Dataset([1.0, 2.0, 3.0])
        bs = ds.bootstrap_resample(np.random.default_rng(42))
        assert not bs.is_weighted

    def test_empty_dataset(self):
        ds = Dataset([])
        bs = ds.bootstrap_resample(np.random.default_rng(42))
        assert len(bs) == 0


# ── Sample.from_dataset ─────────────────────────────────────────────────

class TestSampleFromDataset:
    def test_basic(self):
        ds = Dataset(np.concatenate([np.full(100, 0.5), np.full(200, 1.5)]))
        s = Sample.from_dataset("proc", ds, edges=[0.0, 1.0, 2.0])
        np.testing.assert_array_equal(s.expected({}), [100, 200])

    def test_modifiers_work(self):
        ds = Dataset(np.full(100, 0.5))
        s = Sample.from_dataset("sig", ds, edges=[0.0, 1.0])
        s.add_normfactor("mu", nominal=1.0, bounds=(0.0, 5.0))
        np.testing.assert_array_equal(s.expected({"mu": 3.0}), [300])

    def test_rejects_non_dataset(self):
        with pytest.raises(TypeError, match="Dataset"):
            Sample.from_dataset("bad", [1.0, 2.0], edges=[0.0, 1.0, 2.0])

    def test_requires_edges_or_strategy(self):
        with pytest.raises(ValueError, match="edges or strategy"):
            Sample.from_dataset("bad", Dataset([1.0, 2.0]))


# ── Channel.set_data_from_dataset ────────────────────────────────────────

class TestChannelSetDataFromDataset:
    def test_basic(self):
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        sig = Sample("signal", Histogram([10, 20, 30], edges))
        ch = Channel("SR")
        ch.add_sample(sig)
        ds = Dataset(np.concatenate([np.full(15, 0.5), np.full(25, 1.5), np.full(35, 2.5)]))
        ch.set_data_from_dataset(ds)
        np.testing.assert_array_equal(ch.data.values, [15, 25, 35])

    def test_uses_channel_edges(self):
        edges = np.array([0.0, 1.0, 2.0])
        sig = Sample("signal", Histogram([10, 20], edges))
        ch = Channel("SR")
        ch.add_sample(sig)
        ds = Dataset([0.5, 0.5, 1.5])
        ch.set_data_from_dataset(ds)
        assert ch.data.nbins == 2


# ── Full model integration ───────────────────────────────────────────────

class TestDatasetModelIntegration:
    def test_dataset_model_matches_manual(self):
        """A model built from datasets must produce identical fit results
        to the manually constructed equivalent."""
        edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        rng = np.random.default_rng(42)

        # Generate raw events from signal and background distributions
        sig_events = rng.choice(
            np.arange(0.5, 5.0, 1.0), size=1000, p=[0.1, 0.3, 0.3, 0.2, 0.1]
        )
        bkg_events = rng.uniform(0, 5, 2000)
        data_events = np.concatenate([sig_events, bkg_events])

        # Path A: manual histogram construction
        sig_h = Histogram(*np.histogram(sig_events, bins=edges))
        bkg_h = Histogram(*np.histogram(bkg_events, bins=edges))
        data_h = Histogram(*np.histogram(data_events, bins=edges))
        sig_a = Sample("signal", sig_h)
        sig_a.add_normfactor("mu", nominal=1.0, bounds=(0.0, 5.0))
        bkg_a = Sample("background", bkg_h)
        ch_a = Channel("SR")
        ch_a.add_sample(sig_a)
        ch_a.add_sample(bkg_a)
        ch_a.set_data(data_h)
        m_a = Model("manual")
        m_a.add_channel(ch_a)
        r_a = fit(m_a)

        # Path B: dataset construction
        sig_b = Sample.from_dataset("signal", Dataset(sig_events), edges=edges)
        sig_b.add_normfactor("mu", nominal=1.0, bounds=(0.0, 5.0))
        bkg_b = Sample.from_dataset("background", Dataset(bkg_events), edges=edges)
        ch_b = Channel("SR")
        ch_b.add_sample(sig_b)
        ch_b.add_sample(bkg_b)
        ch_b.set_data_from_dataset(Dataset(data_events))
        m_b = Model("dataset")
        m_b.add_channel(ch_b)
        r_b = fit(m_b)

        np.testing.assert_allclose(r_a.bestfit["mu"], r_b.bestfit["mu"], rtol=1e-10)
        np.testing.assert_allclose(r_a.nll_min, r_b.nll_min, rtol=1e-10)
