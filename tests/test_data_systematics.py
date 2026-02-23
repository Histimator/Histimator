"""Tests for Sprint 5: Systematic variations from datasets.

The complete workflow test builds a multi-channel model entirely from
raw Dataset objects and runs the full inference chain (fit, discovery
significance, CLs limit).  The results must be statistically consistent
with the manually-constructed equivalent model, proving that the data
loader pathway preserves statistical integrity end-to-end.
"""

import numpy as np
import pytest

from histimator.channels import Channel
from histimator.cls import cls_observed
from histimator.data import Dataset
from histimator.histograms import Histogram
from histimator.inference import discovery_significance
from histimator.interpolation import InterpolationCode
from histimator.likelihood import fit
from histimator.model import Model
from histimator.samples import Sample

EDGES = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])


def _make_events(rng, n, loc=2.5, scale=0.8):
    """Generate signal-like events from a Gaussian."""
    return rng.normal(loc, scale, n)


def _make_bkg_events(rng, n, lo=0.0, hi=5.0):
    """Generate flat background events."""
    return rng.uniform(lo, hi, n)


# ── HistoSys from datasets ───────────────────────────────────────────────

class TestHistoSysFromDatasets:
    def test_edge_consistency(self):
        """All three histograms (nominal, lo, hi) must share identical edges."""
        rng = np.random.default_rng(42)
        nom_ds = Dataset(_make_bkg_events(rng, 5000))
        lo_ds = Dataset(_make_bkg_events(rng, 4500))
        hi_ds = Dataset(_make_bkg_events(rng, 5500))

        bkg = Sample.from_dataset("background", nom_ds, edges=EDGES)
        bkg.add_histosys_from_datasets("bkg_syst", lo_ds, hi_ds)

        # The HistoSys modifier's lo/hi histograms must have same edges
        from histimator.samples import HistoSys
        hsys = [m for m in bkg.modifiers if isinstance(m, HistoSys)][0]
        np.testing.assert_array_equal(hsys.lo_hist.edges, EDGES)
        np.testing.assert_array_equal(hsys.hi_hist.edges, EDGES)

    def test_matches_manual_construction(self):
        """HistoSys from datasets must produce the same expected values
        as manually constructing histograms and passing them directly."""
        rng = np.random.default_rng(42)
        nom_events = _make_bkg_events(rng, 3000)
        lo_events = _make_bkg_events(rng, 2700)
        hi_events = _make_bkg_events(rng, 3300)

        nom_h = Dataset(nom_events).to_histogram(EDGES)
        lo_h = Dataset(lo_events).to_histogram(EDGES)
        hi_h = Dataset(hi_events).to_histogram(EDGES)

        # Manual path
        bkg_manual = Sample("bkg", nom_h)
        bkg_manual.add_histosys("syst", lo_h, hi_h,
                                interp_code=InterpolationCode.PIECEWISE_LINEAR)

        # Dataset path
        bkg_ds = Sample.from_dataset("bkg", Dataset(nom_events), edges=EDGES)
        bkg_ds.add_histosys_from_datasets(
            "syst", Dataset(lo_events), Dataset(hi_events),
            interp_code=InterpolationCode.PIECEWISE_LINEAR,
        )

        # At alpha=0, both should give nominal
        np.testing.assert_array_equal(
            bkg_manual.expected({"syst": 0.0}),
            bkg_ds.expected({"syst": 0.0}),
        )
        # At alpha=1, both should give hi variation
        np.testing.assert_array_equal(
            bkg_manual.expected({"syst": 1.0}),
            bkg_ds.expected({"syst": 1.0}),
        )
        # At alpha=-1, both should give lo variation
        np.testing.assert_array_equal(
            bkg_manual.expected({"syst": -1.0}),
            bkg_ds.expected({"syst": -1.0}),
        )

    def test_uses_sample_edges_by_default(self):
        """When no edges provided, uses the sample's template edges."""
        rng = np.random.default_rng(42)
        bkg = Sample.from_dataset("bkg", Dataset(_make_bkg_events(rng, 3000)), edges=EDGES)
        bkg.add_histosys_from_datasets(
            "syst",
            Dataset(_make_bkg_events(rng, 2700)),
            Dataset(_make_bkg_events(rng, 3300)),
        )
        assert len(bkg.modifiers) == 1

    def test_rejects_non_dataset(self):
        bkg = Sample("bkg", Histogram([50] * 5, EDGES))
        with pytest.raises(TypeError, match="Dataset"):
            bkg.add_histosys_from_datasets("syst", [1, 2], Dataset([3, 4]))


# ── StatError from dataset ───────────────────────────────────────────────

class TestStatErrorFromDataset:
    def test_unweighted_relative_uncertainty(self):
        """For unweighted data, rel_unc_j = 1/sqrt(n_j)."""
        rng = np.random.default_rng(42)
        events = _make_bkg_events(rng, 5000)
        ds = Dataset(events)
        bkg = Sample.from_dataset("bkg", ds, edges=EDGES)
        bkg.add_staterror_from_dataset("mc_stat", ds)

        h = ds.to_histogram(EDGES)
        expected_rel = 1.0 / np.sqrt(np.maximum(h.values, 1e-10))

        from histimator.samples import StatError
        se = [m for m in bkg.modifiers if isinstance(m, StatError)][0]
        np.testing.assert_allclose(se.rel_uncertainties, expected_rel, rtol=1e-10)

    def test_weighted_uses_sumw2(self):
        """For weighted data, rel_unc uses sqrt(sumw2)/sumw."""
        rng = np.random.default_rng(42)
        events = rng.uniform(0, 5, 3000)
        weights = rng.exponential(2.0, 3000)
        ds = Dataset(events, weights=weights)
        bkg = Sample.from_dataset("bkg", ds, edges=EDGES)
        bkg.add_staterror_from_dataset("mc_stat", ds)

        h = ds.to_histogram(EDGES)
        expected_rel = np.sqrt(h.sumw2) / np.maximum(h.values, 1e-10)

        from histimator.samples import StatError
        se = [m for m in bkg.modifiers if isinstance(m, StatError)][0]
        np.testing.assert_allclose(se.rel_uncertainties, expected_rel, rtol=1e-10)

    def test_from_own_histogram(self):
        """When no dataset provided, uses the sample's own histogram."""
        bkg = Sample("bkg", Histogram([100, 200, 300, 200, 100], EDGES))
        bkg.add_staterror_from_dataset("mc_stat")

        from histimator.samples import StatError
        se = [m for m in bkg.modifiers if isinstance(m, StatError)][0]
        expected_rel = 1.0 / np.sqrt(np.array([100, 200, 300, 200, 100]))
        np.testing.assert_allclose(se.rel_uncertainties, expected_rel, rtol=1e-10)

    def test_parameter_count(self):
        bkg = Sample("bkg", Histogram([100, 200, 300, 200, 100], EDGES))
        bkg.add_staterror_from_dataset("mc_stat")
        assert len(bkg.parameters) == 5  # one gamma per bin


# ── Channel convenience method ───────────────────────────────────────────

class TestChannelFromDatasets:
    def test_basic_workflow(self):
        rng = np.random.default_rng(42)
        sig_ds = Dataset(_make_events(rng, 500))
        bkg_ds = Dataset(_make_bkg_events(rng, 3000))
        data_ds = Dataset(np.concatenate([
            _make_events(rng, 500), _make_bkg_events(rng, 3000)
        ]))

        ch = Channel("SR")
        ch.add_data_and_samples_from_datasets(
            {"signal": sig_ds, "background": bkg_ds},
            data_dataset=data_ds,
            edges=EDGES,
        )
        assert ch.data is not None and len(ch.samples) == 2

    def test_auto_data_from_samples(self):
        """When no data_dataset, uses sum of sample expectations."""
        rng = np.random.default_rng(42)
        ch = Channel("SR")
        ch.add_data_and_samples_from_datasets(
            {"signal": Dataset(_make_events(rng, 500)),
             "background": Dataset(_make_bkg_events(rng, 3000))},
            edges=EDGES,
        )
        expected = ch.expected({})
        np.testing.assert_allclose(ch.data.values, expected)

    def test_requires_edges_or_strategy(self):
        with pytest.raises(ValueError, match="edges or strategy"):
            Channel("SR").add_data_and_samples_from_datasets(
                {"sig": Dataset([1.0, 2.0])}
            )

    def test_with_strategy(self):
        rng = np.random.default_rng(42)
        ch = Channel("SR")
        ch.add_data_and_samples_from_datasets(
            {"background": Dataset(_make_bkg_events(rng, 3000))},
            edges=None,
            strategy="scott",
        )
        assert ch.data is not None


# ── Full workflow integration ────────────────────────────────────────────

class TestFullWorkflow:
    def _build_manual_model(self, rng):
        """Build model manually from histograms."""
        sig_events = _make_events(rng, 500, loc=2.5, scale=0.8)
        bkg_events = _make_bkg_events(rng, 3000)
        bkg_lo_events = _make_bkg_events(rng, 2700)
        bkg_hi_events = _make_bkg_events(rng, 3300)
        data_events = np.concatenate([
            _make_events(rng, 750, loc=2.5, scale=0.8),
            _make_bkg_events(rng, 3000),
        ])

        sig_h = Histogram(*np.histogram(sig_events, bins=EDGES))
        bkg_h = Histogram(*np.histogram(bkg_events, bins=EDGES))
        bkg_lo_h = Histogram(*np.histogram(bkg_lo_events, bins=EDGES))
        bkg_hi_h = Histogram(*np.histogram(bkg_hi_events, bins=EDGES))
        data_h = Histogram(*np.histogram(data_events, bins=EDGES))

        sig = Sample("signal", sig_h)
        sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 10.0))
        bkg = Sample("background", bkg_h)
        bkg.add_histosys("bkg_syst", bkg_lo_h, bkg_hi_h,
                         interp_code=InterpolationCode.PIECEWISE_LINEAR)

        ch = Channel("SR")
        ch.add_sample(sig)
        ch.add_sample(bkg)
        ch.set_data(data_h)
        m = Model("manual")
        m.add_channel(ch)
        return m

    def _build_dataset_model(self, rng):
        """Build identical model from datasets."""
        sig_events = _make_events(rng, 500, loc=2.5, scale=0.8)
        bkg_events = _make_bkg_events(rng, 3000)
        bkg_lo_events = _make_bkg_events(rng, 2700)
        bkg_hi_events = _make_bkg_events(rng, 3300)
        data_events = np.concatenate([
            _make_events(rng, 750, loc=2.5, scale=0.8),
            _make_bkg_events(rng, 3000),
        ])

        sig = Sample.from_dataset("signal", Dataset(sig_events), edges=EDGES)
        sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 10.0))
        bkg = Sample.from_dataset("background", Dataset(bkg_events), edges=EDGES)
        bkg.add_histosys_from_datasets(
            "bkg_syst", Dataset(bkg_lo_events), Dataset(bkg_hi_events),
            interp_code=InterpolationCode.PIECEWISE_LINEAR,
        )

        ch = Channel("SR")
        ch.add_sample(sig)
        ch.add_sample(bkg)
        ch.set_data_from_dataset(Dataset(data_events))
        m = Model("dataset")
        m.add_channel(ch)
        return m

    def test_fit_results_match(self):
        """Fit results from dataset path must match manual path exactly."""
        m_manual = self._build_manual_model(np.random.default_rng(42))
        m_dataset = self._build_dataset_model(np.random.default_rng(42))

        r_m = fit(m_manual)
        r_d = fit(m_dataset)

        np.testing.assert_allclose(r_m.bestfit["mu"], r_d.bestfit["mu"], rtol=1e-10)
        np.testing.assert_allclose(r_m.nll_min, r_d.nll_min, rtol=1e-10)
        np.testing.assert_allclose(
            r_m.bestfit["bkg_syst"], r_d.bestfit["bkg_syst"], rtol=1e-10
        )

    def test_discovery_significance_matches(self):
        """Discovery significance from both paths must agree."""
        m_manual = self._build_manual_model(np.random.default_rng(42))
        m_dataset = self._build_dataset_model(np.random.default_rng(42))

        z_m, p_m = discovery_significance(m_manual, "mu")
        z_d, p_d = discovery_significance(m_dataset, "mu")

        np.testing.assert_allclose(z_m, z_d, rtol=1e-6)
        np.testing.assert_allclose(p_m, p_d, rtol=1e-6)

    def test_cls_matches(self):
        """CLs observed values from both paths must agree."""
        m_manual = self._build_manual_model(np.random.default_rng(42))
        m_dataset = self._build_dataset_model(np.random.default_rng(42))

        cls_m = cls_observed(m_manual, "mu", 2.0)
        cls_d = cls_observed(m_dataset, "mu", 2.0)

        np.testing.assert_allclose(cls_m, cls_d, rtol=1e-4)

    def test_full_chain_with_staterror(self):
        """Complete workflow with both HistoSys and StatError from datasets."""
        rng = np.random.default_rng(42)
        sig_events = _make_events(rng, 500)
        bkg_events = _make_bkg_events(rng, 3000)
        bkg_lo = _make_bkg_events(rng, 2700)
        bkg_hi = _make_bkg_events(rng, 3300)
        data = np.concatenate([
            _make_events(rng, 500),
            _make_bkg_events(rng, 3000),
        ])

        sig = Sample.from_dataset("signal", Dataset(sig_events), edges=EDGES)
        sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 10.0))
        bkg = Sample.from_dataset("background", Dataset(bkg_events), edges=EDGES)
        bkg.add_histosys_from_datasets("bkg_syst", Dataset(bkg_lo), Dataset(bkg_hi))
        bkg.add_staterror_from_dataset("mc_stat", Dataset(bkg_events))

        ch = Channel("SR")
        ch.add_sample(sig)
        ch.add_sample(bkg)
        ch.set_data_from_dataset(Dataset(data))
        m = Model("full_workflow")
        m.add_channel(ch)

        r = fit(m)
        assert r.valid
        assert abs(r.bestfit["mu"] - 1.0) < 3 * r.errors["mu"]

    def test_multi_channel_from_datasets(self):
        """Multi-channel model entirely from datasets."""
        rng = np.random.default_rng(42)

        # Signal region
        sig_sr = _make_events(rng, 500)
        bkg_sr = _make_bkg_events(rng, 3000)
        data_sr = np.concatenate([_make_events(rng, 500), _make_bkg_events(rng, 3000)])

        sig_s = Sample.from_dataset("signal", Dataset(sig_sr), edges=EDGES)
        sig_s.add_normfactor("mu", nominal=1.0, bounds=(0.0, 10.0))
        bkg_s = Sample.from_dataset("background", Dataset(bkg_sr), edges=EDGES)
        sr = Channel("SR")
        sr.add_sample(sig_s)
        sr.add_sample(bkg_s)
        sr.set_data_from_dataset(Dataset(data_sr))

        # Control region (background-enriched)
        bkg_cr = _make_bkg_events(rng, 5000)
        data_cr = _make_bkg_events(rng, 5000)
        bkg_c = Sample.from_dataset("background", Dataset(bkg_cr), edges=EDGES)
        bkg_c.add_normsys("bkg_norm", 0.9, 1.1)
        cr = Channel("CR")
        cr.add_sample(bkg_c)
        cr.set_data_from_dataset(Dataset(data_cr))

        m = Model("multi_ch")
        m.add_channel(sr)
        m.add_channel(cr)

        r = fit(m)
        assert r.valid and m.n_channels == 2
        assert "mu" in r.bestfit and "bkg_norm" in r.bestfit
