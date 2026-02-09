"""Tests for Sprint 4: Sample, Channel, Model."""

import numpy as np
import pytest

from histimator.channels import Channel
from histimator.histograms import Histogram
from histimator.interpolation import InterpolationCode
from histimator.model import Model
from histimator.samples import Sample

# ---- fixtures -------------------------------------------------------------

EDGES = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

def make_signal():
    return Sample("signal", Histogram([10, 20, 30, 20], EDGES))

def make_background():
    return Sample("background", Histogram([50, 50, 50, 50], EDGES))


# ===========================================================================
# Sample tests
# ===========================================================================

class TestSample:
    def test_basic_expected(self):
        s = make_signal()
        result = s.expected({})
        np.testing.assert_array_equal(result, [10, 20, 30, 20])

    def test_normfactor(self):
        s = make_signal()
        s.add_normfactor("mu", nominal=1.0, bounds=(0.0, 5.0))
        # mu=2 should double all bins
        result = s.expected({"mu": 2.0})
        np.testing.assert_array_equal(result, [20, 40, 60, 40])

    def test_normfactor_default(self):
        """At nominal value (1.0), expected should equal the template."""
        s = make_signal()
        s.add_normfactor("mu")
        result = s.expected({"mu": 1.0})
        np.testing.assert_array_equal(result, [10, 20, 30, 20])

    def test_normsys(self):
        s = make_background()
        s.add_normsys("bkg_norm", lo=0.9, hi=1.1,
                      interp_code=InterpolationCode.PIECEWISE_LINEAR)
        # At alpha=0, should be nominal
        r0 = s.expected({"bkg_norm": 0.0})
        np.testing.assert_allclose(r0, [50, 50, 50, 50])
        # At alpha=+1, should be 1.1x
        r1 = s.expected({"bkg_norm": 1.0})
        np.testing.assert_allclose(r1, [55, 55, 55, 55])
        # At alpha=-1, should be 0.9x
        rm1 = s.expected({"bkg_norm": -1.0})
        np.testing.assert_allclose(rm1, [45, 45, 45, 45])

    def test_histosys(self):
        nom = Histogram([100, 100, 100, 100], EDGES)
        lo = Histogram([90, 95, 100, 105], EDGES)
        hi = Histogram([110, 105, 100, 95], EDGES)
        s = Sample("proc", nom)
        s.add_histosys("shape", lo, hi, interp_code=InterpolationCode.PIECEWISE_LINEAR)
        # alpha=0 -> nominal
        r0 = s.expected({"shape": 0.0})
        np.testing.assert_allclose(r0, [100, 100, 100, 100])
        # alpha=+1 -> hi
        r1 = s.expected({"shape": 1.0})
        np.testing.assert_allclose(r1, [110, 105, 100, 95])
        # alpha=-1 -> lo
        rm1 = s.expected({"shape": -1.0})
        np.testing.assert_allclose(rm1, [90, 95, 100, 105])

    def test_histosys_bin_mismatch(self):
        s = make_signal()  # 4 bins
        bad = Histogram([1, 2, 3], [0, 1, 2, 3])  # 3 bins
        with pytest.raises(ValueError, match="bins"):
            s.add_histosys("bad", bad, bad)

    def test_chained_modifiers(self):
        """NormFactor then NormSys applied in sequence."""
        s = Sample("sig", Histogram([100, 100], [0, 1, 2]))
        s.add_normfactor("mu", nominal=1.0, bounds=(0, 5))
        s.add_normsys("syst", lo=0.9, hi=1.1,
                      interp_code=InterpolationCode.PIECEWISE_LINEAR)
        # mu=2, syst=+1 -> 100*2*1.1 = 220
        result = s.expected({"mu": 2.0, "syst": 1.0})
        np.testing.assert_allclose(result, [220, 220])

    def test_parameters_property(self):
        s = make_signal()
        s.add_normfactor("mu")
        s.add_normsys("syst", 0.9, 1.1)
        names = [p.name for p in s.parameters]
        assert names == ["mu", "syst"]

    def test_repr(self):
        s = make_signal()
        s.add_normfactor("mu")
        assert "signal" in repr(s)
        assert "modifiers=1" in repr(s)


# ===========================================================================
# Channel tests
# ===========================================================================

class TestChannel:
    def test_add_samples_and_expected(self):
        ch = Channel("SR")
        ch.add_sample(make_signal())
        ch.add_sample(make_background())
        result = ch.expected({})
        np.testing.assert_array_equal(result, [60, 70, 80, 70])

    def test_bin_mismatch_rejected(self):
        ch = Channel("SR")
        ch.add_sample(make_signal())
        bad = Sample("bad", Histogram([1, 2, 3], [0, 1, 2, 3]))
        with pytest.raises(ValueError, match="bins"):
            ch.add_sample(bad)

    def test_edge_mismatch_rejected(self):
        ch = Channel("SR")
        ch.add_sample(make_signal())
        bad = Sample("bad", Histogram([1, 2, 3, 4], [0, 2, 4, 6, 8]))
        with pytest.raises(ValueError, match="edges"):
            ch.add_sample(bad)

    def test_set_data_array(self):
        ch = Channel("SR")
        ch.add_sample(make_signal())
        ch.set_data([15, 25, 35, 25])
        np.testing.assert_array_equal(ch.data.values, [15, 25, 35, 25])

    def test_set_data_histogram(self):
        ch = Channel("SR")
        ch.add_sample(make_signal())
        d = Histogram([15, 25, 35, 25], EDGES)
        ch.set_data(d)
        assert ch.data == d

    def test_set_data_wrong_bins(self):
        ch = Channel("SR")
        ch.add_sample(make_signal())
        with pytest.raises(ValueError, match="bins"):
            ch.set_data([1, 2, 3])  # 3 instead of 4

    def test_parameters_merged(self):
        ch = Channel("SR")
        sig = make_signal()
        sig.add_normfactor("mu")
        bkg = make_background()
        bkg.add_normsys("bkg_syst", 0.9, 1.1)
        ch.add_sample(sig)
        ch.add_sample(bkg)
        names = [p.name for p in ch.parameters]
        assert "mu" in names
        assert "bkg_syst" in names

    def test_chaining(self):
        ch = Channel("SR").add_sample(make_signal()).set_data([10, 20, 30, 20])
        assert ch.name == "SR"


# ===========================================================================
# Model tests
# ===========================================================================

class TestModel:
    def _build_simple_model(self):
        """Single-channel, signal + background model."""
        sig = make_signal()
        sig.add_normfactor("mu", nominal=1.0, bounds=(0, 5))
        bkg = make_background()

        ch = Channel("SR")
        ch.add_sample(sig)
        ch.add_sample(bkg)
        ch.set_data([65, 75, 85, 75])

        m = Model("test")
        m.add_channel(ch)
        return m

    def test_single_channel_expected(self):
        m = self._build_simple_model()
        result = m.expected({"mu": 1.0})
        np.testing.assert_array_equal(result, [60, 70, 80, 70])

    def test_single_channel_data(self):
        m = self._build_simple_model()
        np.testing.assert_array_equal(m.data, [65, 75, 85, 75])

    def test_parameters(self):
        m = self._build_simple_model()
        assert m.parameter_names == ["mu"]

    def test_nominal_values(self):
        m = self._build_simple_model()
        assert m.nominal_values() == {"mu": 1.0}

    def test_to_minuit_init(self):
        m = self._build_simple_model()
        init = m.to_minuit_init()
        assert init["errordef"] == 1
        assert init["mu"] == 1.0
        assert init["limit_mu"] == (0, 5)

    def test_total_nbins(self):
        m = self._build_simple_model()
        assert m.total_nbins == 4

    def test_channel_slices(self):
        m = self._build_simple_model()
        slices = m.channel_slices
        assert slices == [slice(0, 4)]

    def test_no_data_raises(self):
        ch = Channel("SR")
        ch.add_sample(make_signal())
        # No data set
        m = Model("test")
        with pytest.raises(ValueError, match="no observed data"):
            m.add_channel(ch)

    def test_multi_channel(self):
        sig = make_signal()
        sig.add_normfactor("mu", nominal=1.0, bounds=(0, 5))
        bkg = make_background()

        sr = Channel("SR")
        sr.add_sample(sig)
        sr.add_sample(bkg)
        sr.set_data([60, 70, 80, 70])

        # Control region: same background, no signal
        bkg2 = make_background()
        bkg2.add_normsys("bkg_syst", 0.9, 1.1)
        cr = Channel("CR")
        cr.add_sample(bkg2)
        cr.set_data([50, 50, 50, 50])

        m = Model("multi")
        m.add_channel(sr)
        m.add_channel(cr)

        assert m.n_channels == 2
        assert m.total_nbins == 8
        assert set(m.parameter_names) == {"mu", "bkg_syst"}

        result = m.expected({"mu": 1.0, "bkg_syst": 0.0})
        # SR: 10+50, 20+50, 30+50, 20+50 = 60,70,80,70
        # CR: 50,50,50,50 (at syst=0)
        np.testing.assert_allclose(result, [60, 70, 80, 70, 50, 50, 50, 50])

    def test_shared_parameter_across_channels(self):
        """A normfactor with the same name in two channels should be one parameter."""
        sig1 = Sample("sig", Histogram([10, 20], [0, 1, 2]))
        sig1.add_normfactor("mu", nominal=1.0, bounds=(0, 5))
        sig2 = Sample("sig", Histogram([5, 15], [0, 1, 2]))
        sig2.add_normfactor("mu", nominal=1.0, bounds=(0, 5))

        ch1 = Channel("SR1")
        ch1.add_sample(sig1)
        ch1.set_data([10, 20])

        ch2 = Channel("SR2")
        ch2.add_sample(sig2)
        ch2.set_data([5, 15])

        m = Model("shared")
        m.add_channel(ch1)
        m.add_channel(ch2)

        # Only one "mu" parameter
        assert m.parameter_names == ["mu"]

        # mu=2 should scale both channels
        result = m.expected({"mu": 2.0})
        np.testing.assert_array_equal(result, [20, 40, 10, 30])
