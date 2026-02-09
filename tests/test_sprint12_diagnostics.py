"""Tests for Sprint 12: Fit diagnostics.

Validation strategy
-------------------
Fit diagnostics are the analyst-facing tooling that turns a raw fit
result into interpretable information.  We validate each diagnostic
by constructing models where the expected behaviour is known:

- Pre-fit yields should match model.expected() at nominal parameter
  values exactly.
- Post-fit yields should match model.expected() at best-fit values
  (which should be close to data for a well-specified model).
- Pulls should be large when the data forces a nuisance parameter
  away from its nominal value, and small otherwise.
- Impacts should identify the nuisance parameter most correlated
  with the POI (highest impact) and correctly rank them.
"""

import numpy as np

from histimator.channels import Channel
from histimator.diagnostics import (
    impacts,
    nuisance_parameter_pulls,
    prefit_postfit_yields,
)
from histimator.histograms import Histogram
from histimator.model import Model
from histimator.samples import Sample

EDGES = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])


# ===================================================================
# Model builders
# ===================================================================

def _model_with_pulled_np(seed=42):
    """A model where one nuisance parameter is forced to pull.

    The background template is 100/bin, but we add a NormSys with
    lo=0.8, hi=1.2.  The data is 120/bin, so the fit must pull
    bkg_norm positive to match the 20% excess.  The signal is weak
    enough that mu absorbs very little of the excess.

    Expected behaviour: bkg_norm is pulled significantly positive,
    while other NPs (if any) stay near nominal.
    """
    rng = np.random.default_rng(seed)
    sig_t = np.array([2.0, 5.0, 10.0, 5.0, 2.0])
    bkg_t = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    data = np.array([120.0, 120.0, 120.0, 120.0, 120.0])

    sig = Sample("signal", Histogram(sig_t, EDGES))
    sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 20.0))
    bkg = Sample("background", Histogram(bkg_t, EDGES))
    bkg.add_normsys("bkg_norm", lo=0.8, hi=1.2)

    ch = Channel("SR")
    ch.add_sample(sig)
    ch.add_sample(bkg)
    ch.set_data(data)

    m = Model("pulled")
    m.add_channel(ch)
    return m


def _model_with_two_nps(seed=42):
    """A model with two nuisance parameters, one strongly correlated
    with the POI and one weakly correlated.

    bkg_norm scales the background coherently (anti-correlated with mu
    because both affect the total yield).  bkg_shape is a HistoSys
    that tilts the background shape without changing the total, so it
    is weakly correlated with mu.

    Expected behaviour: bkg_norm has larger impact on mu than bkg_shape.
    """
    rng = np.random.default_rng(seed)
    sig_t = np.array([5.0, 20.0, 40.0, 20.0, 5.0])
    bkg_t = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    data = rng.poisson(2.0 * sig_t + bkg_t).astype(float)

    sig = Sample("signal", Histogram(sig_t, EDGES))
    sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 20.0))
    bkg = Sample("background", Histogram(bkg_t, EDGES))
    bkg.add_normsys("bkg_norm", lo=0.9, hi=1.1)
    # Shape variation: tilt left/right (total stays ~constant)
    lo_shape = Histogram([110, 105, 100, 95, 90], EDGES)
    hi_shape = Histogram([90, 95, 100, 105, 110], EDGES)
    bkg.add_histosys("bkg_shape", lo_hist=lo_shape, hi_hist=hi_shape)

    ch = Channel("SR")
    ch.add_sample(sig)
    ch.add_sample(bkg)
    ch.set_data(data)

    m = Model("two_np")
    m.add_channel(ch)
    return m


def _simple_model():
    """A model with no nuisance parameters, just a normfactor."""
    sig_t = np.array([5.0, 20.0, 40.0, 20.0, 5.0])
    bkg_t = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    data = 2.0 * sig_t + bkg_t

    sig = Sample("signal", Histogram(sig_t, EDGES))
    sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 20.0))
    bkg = Sample("background", Histogram(bkg_t, EDGES))

    ch = Channel("SR")
    ch.add_sample(sig)
    ch.add_sample(bkg)
    ch.set_data(data)

    m = Model("simple")
    m.add_channel(ch)
    return m


# ===================================================================
# Tests: prefit_postfit_yields
# ===================================================================


class TestPrefitPostfitYields:
    """Pre-fit yields are evaluated at nominal parameter values.
    Post-fit yields are evaluated at the best-fit values.  Both
    are returned per-channel with per-sample and total breakdowns.
    """

    def test_prefit_matches_nominal(self):
        """Pre-fit yields should exactly equal model.expected() at
        nominal values.
        """
        model = _simple_model()
        result = prefit_postfit_yields(model)

        ch_result = result["SR"]
        nominal_params = model.nominal_values()
        expected_total = model.channels[0].expected(nominal_params)
        np.testing.assert_allclose(ch_result["prefit"]["total"], expected_total)

    def test_postfit_near_data(self):
        """For a well-specified model, post-fit total should approximate
        the observed data.
        """
        model = _simple_model()
        result = prefit_postfit_yields(model)

        ch_result = result["SR"]
        data = model.channels[0].data.values
        np.testing.assert_allclose(
            ch_result["postfit"]["total"], data, rtol=0.05
        )

    def test_contains_per_sample_breakdown(self):
        """Result should include yields for each sample individually."""
        model = _simple_model()
        result = prefit_postfit_yields(model)

        ch_result = result["SR"]
        assert "signal" in ch_result["prefit"]["samples"]
        assert "background" in ch_result["prefit"]["samples"]

    def test_prefit_samples_sum_to_total(self):
        """The per-sample yields should sum to the total."""
        model = _simple_model()
        result = prefit_postfit_yields(model)

        ch = result["SR"]
        sample_sum = sum(ch["prefit"]["samples"].values())
        np.testing.assert_allclose(sample_sum, ch["prefit"]["total"])

    def test_postfit_samples_sum_to_total(self):
        model = _model_with_pulled_np()
        result = prefit_postfit_yields(model)

        ch = result["SR"]
        sample_sum = sum(ch["postfit"]["samples"].values())
        np.testing.assert_allclose(sample_sum, ch["postfit"]["total"])


# ===================================================================
# Tests: nuisance_parameter_pulls
# ===================================================================


class TestNuisanceParameterPulls:
    """A pull is (theta_hat - theta_0) / sigma_prefit, measuring how
    far the post-fit value is from the nominal in units of the pre-fit
    uncertainty.  A large pull indicates the data strongly constrains
    or shifts that parameter.

    The constraint ratio (sigma_postfit / sigma_prefit) measures how
    much the data constrains the parameter beyond its prior.  A ratio
    near 1 means the prior dominates; much less than 1 means the data
    is highly informative.
    """

    def test_pulled_np_has_large_pull(self):
        """bkg_norm should have a significant pull when data is 20%
        above the nominal background.
        """
        model = _model_with_pulled_np()
        pulls = nuisance_parameter_pulls(model)

        bkg_pull = pulls["bkg_norm"]
        assert abs(bkg_pull["pull"]) > 0.3, (
            "bkg_norm should be pulled significantly"
        )

    def test_pull_direction_correct(self):
        """With data above nominal, bkg_norm should be pulled positive
        (upward normalisation shift).
        """
        model = _model_with_pulled_np()
        pulls = nuisance_parameter_pulls(model)
        assert pulls["bkg_norm"]["pull"] > 0

    def test_constraint_ratio_less_than_one(self):
        """Post-fit uncertainty should be smaller than or equal to
        pre-fit uncertainty (the data adds information).
        """
        model = _model_with_pulled_np()
        pulls = nuisance_parameter_pulls(model)
        assert pulls["bkg_norm"]["constraint"] <= 1.0

    def test_returns_all_nps(self):
        """Should return an entry for every constrained NP."""
        model = _model_with_two_nps()
        pulls = nuisance_parameter_pulls(model)
        assert "bkg_norm" in pulls
        assert "bkg_shape" in pulls

    def test_does_not_include_poi(self):
        """The POI (mu) should not appear in the pull dictionary."""
        model = _model_with_two_nps()
        pulls = nuisance_parameter_pulls(model)
        assert "mu" not in pulls


# ===================================================================
# Tests: impacts
# ===================================================================


class TestImpacts:
    """Impact is the shift in the POI caused by fixing a nuisance
    parameter at its +1sigma or -1sigma post-fit value and refitting.
    Parameters are ranked by the magnitude of their impact.

    Impact is measured as (mu_shifted - mu_nominal), so positive impact
    means the NP shifts the POI upward.
    """

    def test_norm_impact_larger_than_shape(self):
        """bkg_norm should have larger impact on mu than bkg_shape,
        because the overall normalisation is more degenerate with
        signal strength than a shape tilt.
        """
        model = _model_with_two_nps()
        impact_list = impacts(model, "mu")

        # Find impacts for each NP
        impact_dict = {entry["name"]: entry for entry in impact_list}
        norm_impact = max(
            abs(impact_dict["bkg_norm"]["impact_up"]),
            abs(impact_dict["bkg_norm"]["impact_down"]),
        )
        shape_impact = max(
            abs(impact_dict["bkg_shape"]["impact_up"]),
            abs(impact_dict["bkg_shape"]["impact_down"]),
        )

        assert norm_impact > shape_impact

    def test_ranked_by_impact(self):
        """The impact list should be sorted by decreasing total impact."""
        model = _model_with_two_nps()
        impact_list = impacts(model, "mu")

        total_impacts = [
            max(abs(e["impact_up"]), abs(e["impact_down"]))
            for e in impact_list
        ]
        # Should be monotonically non-increasing
        for i in range(len(total_impacts) - 1):
            assert total_impacts[i] >= total_impacts[i + 1]

    def test_returns_all_nps(self):
        """Should include all constrained nuisance parameters."""
        model = _model_with_two_nps()
        impact_list = impacts(model, "mu")
        names = {e["name"] for e in impact_list}
        assert "bkg_norm" in names
        assert "bkg_shape" in names

    def test_does_not_include_poi(self):
        """The POI should not appear as an impact entry."""
        model = _model_with_two_nps()
        impact_list = impacts(model, "mu")
        names = {e["name"] for e in impact_list}
        assert "mu" not in names

    def test_impact_structure(self):
        """Each impact entry should have the expected fields."""
        model = _model_with_two_nps()
        impact_list = impacts(model, "mu")

        for entry in impact_list:
            assert "name" in entry
            assert "impact_up" in entry
            assert "impact_down" in entry
            assert "bestfit" in entry
            assert "error" in entry
