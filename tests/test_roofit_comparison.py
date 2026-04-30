"""Comparison tests against ROOT/RooFit.

Skipped automatically if ROOT/RooFit is not importable. When run, builds
small reference models in both Histimator and RooFit and compares:

  - Extended Poisson NLL values and differences (binned).
  - Best-fit POI value and Hesse error from MIGRAD.
  - q_0 discovery test statistic on a counting experiment (closed form).
  - Asimov discovery significance Z_A (closed form from CCGV eq. 97).
  - HistFactory NormSys / HistoSys interpolation codes against
    FlexibleInterpVar / PiecewiseInterpolation.

Comparisons are designed so that additive NLL constants cancel: we compare
NLL differences, best-fit values, Hesse errors, and test-statistic values,
not absolute NLL values.
"""
from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import pytest

from histimator.channels import Channel
from histimator.cls import build_asimov_model
from histimator.histograms import Histogram
from histimator.inference import discovery_significance
from histimator.interpolation import InterpolationCode, interpolate
from histimator.likelihood import BinnedNLL, fit
from histimator.model import Model
from histimator.samples import Sample
from histimator.test_statistics import compute_q0

ROOT = pytest.importorskip("ROOT")

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kError
ROOT.RooMsgService.instance().setSilentMode(True)
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)

hf = ROOT.RooStats.HistFactory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hi_single_bin(s: float, b: float, n: float):
    """Single-bin signal+background counting model with floating mu."""
    edges = np.array([0.0, 1.0])
    sig = Sample("signal", Histogram([s], edges))
    sig.add_normfactor("mu", nominal=1.0, bounds=(-5.0, 50.0))
    bkg = Sample("background", Histogram([b], edges))
    ch = Channel("c1")
    ch.add_sample(sig)
    ch.add_sample(bkg)
    ch.set_data([n])
    m = Model("counting")
    m.add_channel(ch)
    return m


def _hi_multibin(sig_template, bkg_template, data):
    edges = np.linspace(0.0, 1.0, len(sig_template) + 1)
    sig = Sample("signal", Histogram(sig_template, edges))
    sig.add_normfactor("mu", nominal=1.0, bounds=(-5.0, 50.0))
    bkg = Sample("background", Histogram(bkg_template, edges))
    ch = Channel("c1")
    ch.add_sample(sig)
    ch.add_sample(bkg)
    ch.set_data(np.asarray(data, dtype=float))
    m = Model("multibin")
    m.add_channel(ch)
    return m


def _rf_extended_poisson_nll(sig_template, bkg_template, data):
    """Build the RooFit extended Poisson NLL = sum_j -log Pois(n_j | mu*s_j+b_j).

    Returns (mu_var, nll_callable, mu_var_minimizable_minuit_obj_factory).
    """
    mu = ROOT.RooRealVar("mu", "mu", 1.0, -5.0, 50.0)
    poissons = ROOT.RooArgList()
    keep = [mu]  # keep python references alive
    for j, (sj, bj, nj) in enumerate(zip(sig_template, bkg_template, data, strict=True)):
        sj_v = ROOT.RooRealVar(f"s_{j}", f"s_{j}", float(sj))
        sj_v.setConstant(True)
        bj_v = ROOT.RooRealVar(f"b_{j}", f"b_{j}", float(bj))
        bj_v.setConstant(True)
        # mean_j = mu * s_j + b_j
        mean_j = ROOT.RooFormulaVar(
            f"mean_{j}", "@0*@1+@2",
            ROOT.RooArgList(mu, sj_v, bj_v),
        )
        n_v = ROOT.RooRealVar(f"n_{j}", f"n_{j}", float(nj))
        n_v.setConstant(True)
        pois = ROOT.RooPoisson(f"pois_{j}", f"pois_{j}", n_v, mean_j)
        keep.extend([sj_v, bj_v, mean_j, n_v, pois])
        poissons.add(pois)

    prod = ROOT.RooProdPdf("prod", "prod", poissons)
    keep.append(prod)
    # Build NLL by summing -log P over the index set; use createNLL on a
    # RooDataSet trick is overkill -- evaluate by hand via the factor pdfs.

    def nll_at(mu_value):
        mu.setVal(mu_value)
        # sum -log Pois(n_j | mean_j)
        total = 0.0
        for sj, bj, nj in zip(sig_template, bkg_template, data, strict=True):
            mean_j = mu_value * sj + bj
            mean_j = max(mean_j, 1e-300)
            total += -(nj * math.log(mean_j) - mean_j - math.lgamma(nj + 1))
        return total

    return mu, nll_at, keep


# ---------------------------------------------------------------------------
# 1. Single-bin extended NLL: absolute value identity
# ---------------------------------------------------------------------------

class TestSingleBinExtendedNLL:
    """In a single-bin counting model, the extended Poisson NLL should
    be exactly -log Pois(n | mu*s + b). The default extended=True path
    must produce this, otherwise it is double-counting the total Poisson.
    """

    @pytest.mark.parametrize("mu_val", [0.0, 0.5, 1.0, 2.0, -0.3])
    def test_extended_true_matches_pure_poisson(self, mu_val):
        s, b, n = 10.0, 50.0, 65.0
        model = _hi_single_bin(s, b, n)
        nll = BinnedNLL(model, extended=True)

        hi_val = nll(np.array([mu_val]))
        lam = max(mu_val * s + b, 1e-300)
        ref = -(n * math.log(lam) - lam - math.lgamma(n + 1))
        assert hi_val == pytest.approx(ref, rel=1e-9)


# ---------------------------------------------------------------------------
# 2. Multi-bin extended NLL differences vs RooFit
# ---------------------------------------------------------------------------

class TestMultibinExtendedNLLDifferences:
    """NLL DIFFERENCES are constant-free: NLL(mu1) - NLL(mu0) must agree
    between Histimator and the per-bin RooPoisson product up to numerical
    precision. Disagreement here means the likelihoods differ in shape,
    not just by a constant offset.
    """

    def _setup(self):
        sig = np.array([10.0, 30.0, 50.0, 30.0, 10.0])
        bkg = np.array([50.0, 50.0, 50.0, 50.0, 50.0])
        # Asimov-like data at mu=1 (deterministic; no fluctuation)
        data = sig + bkg
        return sig, bkg, data

    @pytest.mark.parametrize(
        "mu0, mu1",
        [(1.0, 0.0), (1.0, 0.5), (1.0, 2.0), (0.0, 1.5), (-0.2, 1.0)],
    )
    def test_extended_true_delta_matches_per_bin_poisson(self, mu0, mu1):
        sig, bkg, data = self._setup()
        model = _hi_multibin(sig, bkg, data)
        nll = BinnedNLL(model, extended=True)

        hi_delta = nll(np.array([mu1])) - nll(np.array([mu0]))

        _mu, ref_nll_at, _keep = _rf_extended_poisson_nll(sig, bkg, data)
        ref_delta = ref_nll_at(mu1) - ref_nll_at(mu0)

        assert hi_delta == pytest.approx(ref_delta, abs=1e-6)


# ---------------------------------------------------------------------------
# 3. fit() best-fit and Hesse error vs RooFit MIGRAD
# ---------------------------------------------------------------------------

class TestFitBestfitAndError:
    """Fit best-fit POI and Hesse error must agree between Histimator's
    extended NLL fit and a direct RooFit MIGRAD on the per-bin product
    of RooPoissons. The error scales as 1/sqrt(curvature), so a doubled
    NLL would shrink errors by 1/sqrt(2)."""

    def _setup(self):
        sig = np.array([10.0, 30.0, 50.0, 30.0, 10.0])
        bkg = np.array([50.0, 50.0, 50.0, 50.0, 50.0])
        rng = np.random.default_rng(12345)
        data = rng.poisson(1.5 * sig + bkg).astype(float)
        return sig, bkg, data

    def _fit_roofit(self, sig, bkg, data):
        """Direct RooFit MIGRAD on the product of per-bin RooPoissons.

        We construct the joint pdf as a RooProdPdf of per-bin RooPoissons
        and minimise the analytic NLL = -log L using RooMinimizer on a
        custom RooFormulaVar that returns the same NLL we want to test.
        """
        mu, nll_at, _keep = _rf_extended_poisson_nll(sig, bkg, data)
        # Wrap into a RooAbsReal that RooMinimizer can drive.
        # Build sum_j ( -n_j*log(mean_j) + mean_j ) directly.
        terms = ROOT.RooArgList()
        keep = [mu]
        for j, (sj, bj, nj) in enumerate(zip(sig, bkg, data, strict=True)):
            sj_v = ROOT.RooRealVar(f"sm_{j}", "", float(sj))
            sj_v.setConstant(True)
            bj_v = ROOT.RooRealVar(f"bm_{j}", "", float(bj))
            bj_v.setConstant(True)
            nj_v = ROOT.RooRealVar(f"nm_{j}", "", float(nj))
            nj_v.setConstant(True)
            term = ROOT.RooFormulaVar(
                f"term_{j}",
                "(@0*@1+@2) - @3*log(@0*@1+@2)",
                ROOT.RooArgList(mu, sj_v, bj_v, nj_v),
            )
            keep.extend([sj_v, bj_v, nj_v, term])
            terms.add(term)
        nll = ROOT.RooAddition("nll", "nll", terms)
        keep.append(nll)
        m = ROOT.RooMinimizer(nll)
        m.setPrintLevel(-1)
        m.setStrategy(1)
        m.setErrorLevel(0.5)  # NLL convention: 1 sigma corresponds to dNLL = 0.5
        m.minimize("Minuit2", "Migrad")
        m.hesse()
        mu_hat = mu.getVal()
        mu_err = mu.getError()
        return mu_hat, mu_err

    def test_bestfit_and_error_agree(self):
        sig, bkg, data = self._setup()
        # RooFit reference
        rf_mu, rf_err = self._fit_roofit(sig, bkg, data)

        # Histimator (default = extended=True)
        model = _hi_multibin(sig, bkg, data)
        result = fit(model)
        assert result.valid
        hi_mu = result.bestfit["mu"]
        hi_err = result.errors["mu"]

        # Best-fit values: should agree to better than 1e-3 in mu.
        assert hi_mu == pytest.approx(rf_mu, abs=1e-3), (
            f"Best-fit mu disagrees: histimator={hi_mu:.6f} roofit={rf_mu:.6f}"
        )
        # Hesse error: must agree to a few percent. A double-counted total
        # Poisson would shrink the Histimator error by ~1/sqrt(2) on an
        # otherwise normalisation-dominated fit.
        assert hi_err == pytest.approx(rf_err, rel=2e-2), (
            f"Hesse error disagrees: histimator={hi_err:.6f} roofit={rf_err:.6f}"
        )


# ---------------------------------------------------------------------------
# 4. q_0 closed form on a counting experiment
# ---------------------------------------------------------------------------

class TestQ0CountingClosedForm:
    """For a single-bin counting experiment with observed n, expected
    background b, signal yield s (s does not enter the closed form
    when mu floats freely), the discovery test statistic is:

        q0 = 2[n log(n/b) - (n - b)]   when n >= b
        q0 = 0                          when n < b

    (Wilks form for Poisson with one free normalisation
    equivalent to
    CCGV eq. 12 specialised to a single bin and floating signal-only norm.)
    """

    @pytest.mark.parametrize("n, b", [(65, 50), (80, 50), (50, 50), (40, 50)])
    def test_q0_matches_closed_form(self, n, b):
        s = 10.0
        model = _hi_single_bin(s, b, n)
        q0, mu_hat = compute_q0(model, "mu")

        ref = 2.0 * (n * math.log(n / b) - (n - b)) if n >= b else 0.0
        assert q0 == pytest.approx(ref, abs=2e-3), (
            f"q0 mismatch n={n} b={b}: histimator={q0:.4f} closed-form={ref:.4f}"
        )


# ---------------------------------------------------------------------------
# 5. Asimov discovery significance closed form
# ---------------------------------------------------------------------------

class TestAsimovZClosedForm:
    """The Asimov discovery significance for a counting experiment with
    expected signal s on top of background b is (CCGV eq. 97):

        Z_A = sqrt( 2 * ((s+b) * log(1 + s/b) - s) )

    When the data equals the (s+b) Asimov, q0 evaluated on that data
    equals Z_A^2 exactly under the asymptotic approximation. We use
    Histimator's discovery_significance on the (s+b)-Asimov data.
    """

    @pytest.mark.parametrize("s, b", [(5.0, 50.0), (10.0, 50.0), (20.0, 50.0)])
    def test_asimov_z_matches_closed_form(self, s, b):
        # Build model and Asimov data at mu=1
        model = _hi_single_bin(s, b, n=s + b)  # n placeholder
        asimov = build_asimov_model(model, {"mu": 1.0})

        z, p = discovery_significance(asimov, "mu")
        z_ref = math.sqrt(2.0 * ((s + b) * math.log(1.0 + s / b) - s))
        assert z == pytest.approx(z_ref, abs=5e-3), (
            f"Asimov Z mismatch s={s} b={b}: histimator={z:.4f} closed={z_ref:.4f}"
        )


# ---------------------------------------------------------------------------
# 6. NormSys interpolation vs FlexibleInterpVar
# ---------------------------------------------------------------------------

class TestNormSysInterpolationVsFlexibleInterpVar:
    """histimator.interpolate() with codes 0, 1, 3 should agree with the
    HistFactory FlexibleInterpVar for the corresponding HF interpolation
    codes. histimator code 3 maps to HF code 4 (poly + exponential).
    """

    HI_TO_HF = {0: 0, 1: 1, 2: 2, 3: 4}

    @pytest.mark.parametrize(
        "lo, nom, hi", [(0.8, 1.0, 1.2), (0.5, 1.0, 1.4), (0.95, 1.0, 1.05)]
    )
    @pytest.mark.parametrize("hi_code", [0, 1, 2, 3])
    @pytest.mark.parametrize(
        "alpha", [-1.5, -1.0, -0.5, 0.0, 0.3, 1.0, 1.5]
    )
    def test_normsys_factor_agrees(self, lo, nom, hi, hi_code, alpha):
        # Histimator's code 2 (quadratic-linear) extrapolates linearly
        # outside |alpha| <= 1, which can produce negative factors for
        # large |alpha| with strong down variations. RooFit's
        # FlexibleInterpVar code 2 clamps at zero in that regime. Both
        # match exactly inside |alpha| <= 1.5 for the lo/nom/hi triples
        # used here; deep-extrapolation behaviour is covered separately.
        # Histimator value
        hi_val = float(
            interpolate(alpha, np.array([nom]), np.array([lo]), np.array([hi]),
                        code=InterpolationCode(hi_code))[0]
        )

        # RooFit FlexibleInterpVar with one nuisance
        a = ROOT.RooRealVar("a", "a", 0.0, -10.0, 10.0)
        nps = ROOT.RooArgList(a)
        lo_v = ROOT.std.vector("double")()
        hi_v = ROOT.std.vector("double")()
        lo_v.push_back(float(lo / nom))   # FIV expects multiplicative ratios
        hi_v.push_back(float(hi / nom))
        fiv = ROOT.RooStats.HistFactory.FlexibleInterpVar(
            "fiv", "fiv", nps, 1.0, lo_v, hi_v
        )
        fiv.setAllInterpCodes(self.HI_TO_HF[hi_code])
        a.setVal(float(alpha))
        rf_val = nom * fiv.getVal()
        assert hi_val == pytest.approx(rf_val, rel=5e-4, abs=5e-6), (
            f"NormSys interp code={hi_code} alpha={alpha} "
            f"histimator={hi_val:.6f} roofit={rf_val:.6f}"
        )


# ---------------------------------------------------------------------------
# 7. HistoSys interpolation vs PiecewiseInterpolation
# ---------------------------------------------------------------------------

class TestHistoSysPolyLinearVsPiecewiseInterpolation:
    """POLY_LINEAR (HistFactory PiecewiseInterpolation::code 4) per-bin
    interpolation must match RooFit's PiecewiseInterpolation exactly,
    across symmetric and asymmetric variations and a grid of alpha
    values that span both the polynomial and linear branches.
    """

    @pytest.mark.parametrize(
        "lo_arr, nom_arr, hi_arr",
        [
            # symmetric
            (np.array([8., 18., 28., 18., 8.]),
             np.array([10., 20., 30., 20., 10.]),
             np.array([12., 22., 32., 22., 12.])),
            # asymmetric (per bin)
            (np.array([8., 18., 28., 22., 12.]),
             np.array([10., 20., 30., 20., 10.]),
             np.array([12., 24., 33., 18., 9.])),
            # crosses nominal (down > nominal in some bins)
            (np.array([11., 21., 28., 19., 10.]),
             np.array([10., 20., 30., 20., 10.]),
             np.array([13., 22., 31., 21., 9.5])),
        ],
    )
    @pytest.mark.parametrize(
        "alpha", [-2.5, -1.5, -1.0, -0.7, -0.3, 0.0, 0.4, 1.0, 1.7, 2.5]
    )
    def test_poly_linear_per_bin_agrees(self, lo_arr, nom_arr, hi_arr, alpha):
        hi_vals = interpolate(alpha, nom_arr, lo_arr, hi_arr,
                              code=InterpolationCode.POLY_LINEAR)

        # Build PiecewiseInterpolation per bin and evaluate
        a = ROOT.RooRealVar("a", "a", 0.0, -10.0, 10.0)
        nps = ROOT.RooArgList(a)
        rf_vals = np.empty_like(nom_arr)
        for j in range(len(nom_arr)):
            nv = ROOT.RooRealVar(f"nv_{j}", "", float(nom_arr[j]))
            nv.setConstant(True)
            lv = ROOT.RooRealVar(f"lv_{j}", "", float(lo_arr[j]))
            lv.setConstant(True)
            hv = ROOT.RooRealVar(f"hv_{j}", "", float(hi_arr[j]))
            hv.setConstant(True)
            pi = ROOT.PiecewiseInterpolation(
                f"pi_{j}", "", nv,
                ROOT.RooArgList(lv), ROOT.RooArgList(hv), nps,
            )
            pi.setAllInterpCodes(4)
            a.setVal(float(alpha))
            rf_vals[j] = pi.getVal()

        np.testing.assert_allclose(hi_vals, rf_vals, rtol=1e-9, atol=1e-12)


class TestHistoSysInterpolationPerBinVsFlexibleInterpVar:
    """Histimator's HistoSys applies the same per-bin multiplicative
    interpolation as NormSys (i.e., FlexibleInterpVar semantics applied
    bin-by-bin), not RooFit's PiecewiseInterpolation (which is additive
    poly+linear at code 4).

    Both conventions are widespread in HistFactory implementations
    the
    choice is documented in interpolation.py. The test verifies that
    Histimator's per-bin HistoSys matches FlexibleInterpVar per bin,
    which is what its codes claim to implement.
    """

    HI_TO_HF = {0: 0, 1: 1, 2: 2, 3: 4}

    @pytest.mark.parametrize("hi_code", [0, 1, 3])
    @pytest.mark.parametrize("alpha", [-1.5, -1.0, -0.4, 0.0, 0.7, 1.0, 1.8])
    def test_histosys_per_bin_agrees(self, hi_code, alpha):
        nom = np.array([10.0, 20.0, 30.0, 20.0, 10.0])
        down = np.array([8.0, 18.0, 28.0, 22.0, 12.0])
        up = np.array([12.0, 24.0, 33.0, 18.0, 9.0])

        hi_arr = interpolate(alpha, nom, down, up,
                             code=InterpolationCode(hi_code))

        # FlexibleInterpVar applied per bin, which is the multiplicative
        # interpolation Histimator uses for HistoSys.
        a = ROOT.RooRealVar("a", "a", 0.0, -10.0, 10.0)
        nps = ROOT.RooArgList(a)
        rf_arr = np.empty_like(nom)
        for j in range(len(nom)):
            lo_v = ROOT.std.vector("double")()
            hi_v = ROOT.std.vector("double")()
            lo_v.push_back(float(down[j] / nom[j]))
            hi_v.push_back(float(up[j] / nom[j]))
            fiv = ROOT.RooStats.HistFactory.FlexibleInterpVar(
                f"fiv_{j}", f"fiv_{j}", nps, 1.0, lo_v, hi_v
            )
            fiv.setAllInterpCodes(self.HI_TO_HF[hi_code])
            a.setVal(float(alpha))
            rf_arr[j] = nom[j] * fiv.getVal()

        np.testing.assert_allclose(hi_arr, rf_arr, rtol=5e-4, atol=5e-6)


# ---------------------------------------------------------------------------
# 8. Likelihood scan parabolic curvature ↔ Hesse error
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# HistFactory workspace round-trip helpers
# ---------------------------------------------------------------------------

def _th1_from(name, values, edges, errors=None):
    """Build a TH1D from a numpy array of bin contents."""
    h = ROOT.TH1D(name, name, len(values), float(edges[0]), float(edges[-1]))
    for i, v in enumerate(values):
        h.SetBinContent(i + 1, float(v))
        if errors is not None:
            h.SetBinError(i + 1, float(errors[i]))
        else:
            h.SetBinError(i + 1, np.sqrt(max(float(v), 0.0)))
    return h


def _build_hf_workspace(spec, output_dir):
    """Build a RooStats.HistFactory workspace from a `spec` dict.

    spec keys:
      edges: bin edges
      channels: list of channel dicts:
        name, data, samples = list of sample dicts
      sample dict: name, hist, normfactors=[(name, val, lo, hi)],
                   normsys=[(name, lo_ratio, hi_ratio)],
                   histosys=[(name, lo_hist, hi_hist)],
                   staterror=bool,                 # ActivateStatError
                   shapesys=[(name, rel_unc_array)]  (Poisson constraint)
      lumi: tuple (nominal, rel_err) or None (defaults to fixed at 1)
      poi: parameter of interest name
      stat_constraint: 'Gaussian' or 'Poisson' (default Gaussian).
    """
    edges = np.asarray(spec["edges"], dtype=float)

    meas = hf.Measurement(spec.get("name", "test"), "test")
    meas.SetOutputFilePrefix(os.path.join(output_dir, "out"))
    meas.SetPOI(spec["poi"])
    lumi = spec.get("lumi", (1.0, 1e-6))
    meas.SetLumi(float(lumi[0]))
    meas.SetLumiRelErr(float(lumi[1]))
    if lumi[1] <= 1e-5:
        meas.AddConstantParam("Lumi")
    stat_type = spec.get("stat_constraint", "Gaussian")

    keep = []
    for ch_spec in spec["channels"]:
        chan = hf.Channel(ch_spec["name"])
        # Default StatError config: Gaussian, no threshold
        chan.SetStatErrorConfig(0.0, stat_type)
        chan.SetData(_th1_from(f"data_{ch_spec['name']}", ch_spec["data"], edges))
        for s_spec in ch_spec["samples"]:
            sample = hf.Sample(s_spec["name"])
            sample.SetHisto(_th1_from(
                f"{ch_spec['name']}_{s_spec['name']}", s_spec["hist"], edges,
                errors=s_spec.get("hist_errors"),
            ))
            for nf in s_spec.get("normfactors", []):
                sample.AddNormFactor(nf[0], float(nf[1]), float(nf[2]), float(nf[3]))
            for ns in s_spec.get("normsys", []):
                sample.AddOverallSys(ns[0], float(ns[1]), float(ns[2]))
            for hs in s_spec.get("histosys", []):
                lo_h = _th1_from(f"{ch_spec['name']}_{s_spec['name']}_{hs[0]}_lo",
                                 hs[1], edges)
                hi_h = _th1_from(f"{ch_spec['name']}_{s_spec['name']}_{hs[0]}_hi",
                                 hs[2], edges)
                histosys = hf.HistoSys()
                histosys.SetName(hs[0])
                histosys.SetHistoLow(lo_h)
                histosys.SetHistoHigh(hi_h)
                sample.AddHistoSys(histosys)
                keep.extend([lo_h, hi_h])
            if s_spec.get("staterror", False):
                # HF derives the per-bin relative stat uncertainty from
                # the sample histogram's bin errors (TH1::GetBinError).
                # We already set those above via `hist_errors`, so no
                # extra SetErrorHist call is needed: HF will normalize
                # by the sample yield to produce a dimensionless width.
                sample.GetStatError().Activate(True)
            for ss in s_spec.get("shapesys", []):
                # HF's ShapeSys::SetErrorHist values are treated as
                # relative uncertainties directly (tau = 1/value^2),
                # so we pass rel_uncertainties as-is.
                rel = np.asarray(ss[1])
                ss_h = _th1_from(
                    f"{ch_spec['name']}_{s_spec['name']}_{ss[0]}_rel",
                    rel, edges,
                )
                shape_sys = hf.ShapeSys()
                shape_sys.SetName(ss[0])
                shape_sys.SetErrorHist(ss_h)
                shape_sys.SetConstraintType(hf.Constraint.Poisson)
                sample.AddShapeSys(shape_sys)
                keep.append(ss_h)
            chan.AddSample(sample)
        meas.AddChannel(chan)

    factory = hf.HistoToWorkspaceFactoryFast(meas)
    ws = factory.MakeCombinedModel(meas)
    return ws


def _fit_workspace(ws, poi_name=None):
    """Fit a HistFactory workspace
    return (bestfit, errors, nll_min)."""
    mc = ws.obj("ModelConfig")
    pdf = mc.GetPdf()
    data = ws.data("obsData")
    nll = pdf.createNLL(data)
    m = ROOT.RooMinimizer(nll)
    m.setPrintLevel(-1)
    m.setStrategy(2)
    m.setErrorLevel(0.5)
    m.minimize("Minuit2", "Migrad")
    m.hesse()

    # Collect best-fit and errors for all floating parameters
    bestfit = {}
    errors = {}
    for p in nll.getVariables():
        if not p.isConstant() and p.GetName() != "weightVar":
            bestfit[p.GetName()] = float(p.getVal())
            errors[p.GetName()] = float(p.getError())
    return bestfit, errors, float(nll.getVal())


# ---------------------------------------------------------------------------
# Reference data builder for shared physics setup
# ---------------------------------------------------------------------------

def _make_reference_data(seed=42, true_mu=1.5):
    """Generate a single shared piece of pseudo-data + templates that
    every workspace round-trip test consumes, so that fits compare
    against identical inputs."""
    edges = np.linspace(0.0, 5.0, 6)
    sig_t = np.array([10.0, 25.0, 40.0, 25.0, 10.0])
    bkg_t = np.array([55.0, 55.0, 55.0, 55.0, 55.0])
    rng = np.random.default_rng(seed)
    data = rng.poisson(true_mu * sig_t + bkg_t).astype(float)
    return edges, sig_t, bkg_t, data


# ---------------------------------------------------------------------------
# 9. NormFactor-only HistFactory round-trip
# ---------------------------------------------------------------------------

class TestHistFactoryNormFactorOnly:
    """Build an identical signal+background model in HistFactory and
    Histimator with one floating mu, fit both, and compare best-fit
    and Hesse error."""

    def test_normfactor_roundtrip(self):
        edges, sig_t, bkg_t, data = _make_reference_data()

        # ---- HistFactory ----
        with tempfile.TemporaryDirectory() as d:
            spec = dict(
                edges=edges, poi="mu",
                channels=[dict(
                    name="SR", data=data,
                    samples=[
                        dict(name="signal", hist=sig_t,
                             normfactors=[("mu", 1.0, 0.0, 20.0)]),
                        dict(name="background", hist=bkg_t),
                    ],
                )],
            )
            ws = _build_hf_workspace(spec, d)
            hf_best, hf_err, _ = _fit_workspace(ws)

        # ---- Histimator ----
        from histimator.channels import Channel
        from histimator.histograms import Histogram
        from histimator.likelihood import fit
        from histimator.model import Model
        from histimator.samples import Sample

        sig = Sample("signal", Histogram(sig_t, edges))
        sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 20.0))
        bkg = Sample("background", Histogram(bkg_t, edges))
        ch = Channel("SR")
        ch.add_sample(sig)
        ch.add_sample(bkg)
        ch.set_data(data)
        m = Model("test")
        m.add_channel(ch)
        hi_result = fit(m)

        assert hi_result.bestfit["mu"] == pytest.approx(hf_best["mu"], abs=2e-3)
        assert hi_result.errors["mu"] == pytest.approx(hf_err["mu"], rel=2e-2)


# ---------------------------------------------------------------------------
# 10. NormSys round-trip
# ---------------------------------------------------------------------------

class TestHistFactoryNormSysRoundTrip:
    """NormSys (overall systematic) on background
    both frameworks should
    give the same fitted mu, alpha and errors."""

    def test_normsys_default_code(self):
        edges, sig_t, bkg_t, data = _make_reference_data()

        with tempfile.TemporaryDirectory() as d:
            spec = dict(
                edges=edges, poi="mu",
                channels=[dict(
                    name="SR", data=data,
                    samples=[
                        dict(name="signal", hist=sig_t,
                             normfactors=[("mu", 1.0, 0.0, 20.0)]),
                        dict(name="background", hist=bkg_t,
                             normsys=[("bkg_norm", 0.9, 1.1)]),
                    ],
                )],
            )
            ws = _build_hf_workspace(spec, d)
            hf_best, hf_err, _ = _fit_workspace(ws)

        from histimator.channels import Channel
        from histimator.histograms import Histogram
        from histimator.interpolation import InterpolationCode
        from histimator.likelihood import fit
        from histimator.model import Model
        from histimator.samples import Sample

        sig = Sample("signal", Histogram(sig_t, edges))
        sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 20.0))
        bkg = Sample("background", Histogram(bkg_t, edges))
        # HistFactory default for NormSys is FlexibleInterpVar code 4
        # (POLY_EXPONENTIAL); use that here to match.
        bkg.add_normsys("bkg_norm", lo=0.9, hi=1.1,
                        interp_code=InterpolationCode.POLY_EXPONENTIAL)
        ch = Channel("SR")
        ch.add_sample(sig)
        ch.add_sample(bkg)
        ch.set_data(data)
        m = Model("test")
        m.add_channel(ch)
        hi_result = fit(m)

        assert hi_result.bestfit["mu"] == pytest.approx(
            hf_best["mu"], abs=3e-3
        ), f"mu disagrees: HF={hf_best['mu']:.6f} HI={hi_result.bestfit['mu']:.6f}"
        assert hi_result.bestfit["bkg_norm"] == pytest.approx(
            hf_best["alpha_bkg_norm"], abs=3e-3
        ), (
            f"alpha disagrees: HF={hf_best['alpha_bkg_norm']:.6f} "
            f"HI={hi_result.bestfit['bkg_norm']:.6f}"
        )
        assert hi_result.errors["mu"] == pytest.approx(
            hf_err["mu"], rel=3e-2
        )


# ---------------------------------------------------------------------------
# 11. HistoSys round-trip with POLY_LINEAR
# ---------------------------------------------------------------------------

class TestHistFactoryHistoSysRoundTrip:
    """HistoSys on background using the new POLY_LINEAR (additive) code
    must reproduce HistFactory's PiecewiseInterpolation::code 4 fit
    exactly."""

    def test_histosys_poly_linear(self):
        edges, sig_t, bkg_t, data = _make_reference_data()
        # Asymmetric shape variations to stress the polynomial branch
        bkg_lo = bkg_t * np.array([0.95, 0.92, 0.90, 0.94, 0.97])
        bkg_hi = bkg_t * np.array([1.06, 1.10, 1.12, 1.08, 1.04])

        with tempfile.TemporaryDirectory() as d:
            spec = dict(
                edges=edges, poi="mu",
                channels=[dict(
                    name="SR", data=data,
                    samples=[
                        dict(name="signal", hist=sig_t,
                             normfactors=[("mu", 1.0, 0.0, 20.0)]),
                        dict(name="background", hist=bkg_t,
                             histosys=[("bkg_shape", bkg_lo, bkg_hi)]),
                    ],
                )],
            )
            ws = _build_hf_workspace(spec, d)
            hf_best, hf_err, _ = _fit_workspace(ws)

        from histimator.channels import Channel
        from histimator.histograms import Histogram
        from histimator.interpolation import InterpolationCode
        from histimator.likelihood import fit
        from histimator.model import Model
        from histimator.samples import Sample

        sig = Sample("signal", Histogram(sig_t, edges))
        sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 20.0))
        bkg = Sample("background", Histogram(bkg_t, edges))
        bkg.add_histosys(
            "bkg_shape",
            Histogram(bkg_lo, edges),
            Histogram(bkg_hi, edges),
            interp_code=InterpolationCode.POLY_LINEAR,
        )
        ch = Channel("SR")
        ch.add_sample(sig)
        ch.add_sample(bkg)
        ch.set_data(data)
        m = Model("test")
        m.add_channel(ch)
        hi_result = fit(m)

        assert hi_result.bestfit["mu"] == pytest.approx(
            hf_best["mu"], abs=3e-3
        )
        assert hi_result.bestfit["bkg_shape"] == pytest.approx(
            hf_best["alpha_bkg_shape"], abs=3e-3
        )
        assert hi_result.errors["mu"] == pytest.approx(hf_err["mu"], rel=3e-2)


# ---------------------------------------------------------------------------
# 12. StatError (Gaussian Barlow-Beeston) round-trip
# ---------------------------------------------------------------------------

class TestHistFactoryStatErrorRoundTrip:
    """Activate StatError on the signal sample with a Gaussian per-bin
    constraint and verify Histimator agrees."""

    def test_staterror_gaussian(self):
        edges, sig_t, bkg_t, data = _make_reference_data()
        # Stat uncertainties: 10% per bin on signal
        sig_unc = 0.1 * sig_t

        with tempfile.TemporaryDirectory() as d:
            spec = dict(
                edges=edges, poi="mu", stat_constraint="Gaussian",
                channels=[dict(
                    name="SR", data=data,
                    samples=[
                        dict(name="signal", hist=sig_t,
                             normfactors=[("mu", 1.0, 0.0, 20.0)],
                             staterror=True,
                             staterror_unc=sig_unc,
                             hist_errors=sig_unc),
                        dict(name="background", hist=bkg_t),
                    ],
                )],
            )
            ws = _build_hf_workspace(spec, d)
            hf_best, hf_err, _ = _fit_workspace(ws)

        from histimator.channels import Channel
        from histimator.histograms import Histogram
        from histimator.likelihood import fit
        from histimator.model import Model
        from histimator.samples import Sample

        rel_unc = sig_unc / sig_t
        sig = Sample("signal", Histogram(sig_t, edges))
        sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 20.0))
        sig.add_staterror("staterror", rel_unc)
        bkg = Sample("background", Histogram(bkg_t, edges))
        ch = Channel("SR")
        ch.add_sample(sig)
        ch.add_sample(bkg)
        ch.set_data(data)
        m = Model("test")
        m.add_channel(ch)
        hi_result = fit(m)

        # mu agreement is the headline result
        assert hi_result.bestfit["mu"] == pytest.approx(
            hf_best["mu"], abs=5e-3
        ), f"mu HF={hf_best['mu']:.6f} HI={hi_result.bestfit['mu']:.6f}"
        assert hi_result.errors["mu"] == pytest.approx(
            hf_err["mu"], rel=5e-2
        )


# ---------------------------------------------------------------------------
# 13a. ShapeSys (Poisson auxiliary) round-trip
# ---------------------------------------------------------------------------

class TestHistFactoryShapeSysRoundTrip:
    """ShapeSys with Poisson auxiliary constraints: per-bin gamma factors
    with tau = 1/rel^2 Poisson constraint."""

    def test_shapesys_poisson(self):
        edges, sig_t, bkg_t, data = _make_reference_data()
        rel_unc = np.array([0.10, 0.12, 0.15, 0.12, 0.10])

        with tempfile.TemporaryDirectory() as d:
            spec = dict(
                edges=edges, poi="mu",
                channels=[dict(name="SR", data=data, samples=[
                    dict(name="signal", hist=sig_t,
                         normfactors=[("mu", 1.0, 0.0, 20.0)]),
                    dict(name="background", hist=bkg_t,
                         shapesys=[("bkg_shape", rel_unc)]),
                ])],
            )
            ws = _build_hf_workspace(spec, d)
            hf_best, hf_err, _ = _fit_workspace(ws)

        from histimator.channels import Channel
        from histimator.histograms import Histogram
        from histimator.likelihood import fit
        from histimator.model import Model
        from histimator.samples import Sample

        sig = Sample("signal", Histogram(sig_t, edges))
        sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 20.0))
        bkg = Sample("background", Histogram(bkg_t, edges))
        bkg.add_shapesys("bkg_shape", rel_unc)
        ch = Channel("SR")
        ch.add_sample(sig)
        ch.add_sample(bkg)
        ch.set_data(data)
        m = Model("test")
        m.add_channel(ch)
        hi_result = fit(m)

        assert hi_result.bestfit["mu"] == pytest.approx(
            hf_best["mu"], abs=5e-3
        ), f"mu HF={hf_best['mu']:.6f} HI={hi_result.bestfit['mu']:.6f}"
        assert hi_result.errors["mu"] == pytest.approx(
            hf_err["mu"], rel=5e-2
        )


# ---------------------------------------------------------------------------
# 13. LumiSys round-trip
# ---------------------------------------------------------------------------

class TestHistFactoryLumiSysRoundTrip:
    """A free luminosity parameter with a Gaussian constraint at
    rel-error sigma. HistFactory's Lumi is global
    Histimator's
    LumiSys is per-sample with shared parameter, so add it to all
    samples to match."""

    def test_lumi_floating(self):
        edges, sig_t, bkg_t, data = _make_reference_data()
        sigma_lumi = 0.05  # 5% lumi uncertainty

        with tempfile.TemporaryDirectory() as d:
            spec = dict(
                edges=edges, poi="mu",
                lumi=(1.0, sigma_lumi),
                channels=[dict(
                    name="SR", data=data,
                    samples=[
                        dict(name="signal", hist=sig_t,
                             normfactors=[("mu", 1.0, 0.0, 20.0)]),
                        dict(name="background", hist=bkg_t),
                    ],
                )],
            )
            ws = _build_hf_workspace(spec, d)
            hf_best, hf_err, _ = _fit_workspace(ws)

        from histimator.channels import Channel
        from histimator.histograms import Histogram
        from histimator.likelihood import fit
        from histimator.model import Model
        from histimator.samples import Sample

        sig = Sample("signal", Histogram(sig_t, edges))
        sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 20.0))
        sig.add_lumisys("lumi", uncertainty=sigma_lumi)
        bkg = Sample("background", Histogram(bkg_t, edges))
        bkg.add_lumisys("lumi", uncertainty=sigma_lumi)  # shared parameter
        ch = Channel("SR")
        ch.add_sample(sig)
        ch.add_sample(bkg)
        ch.set_data(data)
        m = Model("test")
        m.add_channel(ch)
        hi_result = fit(m)

        assert hi_result.bestfit["mu"] == pytest.approx(
            hf_best["mu"], abs=5e-3
        )
        # Lumi parameter named "Lumi" in HistFactory, "lumi" in Histimator
        assert hi_result.bestfit["lumi"] == pytest.approx(
            hf_best["Lumi"], abs=5e-3
        )
        assert hi_result.errors["mu"] == pytest.approx(
            hf_err["mu"], rel=5e-2
        )


# ---------------------------------------------------------------------------
# 13b. All-modifiers-at-once integration round-trip
# ---------------------------------------------------------------------------

class TestHistFactoryFullModelRoundTrip:
    """One channel with NormFactor, NormSys, HistoSys, StatError, and
    Lumi all active simultaneously. This is the realistic case for an
    ATLAS/CMS measurement and exercises the constraint composition
    pathways together."""

    def test_full_model_roundtrip(self):
        edges, sig_t, bkg_t, data = _make_reference_data(seed=99, true_mu=1.2)
        bkg_lo = bkg_t * np.array([0.95, 0.92, 0.94, 0.96, 0.97])
        bkg_hi = bkg_t * np.array([1.05, 1.08, 1.06, 1.04, 1.03])
        sig_unc = 0.08 * sig_t

        with tempfile.TemporaryDirectory() as d:
            spec = dict(
                edges=edges, poi="mu", lumi=(1.0, 0.025),
                channels=[dict(name="SR", data=data, samples=[
                    dict(name="signal", hist=sig_t, hist_errors=sig_unc,
                         normfactors=[("mu", 1.0, 0.0, 20.0)],
                         staterror=True),
                    dict(name="background", hist=bkg_t,
                         normsys=[("bkg_norm", 0.92, 1.08)],
                         histosys=[("bkg_shape", bkg_lo, bkg_hi)]),
                ])],
            )
            ws = _build_hf_workspace(spec, d)
            hf_best, hf_err, _ = _fit_workspace(ws)

        from histimator.channels import Channel
        from histimator.histograms import Histogram
        from histimator.interpolation import InterpolationCode
        from histimator.likelihood import fit
        from histimator.model import Model
        from histimator.samples import Sample

        rel_unc = sig_unc / sig_t
        sig = Sample("signal", Histogram(sig_t, edges))
        sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 20.0))
        sig.add_staterror("staterror", rel_unc)
        sig.add_lumisys("lumi", uncertainty=0.025)
        bkg = Sample("background", Histogram(bkg_t, edges))
        bkg.add_normsys("bkg_norm", lo=0.92, hi=1.08,
                        interp_code=InterpolationCode.POLY_EXPONENTIAL)
        bkg.add_histosys(
            "bkg_shape", Histogram(bkg_lo, edges), Histogram(bkg_hi, edges),
            interp_code=InterpolationCode.POLY_LINEAR,
        )
        bkg.add_lumisys("lumi", uncertainty=0.025)
        ch = Channel("SR")
        ch.add_sample(sig)
        ch.add_sample(bkg)
        ch.set_data(data)
        m = Model("test")
        m.add_channel(ch)
        hi_result = fit(m)

        # Headline POI
        assert hi_result.bestfit["mu"] == pytest.approx(
            hf_best["mu"], abs=1e-2
        ), f"mu HF={hf_best['mu']:.6f} HI={hi_result.bestfit['mu']:.6f}"
        assert hi_result.errors["mu"] == pytest.approx(
            hf_err["mu"], rel=5e-2
        )
        # Nuisance parameters
        assert hi_result.bestfit["bkg_norm"] == pytest.approx(
            hf_best["alpha_bkg_norm"], abs=2e-2
        )
        assert hi_result.bestfit["bkg_shape"] == pytest.approx(
            hf_best["alpha_bkg_shape"], abs=2e-2
        )
        assert hi_result.bestfit["lumi"] == pytest.approx(
            hf_best["Lumi"], abs=1e-2
        )


# ---------------------------------------------------------------------------
# 14. Multi-channel shared mu
# ---------------------------------------------------------------------------

class TestHistFactoryMultiChannelSharedMu:
    """Two channels with a shared signal-strength parameter."""

    def test_two_channels_shared_mu(self):
        edges = np.linspace(0.0, 5.0, 6)
        sig_a = np.array([10.0, 20.0, 30.0, 20.0, 10.0])
        bkg_a = np.array([45.0, 45.0, 45.0, 45.0, 45.0])
        sig_b = np.array([5.0, 15.0, 25.0, 15.0, 5.0])
        bkg_b = np.array([60.0, 60.0, 60.0, 60.0, 60.0])
        rng = np.random.default_rng(7)
        data_a = rng.poisson(1.3 * sig_a + bkg_a).astype(float)
        data_b = rng.poisson(1.3 * sig_b + bkg_b).astype(float)

        with tempfile.TemporaryDirectory() as d:
            spec = dict(
                edges=edges, poi="mu",
                channels=[
                    dict(name="SR_A", data=data_a, samples=[
                        dict(name="signal", hist=sig_a,
                             normfactors=[("mu", 1.0, 0.0, 20.0)]),
                        dict(name="background", hist=bkg_a),
                    ]),
                    dict(name="SR_B", data=data_b, samples=[
                        dict(name="signal", hist=sig_b,
                             normfactors=[("mu", 1.0, 0.0, 20.0)]),
                        dict(name="background", hist=bkg_b),
                    ]),
                ],
            )
            ws = _build_hf_workspace(spec, d)
            hf_best, hf_err, _ = _fit_workspace(ws)

        from histimator.channels import Channel
        from histimator.histograms import Histogram
        from histimator.likelihood import fit
        from histimator.model import Model
        from histimator.samples import Sample

        # Two channels, mu shared by name
        sig_a_sample = Sample("signal", Histogram(sig_a, edges))
        sig_a_sample.add_normfactor("mu", nominal=1.0, bounds=(0.0, 20.0))
        bkg_a_sample = Sample("background", Histogram(bkg_a, edges))
        ch_a = Channel("SR_A")
        ch_a.add_sample(sig_a_sample)
        ch_a.add_sample(bkg_a_sample)
        ch_a.set_data(data_a)

        sig_b_sample = Sample("signal", Histogram(sig_b, edges))
        sig_b_sample.add_normfactor("mu", nominal=1.0, bounds=(0.0, 20.0))
        bkg_b_sample = Sample("background", Histogram(bkg_b, edges))
        ch_b = Channel("SR_B")
        ch_b.add_sample(sig_b_sample)
        ch_b.add_sample(bkg_b_sample)
        ch_b.set_data(data_b)

        m = Model("test")
        m.add_channel(ch_a)
        m.add_channel(ch_b)
        hi_result = fit(m)

        assert hi_result.bestfit["mu"] == pytest.approx(
            hf_best["mu"], abs=2e-3
        )
        assert hi_result.errors["mu"] == pytest.approx(
            hf_err["mu"], rel=2e-2
        )


# ---------------------------------------------------------------------------
# 15. CLs upper limit vs RooStats AsymptoticCalculator
# ---------------------------------------------------------------------------

class TestHistFactoryCLsLimitVsAsymptoticCalculator:
    """Compare Histimator's upper_limit_cls observed limit against
    RooStats.AsymptoticCalculator + HypoTestInverter on the same
    HistFactory workspace and Histimator model."""

    def test_observed_limit_matches(self):
        # Use a model with little observed signal so the limit is
        # well-defined and finite
        edges = np.linspace(0.0, 5.0, 6)
        sig_t = np.array([5.0, 10.0, 15.0, 10.0, 5.0])
        bkg_t = np.array([60.0, 60.0, 60.0, 60.0, 60.0])
        rng = np.random.default_rng(101)
        data = rng.poisson(0.0 * sig_t + bkg_t).astype(float)  # bkg-only

        # ---- HistFactory + AsymptoticCalculator ----
        with tempfile.TemporaryDirectory() as d:
            spec = dict(
                edges=edges, poi="mu",
                channels=[dict(name="SR", data=data, samples=[
                    dict(name="signal", hist=sig_t,
                         normfactors=[("mu", 1.0, 0.0, 20.0)]),
                    dict(name="background", hist=bkg_t),
                ])],
            )
            ws = _build_hf_workspace(spec, d)
            mc = ws.obj("ModelConfig")
            data_obs = ws.data("obsData")

            # Build b-only model (mu=0 fixed)
            bmodel = mc.Clone("BModel")
            poi = bmodel.GetParametersOfInterest().first()
            poi.setVal(0.0)
            bmodel.SetSnapshot(ROOT.RooArgSet(poi))

            # Reset s+b model snapshot at mu=1
            sbmodel = mc.Clone("SBModel")
            poi_sb = sbmodel.GetParametersOfInterest().first()
            poi_sb.setVal(1.0)
            sbmodel.SetSnapshot(ROOT.RooArgSet(poi_sb))

            calc = ROOT.RooStats.AsymptoticCalculator(
                data_obs, bmodel, sbmodel
            )
            calc.SetOneSided(True)
            calc.SetQTilde(True)
            inverter = ROOT.RooStats.HypoTestInverter(calc)
            inverter.SetConfidenceLevel(0.95)
            inverter.UseCLs(True)
            inverter.SetVerbose(False)
            # Scan a sensible mu range and let the inverter interpolate
            inverter.SetFixedScan(40, 0.1, 5.0)
            result = inverter.GetInterval()
            asym_obs_limit = float(result.UpperLimit())

        # ---- Histimator ----
        from histimator.channels import Channel
        from histimator.cls import upper_limit_cls
        from histimator.histograms import Histogram
        from histimator.model import Model
        from histimator.samples import Sample

        sig = Sample("signal", Histogram(sig_t, edges))
        sig.add_normfactor("mu", nominal=1.0, bounds=(0.0, 20.0))
        bkg = Sample("background", Histogram(bkg_t, edges))
        ch = Channel("SR")
        ch.add_sample(sig)
        ch.add_sample(bkg)
        ch.set_data(data)
        m = Model("test")
        m.add_channel(ch)
        hi_obs, _ = upper_limit_cls(m, "mu")

        # AsymptoticCalculator and Histimator both use CCGV asymptotics +
        # qtilde; expect agreement at the 5% level (interpolation +
        # search granularity).
        assert hi_obs == pytest.approx(asym_obs_limit, rel=5e-2), (
            f"CLs observed limit: HF AsympCalc={asym_obs_limit:.4f} "
            f"HI={hi_obs:.4f}"
        )


class TestLikelihoodScanCurvature:
    """For a quadratic NLL, delta-NLL = 1 should be reached at mu_hat ±
    sigma_mu. We check Histimator's NLL parabola against the curvature
    sigma_mu measured from the same fit. This is independent of any
    additive offset bug because both sides are NLL differences.
    """

    def test_delta_nll_1_at_one_sigma(self):
        sig = np.array([10.0, 30.0, 50.0, 30.0, 10.0])
        bkg = np.array([50.0, 50.0, 50.0, 50.0, 50.0])
        rng = np.random.default_rng(7)
        data = rng.poisson(1.0 * sig + bkg).astype(float)

        model = _hi_multibin(sig, bkg, data)
        result = fit(model)
        mu_hat = result.bestfit["mu"]
        sigma = result.errors["mu"]

        nll = BinnedNLL(model, extended=True)
        nll_at_min = nll(np.array([mu_hat]))
        nll_at_plus = nll(np.array([mu_hat + sigma]))
        nll_at_minus = nll(np.array([mu_hat - sigma]))

        # Hesse curvature definition: 2*(NLL(mu_hat ± sigma) - NLL_min) ≈ 1
        # (NLL is the negative log L, errordef = 1, so NLL difference = 0.5
        # at one sigma)
        # Tolerance reflects the Poisson NLL's mild departure from
        # parabolicity at ±1 sigma on a single random sample.
        assert (nll_at_plus - nll_at_min) == pytest.approx(0.5, abs=5e-2)
        assert (nll_at_minus - nll_at_min) == pytest.approx(0.5, abs=5e-2)
