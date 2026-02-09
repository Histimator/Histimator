# Changelog

## 0.3.0 (2025-02-XX)

**This is a complete rewrite.  The API is not backward-compatible with
0.2.x.**

### What changed

The package has been rebuilt from scratch around a new imperative model
specification API.  The old `HistiSample` / `HistiModel` / `probfit`
interface has been replaced by a composable system of Parameter,
Histogram, Sample, Channel, and Model objects.

### New capabilities

- Seven modifier types: NormFactor, NormSys, HistoSys, StatError,
  ShapeSys, ShapeFactor, LumiSys
- Four interpolation schemes (piecewise linear, piecewise exponential,
  quadratic-linear, polynomial-exponential)
- CCGV test statistics (q0, qmu, qtilde_mu) with asymptotic p-values
- CLs method with Asimov datasets and Brazil bands
- Toy-based pseudo-experiment calculator
- Likelihood scans (1D and 2D) with interval extraction
- Saturated-model goodness-of-fit test
- Fit diagnostics: pre/post-fit yields, nuisance parameter pulls,
  impact ranking
- 233 unit tests
- 11 pedagogical notebooks (probability through full analysis)

### Breaking changes

- Minimum Python version is now 3.10 (was 2.7)
- `histimator.models.HistiSample` no longer exists
- `probfit` dependency removed; replaced by built-in BinnedNLL
- Package uses `src/` layout with `hatchling` build backend
- All imports have changed (see notebooks for the new API)

### Migration

There is no automated migration path from 0.2.x.  The old API is
preserved on the `legacy-v0.2.x` branch and the `v0.2.3` tag.

## 0.2.3

(Unreleased on PyPI. Available via the v0.2.3 tag and legacy-v0.2.x
branch.)

## 0.2.1 (2018-02-25)

Last version published to PyPI under the original API.
