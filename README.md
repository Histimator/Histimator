# Histimator

Histogram-template statistical inference for particle physics and beyond.

Histimator builds statistical models from binned histogram templates and performs maximum-likelihood estimation with systematic uncertainties, following the HistFactory formalism that underpins every ATLAS and CMS result at the LHC. Version 2.0.0a1 implements the complete inference chain from model construction through CLs limit setting, toy-based validation, and analyst-facing diagnostics.


## Motivation

The particle physics community currently fragments its statistical inference across several tools. RooFit and RooStats provide the C++ reference implementation within ROOT but carry decades of API weight and require compilation.

Histimator presents a fresh approach with a focus on interpretability. If you're doing statistics you need to understand what it is you're doing. Histimator just helps with that. The model specification is imperative Python: you construct Parameter, Histogram, Sample, Channel, and Model objects directly and compose them with method calls. There is no intermediate serialization format between the analyst and the likelihood. The inference layer implements the same asymptotic formulae as presented in the seminal *'Asimov'* paper (arXiv:1007.1727) that RooStats and other tools use, produces the same CLs limits with Brazil bands, and provides the same fit diagnostic outputs, so results are directly comparable. The package is pure Python with dependencies limited to NumPy, SciPy, and iminuit. No pesky json or xml required. Just pure python.


## Installation

```
pip install histimator
```

For development:

```
pip install histimator[dev]
```

Requires Python 3.10 or later. Dependencies are NumPy (>= 1.23), SciPy (>= 1.9), and iminuit (>= 2.20).


## Architecture

The package is organised into layers that correspond to the stages of a HistFactory analysis.

The foundation layer handles data representation. `Parameter` wraps a named floating-point value with bounds, a nominal value, and fixed/floating status. `Histogram` stores binned data with edges, values, and optional sumw2 arrays, and provides arithmetic operations (addition, scalar multiplication, division) along with integration and density computation. These are defined in `parameters.py` and `histograms.py`.

The model layer assembles physics content. A `Sample` represents a single physics process (signal or background) as a nominal histogram modified by zero or more systematic variations. Modifiers attach to samples and include `NormFactor` (free-floating overall normalisation), `NormSys` (Gaussian-constrained normalisation), `HistoSys` (correlated shape variation driven by a single nuisance parameter), `StatError` (per-bin Barlow-Beeston gamma parameters with Gaussian constraints), `ShapeSys` (per-bin gammas with Poisson constraints), `ShapeFactor` (unconstrained per-bin normalisation), and `LumiSys` (correlated luminosity uncertainty). A `Channel` groups samples that share the same binning and observed data. A `Model` collects channels into a complete analysis. These live in `samples.py`, `channels.py`, and `model.py`.

The interpolation layer (`interpolation.py`) provides four interpolation codes for mapping nuisance parameter values to systematic variations. Code 0 is piecewise linear. Code 1 is piecewise exponential, guaranteeing positivity. Code 2 is quadratic near the origin with linear extrapolation beyond +/-1 sigma. Code 3 is polynomial near the origin with exponential extrapolation, combining smoothness with positivity. All four satisfy the boundary conditions: alpha=0 returns nominal, alpha=+1 returns the up variation, alpha=-1 returns the down variation.

The likelihood layer (`likelihood.py`) constructs and minimises the negative log-likelihood. `BinnedNLL` computes the extended Poisson likelihood across all channels with Gaussian and Poisson constraint terms for nuisance parameters. The `fit` function wraps iminuit's MIGRAD, HESSE, and optional MINOS to return best-fit values, symmetric and asymmetric errors, the covariance matrix, and fit quality diagnostics.

The inference layer spans several modules. `inference.py` provides the basic building blocks: `profile_likelihood_ratio`, `upper_limit` (profile likelihood ratio inversion), and `discovery_significance`. `test_statistics.py` implements the three CCGV test statistics (q0, qmu, qtilde_mu) and their asymptotic p-value distributions. `cls.py` builds on these to provide Asimov datasets, the CLs statistic, expected limit bands at the five standard quantiles, and full CLs upper limit scans. `toys.py` provides the non-asymptotic alternative: pseudo-data generation, nuisance parameter sampling from priors, and toy-based CLs computation for the low-statistics regime where the asymptotic approximation breaks down. `scans.py` adds 1D and 2D profiled likelihood scans with interval extraction and saturated-model goodness-of-fit testing. `diagnostics.py` provides pre-fit/post-fit yield extraction, nuisance parameter pull computation, and impact ranking.


## The 'Asymptotic' Test Statistics

The entire LHC exclusion and discovery programme rests on three bounded test statistics defined in Cowan, Cranmer, Gross, and Vitells (Eur. Phys. J. C 71, 1554, 2011). Understanding why they differ from the naive profile likelihood ratio is essential to understanding the inference layer.

The naive two-sided statistic t(mu) = 2 * [NLL(mu) - NLL(mu_hat)] treats all departures from the hypothesis equally. This is wrong for hypothesis testing at the LHC because the direction of the departure matters.

For discovery, the relevant question is whether the data shows an excess over background. A deficit (mu_hat < 0) should not count as evidence for signal. The discovery statistic q0 therefore equals t(0) when mu_hat >= 0 and zero otherwise. Its asymptotic distribution under background-only is a mixture of a delta function at zero (weight 1/2) and chi2(1) (weight 1/2), producing p-values via p0 = 1 - Phi(sqrt(q0)).

For exclusion, the relevant question is whether the data is incompatible with a signal hypothesis at strength mu. An upward fluctuation (mu_hat > mu) should not count against the hypothesis because it means the signal is even larger than assumed. The exclusion statistic qmu therefore equals t(mu) when mu_hat <= mu and zero otherwise.

The physical boundary variant qtilde_mu additionally enforces mu_hat >= 0. When the unconstrained best fit is negative (which happens when background fluctuates down), the denominator of the likelihood ratio switches from NLL(mu_hat) to NLL(0). This ensures the test statistic remains well-defined and produces the correct CLs values in the deficit regime.

The CLs quantity itself is CLs = p_{s+b} / (1 - p_b), where p_{s+b} and p_b are the tail probabilities under the signal-plus-background and background-only hypotheses respectively. Dividing by (1 - p_b) protects against excluding signal hypotheses for which the experiment has no sensitivity. The observed CLs limit is the signal strength where CLs = 0.05. The expected limit replaces observed data with the Asimov dataset (where every bin equals the expected yield under background-only) and evaluates CLs at five quantiles of the background-only test statistic distribution to produce the Brazil band.


## Modifier Types

Histimator implements seven modifier types that collectively cover the systematic uncertainty model of any HistFactory analysis.

`NormFactor` is a free-floating multiplicative parameter with no constraint term. It scales all bins of a sample uniformly. The signal strength mu is typically a NormFactor.

`NormSys` is a Gaussian-constrained normalisation. A nuisance parameter alpha, constrained at N(0,1), interpolates the sample normalisation between down (alpha = -1) and up (alpha = +1) variations using the selected interpolation code.

`HistoSys` is a Gaussian-constrained shape variation. Like NormSys, a nuisance parameter alpha interpolates between up and down histogram templates, but the variation can be different in each bin, allowing shape changes (tilts, shifts, broadening).

`StatError` implements the Barlow-Beeston approach to MC statistical uncertainties. One gamma parameter per bin, shared across all samples in a channel, is constrained at N(1, delta_i) where delta_i is the relative statistical uncertainty in bin i. The gamma multiplies the expected yield in the corresponding bin, allowing the fit to absorb template statistical fluctuations at the cost of widened parameter uncertainties.

`ShapeSys` provides per-bin gamma parameters with Poisson auxiliary constraints. The Poisson parameter tau = 1/rel^2 sets the constraint strength. This is appropriate when the uncertainty in a bin derives from an auxiliary measurement (such as a control region sideband) rather than from MC sample size.

`ShapeFactor` provides completely unconstrained per-bin normalisation parameters. Because there is no constraint term, the fit can reproduce any data shape exactly. This is the standard approach for data-driven background estimation where the background shape in a signal region is determined entirely by the fit to data.

`LumiSys` is a single Gaussian-constrained parameter at N(1, sigma_lumi) that scales all bins of a sample coherently. It enforces the correlation between luminosity uncertainty effects on different samples automatically, which would be lost if one applied separate NormSys modifiers to each sample.


## Toy-Based Inference

The asymptotic formulae from the CCGV paper assume large sample sizes. When bin counts drop below roughly 5 to 10 events, which is common in searches for rare processes, the asymptotic approximation becomes unreliable. The toy-based calculator in `toys.py` addresses this.

`generate_toy_data` draws Poisson pseudo-data from the model's expected yields at specified parameter values. `sample_nuisance_parameters` draws constrained nuisance parameters from their prior distributions (N(0,1) for alphas, N(1, delta) for StatError gammas, Gamma for ShapeSys gammas). Together they produce complete pseudo-experiments.

`toy_cls` generates pseudo-experiments under both the signal-plus-background and background-only hypotheses, computes the test statistic for each toy, and estimates CLs by counting tail fractions. `toy_cls_limit` scans signal strength values and interpolates to find the CLs = 0.05 crossing. At high statistics the toy-based and asymptotic results agree, which is validated in the test suite. At low statistics they diverge, and the toy-based result is the one to trust.


## Fit Diagnostics

Three diagnostic functions in `diagnostics.py` produce the standard analyst-facing validation outputs.

`prefit_postfit_yields` evaluates the model at nominal parameter values (pre-fit) and at the best-fit values (post-fit), returning per-channel, per-sample breakdowns together with the observed data. These are the ingredients for the data/MC comparison plots that appear in every analysis note.

`nuisance_parameter_pulls` computes the pull (how far each constrained parameter moved from its nominal, in units of the pre-fit uncertainty) and the constraint ratio (how much the post-fit uncertainty shrank relative to the pre-fit uncertainty). A parameter with a large pull indicates tension between the data and the prior assumption. A constraint ratio well below 1.0 indicates the data is informative beyond the prior.

`impacts` measures, for each nuisance parameter, the shift in the parameter of interest caused by fixing that nuisance at its +/- 1 sigma post-fit values and refitting. The output is sorted by decreasing impact magnitude. This is the standard "impact plot" used in every ATLAS and CMS measurement to identify which systematic uncertainties matter most.


## Likelihood Scans and Goodness-of-Fit

`likelihood_scan_1d` profiles -2 * delta(ln L) as a function of a single parameter, minimising over all others at each scan point. The resulting curve gives confidence intervals by intersection with horizontal thresholds: delta_nll = 1.0 for 68% CL, delta_nll = 3.84 for 95% CL. Unlike the Hesse approximation (which assumes a parabolic likelihood), the scan captures asymmetries.

`likelihood_scan_2d` does the same on a 2D grid of two parameters, producing the data for contour plots. The off-diagonal structure reveals correlations between parameters.

`goodness_of_fit` computes the saturated-model test statistic: the difference in -2 ln L between the best fit and the saturated model (where every bin's expected yield equals its observed count). Under the null hypothesis that the model is correct, this statistic is asymptotically chi2-distributed with ndof = nbins - nfloated. A small p-value indicates the model cannot describe the data.


## Testing

The test suite contains 233 tests across 11 modules, organised by sprint. Every test targets a specific physical or statistical property. Closed-form analytical results (counting experiment q0, Asimov significance, expected limits) provide exact validation where possible. Cross-validation between asymptotic and toy-based methods checks internal consistency. Physical constraints (non-negativity, monotonicity, boundary behaviour, correct pull direction) catch implementation errors that numerical agreement alone would miss. See `tests/README.md` for a complete listing with justifications.

Run the tests:

```
python -m pytest tests/ -v
```


## Status

Version 2.0.0a1 implements the complete HistFactory inference chain. The remaining items from the original gap analysis are JSON workspace serialization (pyhf compatibility), Feldman-Cousins intervals, and channel masking. These are planned for subsequent development.


## References

Cowan, Cranmer, Gross, Vitells. "Asymptotic formulae for likelihood-based tests of new physics." Eur. Phys. J. C 71, 1554 (2011). arXiv:1007.1727.

Cranmer, Lewis, Moneta, Shibata, Verkerke. "HistFactory: A tool for creating statistical models for use with RooFit and RooStats." CERN-OPEN-2012-016 (2012).

Read. "Presentation of search results: the CLs technique." J. Phys. G 28, 2693 (2002).

Barlow, Beeston. "Fitting using finite Monte Carlo samples." Comput. Phys. Commun. 77, 219 (1993).