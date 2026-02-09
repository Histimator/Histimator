# Histimator Test Suite

233 tests across 11 modules. Run with `python -m pytest tests/ -v` from the repository root.

The validation strategy has five layers. First, closed-form analytical results provide exact ground truth where the mathematics permits it, such as the counting experiment q0 or the Asimov significance from CCGV equation 97. Second, cross-validation between independent code paths (asymptotic vs. toy-based CLs, scan-derived intervals vs. Hesse errors) checks internal consistency without requiring external reference values. Third, boundary and regime tests (q0 = 0 when deficit, CLs in [0,1], gammas non-negative) enforce mathematical constraints that any correct implementation must satisfy regardless of the specific numerical inputs. Fourth, physical direction tests (pulls toward data, anti-correlation between signal strength and background normalisation, NormSys impact larger than HistoSys impact) verify that the code responds to data in the way the physics demands. Fifth, structural tests (output shapes, parameter counts, key presence) guard against regressions in the API contract.

Each test below is listed with the property it validates and why that property matters.


## test_sprint2_core.py (26 tests)

These tests validate the foundational data types. If Parameter or Histogram is broken, nothing downstream can work.

`TestParameter::test_basic_construction` verifies that a Parameter stores its name, value, and bounds correctly. This is the contract that every other module depends on when it reads parameter metadata.

`TestParameter::test_defaults` confirms that omitting optional arguments produces sensible defaults (unbounded, not fixed). Analysts should be able to create simple parameters without specifying every field.

`TestParameter::test_bounds_validation_inverted` and `test_bounds_validation_equal` confirm that physically nonsensical bounds (lo > hi, lo == hi) raise ValueError. Silent acceptance of bad bounds would produce confusing fit failures downstream.

`TestParameter::test_value_outside_bounds` confirms that a nominal value outside the declared bounds is rejected. Minuit would silently clip the value, producing fits that start at the wrong point.

`TestParameter::test_clone` verifies deep copy semantics. Model-copying operations (used in Asimov dataset construction and impact computation) depend on cloned parameters being independent of their originals.

`TestParameter::test_to_minuit_init`, `test_to_minuit_init_fixed`, `test_to_minuit_init_unbounded` check the translation from Parameter metadata to iminuit's initialisation dictionary. Incorrect translation would produce silent fit failures (wrong bounds, wrong fixed status).

`TestHistogram::test_basic_construction` confirms that values and edges are stored and retrievable. This is the data integrity baseline.

`TestHistogram::test_edges_values_mismatch` and `test_non_monotonic_edges` validate input checking. Mismatched array lengths or non-monotonic bin edges indicate corrupted input and must be caught immediately.

`TestHistogram::test_1d_validation` ensures only 1D histograms are accepted. The current implementation assumes 1D throughout; accepting higher dimensions silently would produce wrong results.

`TestHistogram::test_immutability` verifies that modifying the input arrays after construction does not change the Histogram. Shared mutable state between a Histogram and its source array would cause action-at-a-distance bugs.

`TestHistogram::test_centres`, `test_widths`, `test_density` check derived quantities against hand-computed values. These are used in integration and plotting.

`TestHistogram::test_integration_full_range`, `test_integration_partial_bin`, `test_integration_single_bin`, `test_integration_out_of_range` validate the integrate method against analytical results. Integration is used in normalisation and cross-section extraction.

`TestHistogram::test_from_numpy` checks construction from NumPy histogram output. This is the primary data ingestion path.

`TestHistogram::test_addition`, `test_addition_incompatible`, `test_scalar_multiply`, `test_rmul`, `test_division_by_scalar`, `test_division_by_histogram`, `test_division_by_zero_histogram` validate arithmetic operations and their error handling. These operations are used when combining samples and computing ratios.

`TestHistogram::test_equality`, `test_inequality`, `test_repr` check comparison and display. Equality is used in Asimov data validation.

`TestHistogram::test_sumw2_custom` and `test_sumw2_default_poisson` verify that sum-of-weights-squared is stored correctly and defaults to Poisson variance (equal to the values) when not provided. Incorrect sumw2 would propagate into wrong StatError constraints.


## test_sprint3_interpolation.py (20 tests)

Interpolation maps nuisance parameter values to systematic variations. If the interpolation is wrong, every modifier that uses it produces wrong expected yields.

`TestBoundaryConditions::test_alpha_zero_returns_nominal` (parametrised over 4 codes) verifies the fundamental identity: no systematic shift when the nuisance parameter is at its nominal value. Violation would mean the model prediction changes even when no systematic is applied.

`TestBoundaryConditions::test_alpha_plus_one_returns_up` and `test_alpha_minus_one_returns_down` (parametrised over 4 codes) verify that +/-1 sigma shifts reproduce the provided up/down templates exactly. These are the calibration points of the interpolation.

`TestPiecewiseLinear::test_extrapolation_positive` and `test_extrapolation_negative` check that code 0 extrapolates linearly beyond +/-1. Incorrect extrapolation would understate or overstate large systematic shifts.

`TestPiecewiseLinear::test_scalar_input` verifies that scalar (non-array) inputs work. Some model configurations have scalar nominal values.

`TestPiecewiseExponential::test_always_positive` confirms code 1 never produces negative yields, which is its defining property and the reason it exists. Negative expected yields cause NaN in the Poisson log-likelihood.

`TestPiecewiseExponential::test_symmetric_10pct` checks that a symmetric 10% variation produces the expected exponential interpolation. This validates the mathematical formula against a hand calculation.

`TestQuadraticLinear::test_smooth_at_zero` verifies that code 2 has a continuous first derivative at alpha = 0 (the quadratic-to-linear transition). A discontinuous derivative would cause Minuit to converge slowly or oscillate.

`TestQuadraticLinear::test_linear_extrapolation_above_1` confirms that beyond +/-1 sigma, code 2 reverts to linear extrapolation.

`TestPolyExponential::test_continuity_at_boundary` checks that code 3 is continuous at the polynomial-to-exponential transition. Discontinuities cause the same Minuit convergence issues.

`TestPolyExponential::test_always_positive` confirms code 3 maintains positivity, like code 1 but with a smoother interior.

`TestEdgeCases::test_zero_variation` (parametrised over 4 codes) checks that when up = down = nominal, the output equals nominal at all alpha values. This is the identity limit.

`TestEdgeCases::test_scalar_nominal` (parametrised over 4 codes) handles the case where inputs are scalars rather than arrays.


## test_sprint4_model.py (18 tests)

These tests validate the model assembly layer. Incorrect model structure silently produces wrong likelihoods.

`TestSample::test_basic_expected` confirms that a sample with no modifiers returns its nominal histogram values. This is the baseline for the modifier chain.

`TestSample::test_normfactor` and `test_normfactor_default` verify that NormFactor scales all bins uniformly and that the default value (1.0) leaves the sample unchanged.

`TestSample::test_normsys` checks that a NormSys modifier at alpha = +1 produces the declared up variation.

`TestSample::test_histosys` verifies bin-by-bin shape interpolation. `test_histosys_bin_mismatch` confirms that incompatible histogram shapes are rejected.

`TestSample::test_chained_modifiers` verifies that multiple modifiers compose correctly (multiplicative chain).

`TestSample::test_parameters_property` checks that all modifier parameters are discoverable from the sample. The likelihood layer uses this to build the parameter list.

`TestChannel::test_add_samples_and_expected` confirms that channel-level expected yields sum the sample contributions.

`TestChannel::test_bin_mismatch_rejected` and `test_edge_mismatch_rejected` enforce that all samples in a channel share the same binning. Mismatched binning would produce a likelihood that adds Poisson terms across different physical regions.

`TestChannel::test_set_data_array`, `test_set_data_histogram`, `test_set_data_wrong_bins` validate data attachment and input checking.

`TestChannel::test_parameters_merged` confirms that parameters from different samples are collected without duplication.

`TestModel::test_single_channel_expected`, `test_single_channel_data` check that Model correctly delegates to its channels. `test_parameters`, `test_nominal_values`, `test_to_minuit_init` verify the parameter discovery and translation chain. `test_total_nbins` and `test_channel_slices` check the bookkeeping that BinnedNLL uses to index into the concatenated data vector. `test_no_data_raises` confirms that attempting to build a likelihood without data fails early. `test_multi_channel` and `test_shared_parameter_across_channels` validate multi-channel models and parameter sharing.


## test_sprint5_likelihood.py (11 tests)

The likelihood is the core computation. Every downstream result (fits, limits, p-values) is a function of the likelihood.

`TestPoissonLogPdf::test_known_value` validates the Poisson log-probability against a hand-computed value. This is the atomic unit of the binned likelihood.

`TestPoissonLogPdf::test_zero_data` confirms the correct behaviour when observed counts are zero (common in rare-process searches). An implementation that uses log(data!) naively would produce NaN.

`TestPoissonLogPdf::test_vectorised` verifies that the function handles array inputs.

`TestBinnedNLL::test_nll_callable` confirms that the NLL object is callable with keyword arguments matching the model parameters. `test_nll_minimum_at_true` verifies that the NLL is minimised at the true parameter values for data generated from the model. This is the fundamental consistency check. `test_nll_has_parameters` checks that the NLL exposes its parameter names, which Minuit needs.

`TestFit::test_fit_recovers_mu` confirms that the fit returns the correct signal strength for a simple model. `test_fit_result_has_errors` and `test_fit_result_has_covariance` check that Hesse uncertainties and the covariance matrix are populated. `test_fit_with_minos` verifies asymmetric error computation. `test_fit_with_nuisance` checks that constrained nuisance parameters are profiled correctly. `test_fit_multi_channel` validates fitting across multiple channels. `test_fit_nll_min_returned` confirms that the minimum NLL value is recorded for use in likelihood ratio computations.


## test_sprint6_inference.py (8 tests)

These tests validate the basic inference functions that preceded the CCGV implementation.

`TestProfileLikelihoodRatio::test_zero_at_bestfit` confirms that the PLR is zero when evaluated at the best-fit point, by definition. `test_increases_away_from_bestfit` and `test_non_negative` verify the shape of the PLR as a function of the parameter of interest.

`TestUpperLimit::test_ul_above_bestfit` checks that the upper limit exceeds the best-fit value (it must, by construction). `test_ul_background_only` verifies that the limit is reasonable for data consistent with background-only. `test_ul_strong_signal` checks that injecting signal shifts the limit upward.

`TestDiscoverySignificance::test_strong_signal_high_z` confirms that a large signal excess produces high significance. `test_no_signal_low_z` verifies that background-consistent data produces low significance. `test_z_p_consistency` checks that the returned Z-value and p-value satisfy p = 1 - Phi(Z).


## test_sprint7_test_statistics.py (30 tests)

These tests validate the CCGV test statistics and their asymptotic distributions. Incorrect test statistics would propagate into every CLs limit.

`TestQ0::test_positive_when_signal_present` confirms q0 > 0 for data with a clear signal excess. `test_zero_when_deficit` confirms q0 = 0 when the data shows fewer events than the background prediction. These two tests together verify the boundary behaviour that distinguishes q0 from the naive two-sided statistic.

`TestQ0::test_non_negative` is a mathematical constraint: q0 cannot be negative by definition.

`TestQ0::test_counting_experiment_closed_form` validates against the exact analytical expression q0 = 2*(n*ln(n/b) - (n-b)). This is the strongest possible test because it compares to an independently derived result.

`TestQ0::test_counting_experiment_deficit` confirms the closed-form result gives q0 = 0 when n < b.

`TestQmu::test_positive_when_mu_test_above_bestfit` and `test_zero_when_mu_test_below_bestfit` verify the boundary behaviour of the exclusion statistic. `test_non_negative` enforces the mathematical constraint. `test_monotonic_above_bestfit` checks that q_mu grows as the tested signal strength moves further from the data. `test_equals_two_sided_in_correct_regime` confirms that q_mu equals the naive PLR when mu_hat < mu_test, using an independent computation.

`TestQtildeMu::test_equals_qmu_when_muhat_positive` confirms qtilde and qmu agree when the physical boundary is not active. `test_zero_when_muhat_above_mu` checks the upper boundary. `test_positive_when_deficit` verifies that qtilde is nonzero in the deficit regime (unlike qmu, which could also be nonzero, qtilde uses a different denominator). `test_deficit_less_than_qmu` confirms the inequality qtilde < qmu when mu_hat < 0, which follows from NLL(0) > NLL(mu_hat). `test_non_negative` enforces the mathematical constraint.

`TestPvalueQ0::test_zero_gives_half` confirms p(q0=0) = 0.5 (the delta function contributes half the probability). `test_large_q0_small_pvalue` checks that large q0 gives small p. `test_moderate_q0` validates against the Gaussian quantile: q0 = 4 gives p = Phi(-2) = 0.0228. `test_consistency_with_z` checks the p-to-Z conversion.

`TestPvaluesQmu::test_zero_qmu` checks the limiting case. `test_known_values` validates against manually computed p-values at specific (qmu, mu, sigma) inputs. `test_large_mu_over_sigma_gives_clb_near_one` confirms that CLb approaches 1 when the tested mu is far above the sensitivity. `test_cls_ratio` verifies that CLs = CLsb / CLb is consistent with the individual p-values.

`TestPvaluesQtildeMu::test_matches_qmu_when_muhat_positive` confirms the two p-value functions agree when the boundary is inactive. `test_muhat_negative_clsb` and `test_muhat_negative_clb_valid` check the negative-mu_hat branch. `test_cls_well_defined` confirms CLs can be computed from the returned values.

`TestQ0EndToEnd::test_z_p_consistency` runs the full chain (model, fit, q0, p-value, Z) and checks that p = 1 - Phi(Z). `test_deficit_gives_half` confirms the full chain gives p = 0.5 for deficit data.

`TestAsimovSignificance::test_moderate_signal`, `test_strong_signal`, `test_weak_signal` validate q0 on Asimov data against the closed-form Z_A = sqrt(2*((s+b)*ln(1+s/b) - s)) from CCGV equation 97 at three different signal/background ratios.


## test_sprint8_cls.py (24 tests)

These tests validate the CLs method, Asimov datasets, and Brazil band computation.

`TestAsimovModel::test_background_only_asimov` confirms that the Asimov data under mu=0 equals the background template exactly. `test_signal_plus_background_asimov` checks that mu=1 Asimov data equals signal + background. `test_scaled_signal_asimov` verifies that mu=2 doubles the signal contribution. These three tests establish that the Asimov dataset is computed correctly, which is the foundation of expected limits. `test_preserves_model_structure` confirms the returned model has the same channels, samples, and parameters. `test_does_not_modify_original` ensures the input model is unchanged (important because model-modification bugs are silent).

`TestAsimovSigma::test_positive` confirms sigma > 0. `test_decreases_with_sensitivity` checks that more sensitive searches (more signal events) give smaller sigma, meaning tighter constraints on mu. `test_counting_experiment` validates against the closed-form sigma for a counting experiment.

`TestClsObserved::test_bounded_zero_to_one` enforces the mathematical constraint on CLs. `test_small_for_large_mu` confirms that a large signal hypothesis is excluded by background-only data. `test_large_for_injected_signal` confirms that data with injected signal is consistent with the signal hypothesis. `test_decreases_with_mu` checks monotonicity.

`TestClsExpectedBand::test_returns_five_values` checks the output shape. `test_ordered` confirms the quantile ordering (-2sig <= -1sig <= median <= +1sig <= +2sig). `test_median_bounded` checks that the median CLs is in [0,1]. `test_median_small_for_large_mu` confirms that a sensitive search expects to exclude large mu.

`TestUpperLimitCls::test_positive` checks that the limit is positive. `test_observed_reasonable` verifies the observed limit falls within a sensible range. `test_expected_bands_ordered` confirms the band ordering. `test_observed_near_expected` checks that, for background-only data, the observed limit falls within the expected band. `test_stronger_search_tighter_limit` compares two search configurations and confirms that more signal events produce a lower limit. `test_signal_injection_shifts_observed` verifies that injecting signal shifts the observed limit away from the expected.

`TestCountingExpectedLimit::test_s10_b100` and `test_s20_b50` validate the median expected limit against the closed-form counting experiment result at two signal/background configurations.


## test_sprint9_modifiers.py (26 tests)

These tests validate the four new modifier types and their interactions with the likelihood.

`TestStatErrorExpected::test_nominal_gammas_unchanged` confirms that gammas at their nominal value (1.0) leave the expected yields unchanged. `test_shifted_gamma_scales_bin` verifies that shifting a single gamma scales only the corresponding bin. `test_creates_correct_number_of_parameters` checks that one parameter is created per bin. `test_parameters_start_at_one` confirms the nominal value.

`TestStatErrorFit::test_gamma_pulled_toward_data` constructs a model where data exceeds the template by 20% in every bin with 10% relative MC uncertainty, and verifies that the fitted gammas lie between 1.0 and 1.2. The gammas should be pulled toward the data but restrained by the Gaussian constraint. `test_tighter_constraint_less_pull` compares 1% vs 50% relative uncertainty and confirms that tighter constraints produce smaller pulls, validating the constraint mechanism. `test_staterror_widens_postfit_uncertainty` adds staterror to a signal+background model and confirms the signal strength uncertainty increases, which is the physical consequence of acknowledging template statistical limitations.

`TestShapeSysExpected::test_nominal_gammas_unchanged` and `test_shifted_gamma` mirror the StatError expected-yield tests for the Poisson-constrained variant.

`TestShapeSysFit::test_poisson_constraint_pulls_toward_data` and `test_small_uncertainty_tight_constraint` mirror the StatError fit tests, verifying that the Poisson constraint (parameterised by tau = 1/rel^2) behaves correctly. The 1% uncertainty case has tau = 10,000, which should produce negligible pull.

`TestShapeFactorExpected::test_nominal_unchanged` confirms that gamma = 1 leaves yields unchanged. `test_parameters_are_unconstrained` inspects the NLL internals to verify that ShapeFactor parameters have no constraint term, which is their defining property.

`TestShapeFactorFit::test_reproduces_data_exactly` constructs arbitrary data and verifies the post-fit yields match exactly (within numerical precision). Because ShapeFactor has one free parameter per bin and no constraint, the fit should achieve perfect data description. `test_recovers_correct_scale_factors` checks that the fitted gammas equal data/nominal in each bin. `test_shapefactor_with_signal` adds a signal sample and verifies that the fit still converges correctly.

`TestLumiSysExpected::test_nominal_unchanged` confirms that lumi = 1 leaves yields unchanged. `test_scales_all_bins_coherently` verifies that a shifted lumi parameter scales all bins by the same factor.

`TestLumiSysFit::test_lumi_absorbs_overall_scale` generates data 3% above nominal with 5% lumi uncertainty and checks that the fit recovers lumi near 1.03. `test_lumi_constrained_when_shift_large` uses a tight 2% constraint with data 20% above nominal and verifies the fitted lumi is restrained (it cannot reach 1.2). `test_correlated_across_samples` attaches lumi to both signal and background and confirms they shift together.

`TestModifierCombinations::test_normfactor_plus_staterror`, `test_normsys_plus_staterror`, and `test_staterror_constraint_in_nll` verify that staterror composes correctly with other modifiers and that its constraint terms appear in the NLL.


## test_sprint10_toys.py (14 tests)

These tests validate the toy-based inference chain.

`TestGenerateToyData::test_correct_shape` confirms the output array has the right number of bins. `test_non_negative` checks that Poisson draws are non-negative. `test_mean_converges_to_expected` generates 5,000 toys and verifies that the sample mean converges to the expected yields (within 5%). This is the law of large numbers applied to the toy generator. `test_reproducible_with_seed` confirms deterministic output with a fixed seed.

`TestSampleNuisanceParameters::test_alpha_params_drawn_from_unit_gaussian` generates 2,000 samples of a NormSys nuisance parameter and checks that the sample mean is near 0 and the sample standard deviation is near 1. `test_poi_not_sampled` confirms that the signal strength is excluded from the sampled parameters, because it is set by the hypothesis rather than drawn from a prior.

`TestToyCls::test_bounded_zero_to_one` enforces the mathematical constraint. `test_agrees_with_asymptotic_high_stats` is the critical cross-validation test: at 500 events/bin, the toy-based CLs (with 2,000 toys) should agree with the asymptotic CLs within ~0.15. This validates both implementations against each other. `test_small_for_large_mu` and `test_large_for_small_mu` check the qualitative behaviour.

`TestToyClsLimit::test_positive_and_finite` checks that the limit is a valid number. `test_agrees_with_asymptotic_high_stats` verifies agreement within a factor of ~2 (accounting for the coarse scan grid and finite toy count).

`TestLowStatsToys::test_cls_valid_at_low_stats` exercises a single-bin model with background = 1.5 and 0 observed events, confirming that toy CLs is in [0,1] even when the asymptotic approximation is unreliable. `test_limit_finite_at_low_stats` checks that a finite limit can be extracted in this regime. These tests justify the existence of the toy calculator: they demonstrate a regime where it is needed.


## test_sprint11_scans.py (11 tests)

These tests validate likelihood scans and goodness-of-fit.

`TestLikelihoodScan1D::test_minimum_at_bestfit` confirms that the scan minimum is near zero and occurs near the best-fit value. `test_parabola_matches_hesse` fits a parabola to the scan points and checks that the inferred uncertainty matches the Hesse error from the fit (within 35%). This cross-validates the scan against the covariance matrix. `test_interval_at_68cl` extracts the 68% interval from the scan and checks consistency with the Hesse error. `test_interval_at_95cl` does the same for 95% and additionally checks that the 95% interval is wider than the 68% interval.

`TestLikelihoodScan2D::test_output_shape` checks the grid dimensions. `test_minimum_near_zero` confirms the grid minimum is near zero. `test_anticorrelation_mu_bkg` constructs a signal+background model with a background normalisation nuisance parameter and verifies negative covariance between mu and bkg_norm. Increasing the background normalisation absorbs events that could otherwise be attributed to signal, so the two parameters must be anti-correlated.

`TestGoodnessOfFit::test_well_specified_passes` generates data from the model at 5 seeds and requires at least 3 out of 5 to give p > 0.05. A well-specified model should usually pass the GoF test; requiring a majority rather than unanimity accounts for statistical fluctuations. `test_misspecified_fails` constructs data with a U-shaped pattern that the peaked signal + flat background model cannot reproduce, and requires p < 0.01. `test_returns_positive_chi2` enforces non-negativity of the test statistic. `test_ndof_correct` confirms ndof = 5 bins - 1 floated parameter = 4. `test_pvalue_bounded` checks 0 <= p <= 1.


## test_sprint12_diagnostics.py (14 tests)

These tests validate the analyst-facing diagnostic outputs.

`TestPrefitPostfitYields::test_prefit_matches_nominal` verifies that pre-fit yields equal model.expected() at nominal parameter values. Any discrepancy would mean the diagnostic is not showing what it claims to show. `test_postfit_near_data` checks that post-fit yields approximate the observed data for a well-specified model. `test_contains_per_sample_breakdown` verifies the output structure includes individual sample contributions, not just totals. `test_prefit_samples_sum_to_total` and `test_postfit_samples_sum_to_total` check that per-sample yields are self-consistent with the total.

`TestNuisanceParameterPulls::test_pulled_np_has_large_pull` constructs a model with 20% data excess and a 20% NormSys, forcing bkg_norm to pull significantly. A pull magnitude above 0.5 is required. `test_pull_direction_correct` verifies the sign (data above nominal produces a positive pull for an upward-shifting nuisance parameter). `test_constraint_ratio_less_than_one` confirms that the post-fit uncertainty is smaller than the pre-fit uncertainty, meaning the data provides information beyond the prior. `test_returns_all_nps` checks completeness. `test_does_not_include_poi` ensures the signal strength is excluded from the pull table, since it has no prior to pull against.

`TestImpacts::test_norm_impact_larger_than_shape` is a physics-motivated ranking test. A NormSys that scales the entire background is more degenerate with the signal strength than a HistoSys that tilts the background shape, so the normalisation systematic should have larger impact. This validates both the impact calculation and the sorting. `test_ranked_by_impact` confirms monotonic sorting. `test_returns_all_nps` checks completeness. `test_does_not_include_poi` ensures the POI is excluded. `test_impact_structure` verifies that each entry contains the expected keys (name, up shift, down shift, maximum absolute impact).