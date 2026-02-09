"""Fit diagnostics: pulls, impacts, and pre/post-fit yields.

These tools transform raw fit results into the standard diagnostic
outputs that every LHC analysis produces:

- ``prefit_postfit_yields``: per-channel, per-sample yields at nominal
  and best-fit parameter values (the data behind pre/post-fit plots).
- ``nuisance_parameter_pulls``: how far each NP moved from its nominal
  in units of the pre-fit uncertainty, plus the post-fit constraint
  (the data behind pull plots).
- ``impacts``: the shift in the POI when each NP is fixed at ±1 sigma,
  ranked by magnitude (the data behind impact plots).

References
----------
.. [Combine] CMS FitDiagnostics method.
"""

from __future__ import annotations

import numpy as np
from iminuit import Minuit

from histimator.likelihood import BinnedNLL, fit
from histimator.model import Model
from histimator.samples import HistoSys, NormSys

# ------------------------------------------------------------------
# Pre-fit and post-fit yields
# ------------------------------------------------------------------

def prefit_postfit_yields(
    model: Model,
    extended: bool = True,
) -> dict:
    """Compute per-channel, per-sample pre-fit and post-fit yields.

    Pre-fit yields use nominal parameter values.  Post-fit yields use
    the best-fit parameter values from a fit to the observed data.

    Parameters
    ----------
    model : Model
        Fully constructed model with data.
    extended : bool
        Use extended likelihood for the fit.

    Returns
    -------
    dict
        Nested dictionary: {channel_name: {prefit: ..., postfit: ..., data: ...}}.
        Each prefit/postfit entry contains 'total' (array) and 'samples' (dict
        of sample_name -> array).
    """
    # Get nominal and best-fit parameters
    nominal_params = model.nominal_values()
    result = fit(model, extended=extended)
    bestfit_params = result.bestfit

    output = {}
    for ch in model.channels:
        ch_result = {
            "data": ch.data.values.copy(),
            "prefit": _evaluate_channel(ch, nominal_params),
            "postfit": _evaluate_channel(ch, bestfit_params),
        }
        output[ch.name] = ch_result

    return output


def _evaluate_channel(channel, params: dict[str, float]) -> dict:
    """Evaluate per-sample and total yields for a channel."""
    samples = {}
    total = np.zeros(channel.nbins, dtype=np.float64)
    for sample in channel.samples:
        yields = sample.expected(params)
        samples[sample.name] = yields
        total += yields
    return {"total": total, "samples": samples}


# ------------------------------------------------------------------
# Nuisance parameter pulls
# ------------------------------------------------------------------

def nuisance_parameter_pulls(
    model: Model,
    extended: bool = True,
) -> dict[str, dict[str, float]]:
    """Compute pulls and constraints for all nuisance parameters.

    A pull measures how far the post-fit NP value is from its nominal,
    in units of the pre-fit uncertainty:

        pull = (theta_hat - theta_0) / sigma_prefit

    The constraint ratio measures how much the data constrains the NP
    beyond its prior:

        constraint = sigma_postfit / sigma_prefit

    Parameters
    ----------
    model : Model
        Fully constructed model with data.
    extended : bool
        Use extended likelihood.

    Returns
    -------
    dict[str, dict]
        {np_name: {'pull': float, 'constraint': float,
                    'bestfit': float, 'error': float}}.
        Only constrained NPs are included (not POI or free params).
    """
    result = fit(model, extended=extended)

    # Identify constrained parameters (NormSys, HistoSys alphas)
    constrained_names = set()
    for ch in model.channels:
        for sample in ch.samples:
            for mod in sample.modifiers:
                if isinstance(mod, (NormSys, HistoSys)):
                    constrained_names.add(mod.parameter.name)

    output = {}
    for name in constrained_names:
        theta_hat = result.bestfit[name]
        sigma_postfit = result.errors[name]

        # For N(0,1) constrained parameters:
        # nominal = 0, pre-fit sigma = 1
        theta_0 = 0.0
        sigma_prefit = 1.0

        pull = (theta_hat - theta_0) / sigma_prefit
        constraint = sigma_postfit / sigma_prefit

        output[name] = {
            "pull": pull,
            "constraint": constraint,
            "bestfit": theta_hat,
            "error": sigma_postfit,
        }

    return output


# ------------------------------------------------------------------
# Impact ranking
# ------------------------------------------------------------------

def impacts(
    model: Model,
    poi_name: str,
    extended: bool = True,
) -> list[dict]:
    """Compute the impact of each nuisance parameter on the POI.

    For each NP, the impact is computed by:
    1. Fixing the NP at theta_hat + sigma_postfit ("up" variation).
    2. Refitting with the NP fixed, obtaining mu_up.
    3. Repeating at theta_hat - sigma_postfit ("down" variation).
    4. impact_up = mu_up - mu_hat, impact_down = mu_down - mu_hat.

    The result is sorted by decreasing maximum absolute impact.

    Parameters
    ----------
    model : Model
        Fully constructed model with data.
    poi_name : str
        Name of the parameter of interest.
    extended : bool
        Use extended likelihood.

    Returns
    -------
    list[dict]
        Sorted list of {name, impact_up, impact_down, bestfit, error}.
    """
    # Nominal fit
    result = fit(model, extended=extended)
    mu_hat = result.bestfit[poi_name]

    # Identify constrained NPs
    constrained_names = set()
    for ch in model.channels:
        for sample in ch.samples:
            for mod in sample.modifiers:
                if isinstance(mod, (NormSys, HistoSys)):
                    constrained_names.add(mod.parameter.name)

    impact_list = []
    for np_name in constrained_names:
        theta_hat = result.bestfit[np_name]
        sigma = result.errors[np_name]

        # Up variation: fix NP at theta_hat + sigma, refit
        mu_up = _refit_with_fixed(
            model, poi_name, np_name, theta_hat + sigma, extended,
        )
        # Down variation: fix NP at theta_hat - sigma, refit
        mu_down = _refit_with_fixed(
            model, poi_name, np_name, theta_hat - sigma, extended,
        )

        impact_list.append({
            "name": np_name,
            "impact_up": mu_up - mu_hat,
            "impact_down": mu_down - mu_hat,
            "bestfit": theta_hat,
            "error": sigma,
        })

    # Sort by decreasing max absolute impact
    impact_list.sort(
        key=lambda e: max(abs(e["impact_up"]), abs(e["impact_down"])),
        reverse=True,
    )

    return impact_list


def _refit_with_fixed(
    model: Model,
    poi_name: str,
    fixed_np_name: str,
    fixed_np_value: float,
    extended: bool = True,
) -> float:
    """Refit the model with one NP fixed, return the POI best-fit value."""
    nll = BinnedNLL(model, extended=extended)
    par_names = nll._par_names
    start = [p.value for p in model.parameters]

    m = Minuit(nll, start, name=par_names)
    m.errordef = 1

    for p in model.parameters:
        if p.bounds is not None:
            m.limits[p.name] = p.bounds
        if p.fixed:
            m.fixed[p.name] = True

    # Fix the specified NP
    m.values[fixed_np_name] = fixed_np_value
    m.fixed[fixed_np_name] = True

    m.migrad()
    return float(m.values[poi_name])
