"""Toy-based (pseudo-experiment) calculator for CLs.

When the asymptotic approximation breaks down (low bin counts, very
few expected events), the test statistic distributions must be built
numerically from pseudo-experiments ("toys").  This module provides:

- ``generate_toy_data``: draw Poisson pseudo-data from a model
- ``sample_nuisance_parameters``: sample NPs from their constraints
- ``toy_cls``: compute CLs at a single mu value using toys
- ``toy_cls_limit``: scan mu to find the 95% CL upper limit

The toy procedure is:
1. For each toy under the s+b hypothesis (mu = mu_test):
   a. Sample nuisance parameters from their constraint distributions.
   b. Compute expected yields at (mu_test, sampled NPs).
   c. Draw Poisson pseudo-data from the expected yields.
   d. Compute the test statistic q_tilde_mu on the pseudo-data.

2. Repeat under the b-only hypothesis (mu = 0).

3. CLsb = fraction of s+b toys with q_tilde >= q_obs
   CLb  = fraction of b-only toys with q_tilde >= q_obs
   CLs  = CLsb / CLb

References
----------
.. [CCGV] Cowan, Cranmer, Gross, Vitells, arXiv:1007.1727
.. [CLs] Read, J.Phys. G28 (2002) 2693-2704
"""

from __future__ import annotations

import numpy as np

from histimator.channels import Channel
from histimator.model import Model
from histimator.samples import (
    HistoSys,
    LumiSys,
    NormSys,
    ShapeSys,
    StatError,
)
from histimator.test_statistics import compute_qtilde_mu

# ------------------------------------------------------------------
# Nuisance parameter sampling
# ------------------------------------------------------------------

def sample_nuisance_parameters(
    model: Model,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Sample nuisance parameters from their constraint distributions.

    Only constrained parameters are sampled.  Free parameters
    (NormFactor, ShapeFactor) are excluded because they are set by
    the hypothesis or profiled in the fit.

    Parameters
    ----------
    model : Model
        The statistical model.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    dict[str, float]
        Sampled nuisance parameter values.
    """
    sampled: dict[str, float] = {}
    seen: set[str] = set()

    for ch in model.channels:
        for sample in ch.samples:
            for mod in sample.modifiers:
                if isinstance(mod, (NormSys, HistoSys)):
                    name = mod.parameter.name
                    if name not in seen:
                        seen.add(name)
                        sampled[name] = float(rng.standard_normal())

                elif isinstance(mod, StatError):
                    for p, delta in zip(mod.parameters, mod.rel_uncertainties, strict=True):
                        if p.name not in seen:
                            seen.add(p.name)
                            sampled[p.name] = float(
                                rng.normal(1.0, float(delta))
                            )

                elif isinstance(mod, LumiSys):
                    name = mod.parameter.name
                    if name not in seen:
                        seen.add(name)
                        sampled[name] = float(
                            rng.normal(1.0, mod.uncertainty)
                        )

                elif isinstance(mod, ShapeSys):
                    for p, rel in zip(mod.parameters, mod.rel_uncertainties, strict=True):
                        if p.name not in seen:
                            seen.add(p.name)
                            # Gamma distributed with mean=1, var=rel^2
                            # shape = 1/rel^2, scale = rel^2
                            tau = 1.0 / (float(rel) ** 2)
                            sampled[p.name] = float(
                                rng.gamma(tau, 1.0 / tau)
                            )

                # NormFactor and ShapeFactor are NOT sampled

    return sampled


# ------------------------------------------------------------------
# Toy data generation
# ------------------------------------------------------------------

def generate_toy_data(
    model: Model,
    params: dict[str, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate Poisson pseudo-data from the model.

    Parameters
    ----------
    model : Model
        The statistical model.
    params : dict[str, float]
        Parameter values defining the hypothesis (POI + NPs).
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    numpy.ndarray
        Pseudo-data counts, same shape as model.data.
    """
    expected = model.expected(params)
    # Protect against negative expected yields
    safe_expected = np.maximum(expected, 0.0)
    return rng.poisson(safe_expected).astype(np.float64)


# ------------------------------------------------------------------
# Unbinned toy data generation
# ------------------------------------------------------------------

def _sample_from_density(
    template,
    n_events: int,
    rng: np.random.Generator,
    n_grid: int = 1000,
) -> np.ndarray:
    """Sample events from a template's probability density via rejection.

    Evaluates the density on a fine grid to bound the envelope, then
    uses rejection sampling to draw exact samples from the continuous
    density.  Works for any template type (BinnedTemplate, GPTemplate,
    BSplineTemplate) without special-casing.

    Parameters
    ----------
    template : Template
        Any object satisfying the Template protocol.
    n_events : int
        Number of events to sample.
    rng : numpy.random.Generator
        Random number generator.
    n_grid : int
        Grid resolution for bounding the density envelope.

    Returns
    -------
    numpy.ndarray
        1-D array of sampled event positions.
    """
    if n_events <= 0:
        return np.empty(0, dtype=np.float64)

    lo = float(template.edges[0])
    hi = float(template.edges[-1])

    # Build envelope: evaluate density on a fine grid, take max
    # with a safety margin for peaks between grid points.
    x_grid = np.linspace(lo, hi, n_grid)
    d_grid = template.evaluate_density(x_grid)
    max_density = float(d_grid.max()) * 1.1
    if max_density <= 0:
        return rng.uniform(lo, hi, n_events)

    # Rejection sampling in batches for efficiency.
    # Expected acceptance rate is mean_density / max_density,
    # so oversample by the inverse of that.
    mean_density = float(d_grid.mean())
    oversample = max(2.0, max_density / mean_density) if mean_density > 0 else 10.0

    accepted = []
    n_remaining = n_events
    while n_remaining > 0:
        batch_size = int(n_remaining * oversample) + 100
        x_proposal = rng.uniform(lo, hi, batch_size)
        u = rng.uniform(0.0, max_density, batch_size)
        densities = template.evaluate_density(x_proposal)
        mask = u < densities
        accepted.append(x_proposal[mask])
        n_remaining -= int(mask.sum())

    return np.concatenate(accepted)[:n_events]


def generate_unbinned_toy_data(
    model: Model,
    params: dict[str, float],
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Generate unbinned pseudo-data (event positions) from the model.

    For each channel, each sample contributes Poisson(N_j) events
    drawn from that sample's template density, where N_j is the total
    expected yield of sample j at the given parameter values.

    Parameters
    ----------
    model : Model
        The statistical model.
    params : dict[str, float]
        Parameter values defining the hypothesis (POI + NPs).
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    list[numpy.ndarray]
        One 1-D array of event positions per channel.
    """
    channel_events = []
    for ch in model.channels:
        events_parts = []
        for sample in ch.samples:
            total_yield = float(sample.expected(params).sum())
            n_j = rng.poisson(max(total_yield, 0.0))
            if n_j > 0:
                evts = _sample_from_density(sample.template, n_j, rng)
                events_parts.append(evts)
        if events_parts:
            channel_events.append(np.concatenate(events_parts))
        else:
            channel_events.append(np.empty(0, dtype=np.float64))
    return channel_events


def _build_toy_model(
    model: Model,
    toy_data: np.ndarray,
) -> Model:
    """Create a copy of the model with toy data substituted.

    Reuses the same samples (and therefore the same parameters and
    modifiers) but replaces the observed data in each channel.
    """
    toy_model = Model(f"{model.name}_toy")
    offset = 0
    for ch in model.channels:
        new_ch = Channel(ch.name)
        for sample in ch.samples:
            new_ch.add_sample(sample)
        nbins = ch.nbins
        new_ch.set_data(toy_data[offset:offset + nbins])
        offset += nbins
        toy_model.add_channel(new_ch)
    return toy_model


def _build_unbinned_toy_model(
    model: Model,
    toy_events: list[np.ndarray],
) -> Model:
    """Create a copy of the model with unbinned toy events substituted.

    Reuses the same samples (and therefore the same parameters and
    modifiers) but replaces the observed data in each channel with
    both unbinned events and the corresponding binned histogram.

    Parameters
    ----------
    model : Model
        The original model.
    toy_events : list[numpy.ndarray]
        One array of event positions per channel, as returned by
        ``generate_unbinned_toy_data``.
    """
    toy_model = Model(f"{model.name}_toy")
    for ch, events in zip(model.channels, toy_events, strict=True):
        new_ch = Channel(ch.name)
        for sample in ch.samples:
            new_ch.add_sample(sample)
        # Set binned data (needed by Model.data and binned diagnostics)
        edges = ch.edges
        binned, _ = np.histogram(events, bins=edges)
        new_ch.set_data(binned.astype(np.float64))
        # Set unbinned data
        new_ch.set_unbinned_data(events)
        toy_model.add_channel(new_ch)
    return toy_model


# ------------------------------------------------------------------
# Toy-based CLs at a single mu value
# ------------------------------------------------------------------

def toy_cls(
    model: Model,
    poi_name: str,
    mu_test: float,
    n_toys: int = 1000,
    seed: int = 42,
    unbinned: bool = False,
) -> float:
    """Compute CLs at a single mu value using toy pseudo-experiments.

    Parameters
    ----------
    model : Model
        Model with observed data.
    poi_name : str
        Name of the signal-strength parameter.
    mu_test : float
        The signal strength to test.
    n_toys : int
        Number of pseudo-experiments per hypothesis.
    seed : int
        Random seed for reproducibility.
    unbinned : bool
        If True, generate unbinned toy events and use the unbinned
        extended likelihood for test statistic evaluation.

    Returns
    -------
    float
        CLs = CLsb / CLb.
    """
    # Observed test statistic
    q_obs, _ = compute_qtilde_mu(model, poi_name, mu_test, unbinned=unbinned)

    rng = np.random.default_rng(seed)

    # Generate toys under s+b hypothesis (mu = mu_test)
    q_sb = _generate_test_statistics(
        model, poi_name, mu_test, mu_test, n_toys, rng,
        unbinned=unbinned,
    )

    # Generate toys under b-only hypothesis (mu = 0)
    q_b = _generate_test_statistics(
        model, poi_name, mu_test, 0.0, n_toys, rng,
        unbinned=unbinned,
    )

    # CLsb = P(q >= q_obs | s+b)
    clsb = np.mean(q_sb >= q_obs)
    # CLb = P(q >= q_obs | b-only)
    clb = np.mean(q_b >= q_obs)

    if clb <= 0:
        return 0.0
    return float(clsb / clb)


def _generate_test_statistics(
    model: Model,
    poi_name: str,
    mu_test: float,
    mu_hypothesis: float,
    n_toys: int,
    rng: np.random.Generator,
    unbinned: bool = False,
) -> np.ndarray:
    """Generate test statistic distribution under a given hypothesis.

    For each toy:
    1. Sample nuisance parameters from their constraints.
    2. Compute expected yields at (mu_hypothesis, sampled NPs).
    3. Draw pseudo-data (binned counts or unbinned events).
    4. Compute q_tilde_mu(mu_test) on the pseudo-data.
    """
    q_values = np.zeros(n_toys)

    for i in range(n_toys):
        # Build parameter set for this hypothesis
        np_sample = sample_nuisance_parameters(model, rng)
        params = dict(np_sample)
        params[poi_name] = mu_hypothesis

        # Generate pseudo-data and build toy model
        if unbinned:
            toy_events = generate_unbinned_toy_data(model, params, rng)
            toy_model = _build_unbinned_toy_model(model, toy_events)
        else:
            toy_data = generate_toy_data(model, params, rng)
            toy_model = _build_toy_model(model, toy_data)

        # Compute test statistic
        try:
            q_val, _ = compute_qtilde_mu(
                toy_model, poi_name, mu_test, unbinned=unbinned,
            )
            q_values[i] = q_val
        except Exception:
            # If fit fails, assign q = 0 (conservative)
            q_values[i] = 0.0

    return q_values


# ------------------------------------------------------------------
# Toy-based upper limit
# ------------------------------------------------------------------

def toy_cls_limit(
    model: Model,
    poi_name: str,
    cl: float = 0.95,
    n_toys: int = 500,
    seed: int = 42,
    scan_points: int = 10,
    scan_range: tuple[float, float] | None = None,
    unbinned: bool = False,
) -> float:
    """Find the 95% CL upper limit using toy-based CLs.

    Scans over mu values, computes toy CLs at each point, then
    interpolates to find where CLs = 1 - cl (default 0.05).

    Parameters
    ----------
    model : Model
        Model with observed data.
    poi_name : str
        Name of the signal-strength parameter.
    cl : float
        Confidence level (default 0.95).
    n_toys : int
        Number of toys per mu scan point.
    seed : int
        Random seed.
    scan_points : int
        Number of mu values to scan.
    scan_range : tuple or None
        (lo, hi) for the mu scan.  Auto-determined if None.
    unbinned : bool
        If True, generate unbinned toy events and use the unbinned
        extended likelihood for test statistic evaluation.

    Returns
    -------
    float
        The observed upper limit.
    """
    alpha = 1.0 - cl

    if scan_range is None:
        lo = 0.1
        hi = 10.0
        for p in model.parameters:
            if p.name == poi_name and p.bounds is not None:
                hi = min(p.bounds[1], 30.0)
    else:
        lo, hi = scan_range

    mu_values = np.linspace(lo, hi, scan_points)
    cls_values = np.zeros(scan_points)

    for i, mu in enumerate(mu_values):
        cls_values[i] = toy_cls(
            model, poi_name, mu, n_toys=n_toys, seed=seed + i,
            unbinned=unbinned,
        )

    # Find the crossing point where CLs = alpha
    # If all CLs > alpha, return the largest mu scanned
    if np.all(cls_values > alpha):
        return float(mu_values[-1])

    # If all CLs < alpha, return the smallest mu scanned
    if np.all(cls_values < alpha):
        return float(mu_values[0])

    # Linear interpolation to find crossing
    # CLs is roughly monotonically decreasing in mu, so find the
    # first crossing from above alpha to below
    for j in range(len(cls_values) - 1):
        if cls_values[j] >= alpha and cls_values[j + 1] < alpha:
            # Linear interpolation between j and j+1
            frac = (cls_values[j] - alpha) / (cls_values[j] - cls_values[j + 1])
            return float(mu_values[j] + frac * (mu_values[j + 1] - mu_values[j]))

    # Fallback: return last mu where CLs > alpha
    above = np.where(cls_values >= alpha)[0]
    if len(above) > 0:
        return float(mu_values[above[-1]])
    return float(mu_values[-1])
