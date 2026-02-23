"""Eigenmode constraint: unified treatment of statistical and systematic
template uncertainty through a single eigendecomposition.

The standard HistFactory approach treats MC statistical uncertainty
(staterror / Barlow--Beeston) and systematic shape variations (histosys /
interpolation) as separate modifiers with independent nuisance parameters.
The eigenmode constraint unifies both into a single covariance matrix
whose eigendecomposition provides a compact set of constrained shape modes.

The combined covariance is constructed as:

    Sigma_comb = Sigma_stat + sum_k  delta_k  delta_k^T

where Sigma_stat = (K^{-1} + W)^{-1} is the GP posterior covariance
capturing MC statistical uncertainty, and delta_k = (f_+^k - f_-^k) / 2
is the log-rate shape direction for systematic variation k, estimated from
GP fits at the +/- 1 sigma variation points.

The eigendecomposition of Sigma_comb yields modes that span both
statistical and systematic shape variations.  Each retained mode gets a
unit-Gaussian nuisance parameter z_i, and the expected count is:

    nu_j = mu * s_j + b_j^GP * exp( sum_i sqrt(lambda_i) * z_i * v_{ij} )

This replaces both the N per-bin gamma factors of Barlow--Beeston and
the per-systematic interpolation parameters of histosys with a single
set of k << N eigenmode amplitudes.

References:
    Barlow & Beeston, Comp. Phys. Comm. 77 (1993) 219
    Conway, arXiv:1103.0354 (2011)
    Cranmer et al., CERN-OPEN-2012-016 (2012)
    Cai, Chow & Xi, SIAM J. Matrix Anal. Appl. (2025)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from histimator.gp_template import GPTemplate


@dataclass(frozen=True)
class SystematicDirection:
    """A single systematic shape variation estimated from GP fits.

    Parameters
    ----------
    name : str
        Human-readable label for this systematic (e.g. "JES", "btagSF").
    delta : np.ndarray
        Shape direction in log-rate space: (f_hat_up - f_hat_down) / 2.
        Length must equal the number of bins in the nominal template.
    gp_up : GPTemplate
        GP fit to the +1 sigma variation template.
    gp_down : GPTemplate
        GP fit to the -1 sigma variation template.
    """
    name: str
    delta: np.ndarray
    gp_up: GPTemplate
    gp_down: GPTemplate


def make_systematic_direction(
    name: str,
    gp_nominal: GPTemplate,
    gp_up: GPTemplate,
    gp_down: GPTemplate,
) -> SystematicDirection:
    """Construct a systematic direction from GP fits at nominal and
    +/- 1 sigma variation points.

    The direction is computed in log-rate space as the half-difference
    of the posterior modes:
        delta_j = (f_hat_up_j - f_hat_down_j) / 2

    Parameters
    ----------
    name : str
        Label for this systematic source.
    gp_nominal : GPTemplate
        GP fit to the nominal template.
    gp_up, gp_down : GPTemplate
        GP fits to the +1 sigma and -1 sigma variation templates.

    Returns
    -------
    SystematicDirection
        Frozen dataclass holding the direction vector and GP fits.

    Raises
    ------
    ValueError
        If the templates have inconsistent bin counts.
    """
    if not (gp_nominal.nbins == gp_up.nbins == gp_down.nbins):
        raise ValueError(
            f"Bin count mismatch: nominal={gp_nominal.nbins}, "
            f"up={gp_up.nbins}, down={gp_down.nbins}"
        )
    delta = (gp_up.log_rate - gp_down.log_rate) / 2.0
    return SystematicDirection(
        name=name, delta=delta, gp_up=gp_up, gp_down=gp_down,
    )


class EigenmodeConstraint:
    """Combined eigenmode constraint for statistical and systematic
    template uncertainty.

    This is the core object that replaces both the Barlow--Beeston
    staterror modifier and the histosys interpolation modifier with
    a single eigendecomposition.

    Parameters
    ----------
    gp_nominal : GPTemplate
        GP fit to the nominal MC template.  Its posterior covariance
        provides the statistical component of the combined covariance.
    systematics : list of SystematicDirection, optional
        Systematic shape variations to include.  Each adds a rank-1
        update to the combined covariance.
    variance_threshold : float
        Fraction of total variance to retain (default 0.99).
    min_modes : int
        Minimum number of modes to retain regardless of threshold.
    max_modes : int or None
        Maximum number of modes (None = no limit).
    """

    def __init__(
        self,
        gp_nominal: GPTemplate,
        systematics: list[SystematicDirection] | None = None,
        variance_threshold: float = 0.99,
        min_modes: int = 1,
        max_modes: int | None = None,
    ) -> None:
        self._gp_nominal = gp_nominal
        self._systematics = list(systematics or [])
        self._nbins = gp_nominal.nbins
        self._variance_threshold = variance_threshold

        for syst in self._systematics:
            if len(syst.delta) != self._nbins:
                raise ValueError(
                    f"Systematic '{syst.name}' has {len(syst.delta)} bins, "
                    f"expected {self._nbins}"
                )

        # Build combined covariance
        sigma_stat = gp_nominal.posterior_covariance
        sigma_sys = np.zeros_like(sigma_stat)
        for syst in self._systematics:
            d = syst.delta
            sigma_sys += np.outer(d, d)

        self._sigma_stat = sigma_stat
        self._sigma_sys = sigma_sys
        self._sigma_combined = sigma_stat + sigma_sys

        # Symmetrise (numerical hygiene)
        self._sigma_combined = 0.5 * (
            self._sigma_combined + self._sigma_combined.T
        )

        # Eigendecompose
        eigenvalues, eigenvectors = np.linalg.eigh(self._sigma_combined)
        # eigh returns ascending order; we want descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Clip any tiny negative eigenvalues from numerical noise
        eigenvalues = np.maximum(eigenvalues, 0.0)

        self._eigenvalues_full = eigenvalues
        self._eigenvectors_full = eigenvectors

        # Determine truncation point
        total_var = np.sum(eigenvalues)
        if total_var > 0:
            cumvar = np.cumsum(eigenvalues) / total_var
            k = int(np.searchsorted(cumvar, variance_threshold)) + 1
        else:
            k = 1
        k = max(k, min_modes)
        if max_modes is not None:
            k = min(k, max_modes)
        k = min(k, self._nbins)  # can't exceed number of bins
        self._n_modes = k

        self._eigenvalues = eigenvalues[:k]
        self._eigenvectors = eigenvectors[:, :k]
        self._sqrt_eigenvalues = np.sqrt(self._eigenvalues)

    @property
    def n_bins(self) -> int:
        """Number of template bins."""
        return self._nbins

    @property
    def n_modes(self) -> int:
        """Number of retained eigenmodes."""
        return self._n_modes

    @property
    def n_systematics(self) -> int:
        """Number of systematic sources included."""
        return len(self._systematics)

    @property
    def eigenvalues(self) -> np.ndarray:
        """Retained eigenvalues (descending order)."""
        return self._eigenvalues.copy()

    @property
    def eigenvectors(self) -> np.ndarray:
        """Retained eigenvectors as columns, shape (n_bins, n_modes)."""
        return self._eigenvectors.copy()

    @property
    def sqrt_eigenvalues(self) -> np.ndarray:
        """Square roots of retained eigenvalues."""
        return self._sqrt_eigenvalues.copy()

    @property
    def eigenvalues_full(self) -> np.ndarray:
        """Full eigenvalue spectrum before truncation."""
        return self._eigenvalues_full.copy()

    @property
    def variance_captured(self) -> float:
        """Fraction of total variance captured by retained modes."""
        total = np.sum(self._eigenvalues_full)
        if total <= 0:
            return 1.0
        return float(np.sum(self._eigenvalues) / total)

    @property
    def sigma_stat(self) -> np.ndarray:
        """Statistical component of the combined covariance."""
        return self._sigma_stat.copy()

    @property
    def sigma_sys(self) -> np.ndarray:
        """Systematic component of the combined covariance."""
        return self._sigma_sys.copy()

    @property
    def sigma_combined(self) -> np.ndarray:
        """Full combined covariance matrix."""
        return self._sigma_combined.copy()

    @property
    def nominal_counts(self) -> np.ndarray:
        """Smoothed nominal template counts."""
        return self._gp_nominal.counts()

    @property
    def systematic_names(self) -> list[str]:
        """Names of included systematic sources."""
        return [s.name for s in self._systematics]

    def deformation(self, z: np.ndarray) -> np.ndarray:
        """Compute the multiplicative deformation of the template for
        given eigenmode amplitudes.

        Parameters
        ----------
        z : (n_modes,) array
            Eigenmode amplitudes.  Under the constraint, z_i ~ N(0,1).

        Returns
        -------
        factor : (n_bins,) array
            Multiplicative factor exp(sum_i sqrt(lambda_i) z_i v_ij).
        """
        z = np.asarray(z, dtype=np.float64)
        if z.shape != (self._n_modes,):
            raise ValueError(
                f"Expected z of shape ({self._n_modes},), got {z.shape}"
            )
        log_shift = self._eigenvectors @ (self._sqrt_eigenvalues * z)
        return np.exp(log_shift)

    def expected_background(
        self, z: np.ndarray, scale: float = 1.0,
    ) -> np.ndarray:
        """Compute expected background counts for given eigenmode amplitudes.

        Parameters
        ----------
        z : (n_modes,) array
            Eigenmode amplitudes.
        scale : float
            Optional overall scale factor.

        Returns
        -------
        expected : (n_bins,) array
            Expected background counts: scale * b_j^GP * deformation_j(z).
        """
        return scale * self.nominal_counts * self.deformation(z)

    def constraint_nll(self, z: np.ndarray) -> float:
        """Negative log-likelihood of the eigenmode constraint terms.

        This is simply 0.5 * sum(z_i^2), the unit-Gaussian prior on
        each eigenmode amplitude.
        """
        return 0.5 * float(np.sum(np.asarray(z) ** 2))

    def bb_variance(self) -> np.ndarray:
        """Barlow--Beeston per-bin variance for comparison.

        Returns 1/a_j where a_j are the raw MC counts in the nominal
        template histogram.
        """
        raw = self._gp_nominal.histogram.values
        return 1.0 / np.maximum(raw, 0.5)

    def __repr__(self) -> str:
        syst_str = ", ".join(self.systematic_names) or "none"
        return (
            f"EigenmodeConstraint(n_bins={self._nbins}, "
            f"n_modes={self._n_modes}, "
            f"variance_captured={self.variance_captured:.3f}, "
            f"systematics=[{syst_str}])"
        )
