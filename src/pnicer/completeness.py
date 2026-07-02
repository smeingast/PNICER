"""Photometric completeness models fitted to number counts.

Following Lombardi (2018, Sect. 2.6): the completeness of each band is
modeled as c(m) = 1/2 erfc((m - m50) / (sqrt(2) s)), fitted together with
exponential number counts N(m) = N0 * 10^(alpha m) * c(m) to the observed
magnitude histogram by Poisson maximum likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import erfc

__all__ = ["BandCompleteness", "CompletenessModel"]


@dataclass(frozen=True)
class BandCompleteness:
    """Completeness of one band: c(m) = 1/2 erfc((m - m50)/(sqrt(2) s))."""

    m50: float
    width: float
    alpha: float
    log_norm: float

    def __call__(self, magnitudes: np.ndarray) -> np.ndarray:
        return 0.5 * erfc((magnitudes - self.m50) / (np.sqrt(2.0) * self.width))


def _fit_band(magnitudes: np.ndarray, bin_width: float) -> BandCompleteness:
    """Fit (N0, alpha, m50, s) to one band's magnitude histogram."""
    mags = magnitudes[np.isfinite(magnitudes)]
    if mags.size < 100:
        raise ValueError(f"Too few measurements ({mags.size}) to fit completeness")

    edges = np.arange(mags.min(), mags.max() + bin_width, bin_width)
    if edges.size < 4:
        raise ValueError(
            f"Magnitude range ({mags.min():.2f}..{mags.max():.2f}) too small "
            "to fit a completeness function"
        )
    counts, edges = np.histogram(mags, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Initial guesses: peak bin approximates the completeness limit; the
    # count slope is measured over the rising part of the histogram.
    peak = int(np.argmax(counts))
    m50_init = centers[peak]
    bright = (centers < centers[peak] - 1.0) & (counts > 0)
    if bright.sum() >= 3:
        alpha_init = np.polyfit(centers[bright], np.log10(counts[bright]), 1)[0]
    else:
        alpha_init = 0.33
    alpha_init = float(np.clip(alpha_init, 0.05, 1.0))
    log_norm_init = float(
        np.log(counts[peak] + 1.0) - alpha_init * np.log(10.0) * m50_init
    )

    def negloglike(params: np.ndarray) -> float:
        log_norm, alpha, m50, log_width = params
        width = np.exp(log_width)
        completeness = 0.5 * erfc((centers - m50) / (np.sqrt(2.0) * width))
        log_mu = (
            log_norm + alpha * np.log(10.0) * centers + np.log(completeness + 1e-300)
        )
        mu = np.exp(log_mu)
        return float(np.sum(mu - counts * log_mu))

    start = np.array([log_norm_init, alpha_init, m50_init, np.log(0.3)])
    result = minimize(
        negloglike,
        start,
        method="Nelder-Mead",
        options={"maxiter": 4000, "xatol": 1e-5, "fatol": 1e-8},
    )
    log_norm, alpha, m50, log_width = result.x
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError("Completeness fit did not converge")
    return BandCompleteness(
        m50=float(m50),
        width=float(np.exp(log_width)),
        alpha=float(alpha),
        log_norm=float(log_norm),
    )


@dataclass(frozen=True)
class CompletenessModel:
    """Per-band completeness functions of a photometric catalog."""

    band_names: tuple[str, ...]
    bands: tuple[BandCompleteness, ...]

    @classmethod
    def fit(
        cls,
        magnitudes: np.ndarray,
        band_names: tuple[str, ...],
        bin_width: float = 0.25,
    ) -> CompletenessModel:
        """Fit completeness to each column of a (N, n_bands) magnitude array."""
        fits = tuple(
            _fit_band(magnitudes[:, i], bin_width) for i in range(len(band_names))
        )
        return cls(band_names=band_names, bands=fits)

    @classmethod
    def from_parameters(
        cls,
        band_names: tuple[str, ...],
        m50: np.ndarray,
        width: np.ndarray,
    ) -> CompletenessModel:
        """Build a model from known 50% limits and widths (no fitting)."""
        bands = tuple(
            BandCompleteness(
                m50=float(m), width=float(w), alpha=np.nan, log_norm=np.nan
            )
            for m, w in zip(m50, width, strict=True)
        )
        return cls(band_names=band_names, bands=bands)

    def survival(
        self,
        magnitudes: np.ndarray,
        observed: np.ndarray,
        band_extinction: np.ndarray,
        a_grid: np.ndarray,
        floor: float = 0.01,
    ) -> np.ndarray:
        """Relative detection probability of each source under extinction.

        For source i at extinction A: s_i(A) = prod over its observed bands b
        of c_b(m_ib + A k_b) / max(c_b(m_ib), floor), capped at 1. The
        denominator undoes the decimation already present in the
        (extinction-free) catalog; the floor keeps sources at the detection
        limit from dominating. The cap restricts the model to object *loss*:
        brightening (negative extinction) does not add objects, so the
        population weights stay at their zero-extinction values for A <= 0.

        Parameters
        ----------
        magnitudes : ndarray, shape (N, n_bands)
        observed : ndarray of bool, shape (N, n_bands)
        band_extinction : ndarray, shape (n_bands,)
            Extinction-law coefficients A_band / A_ref.
        a_grid : ndarray, shape (G,)
            Extinction values (in units of A_ref) to evaluate.
        floor : float
            Lower bound on the denominator completeness.

        Returns
        -------
        ndarray, shape (N, G)
        """
        n_sources = magnitudes.shape[0]
        log_s = np.zeros((n_sources, a_grid.size))
        for b, comp in enumerate(self.bands):
            has = observed[:, b]
            mags = magnitudes[has, b]
            denom = np.maximum(comp(mags), floor)
            shifted = mags[:, None] + a_grid[None, :] * band_extinction[b]
            numer = comp(shifted)
            log_s[has] += np.log(numer + 1e-300) - np.log(denom)[:, None]
        return np.minimum(np.exp(log_s), 1.0)
