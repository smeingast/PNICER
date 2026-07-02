"""Per-source extinction posteriors (Lombardi 2018, Sect. 2.3-2.6).

Given a Gaussian mixture model of the intrinsic colors and a source's
observed colors with error covariance, the extinction posterior is an
analytic one-dimensional Gaussian mixture: component k contributes a Gaussian
with mean A_k and variance sigma_k^2 and log-amplitude ln f_k (Eqs. 17-28).
The math is evaluated in log space, vectorized over sources grouped by
missingness pattern, and chunked to bound memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import logsumexp

from pnicer.linalg import batched_cholesky, solve_lower

if TYPE_CHECKING:
    from astropy.coordinates import SkyCoord

    from pnicer.catalog import ExtinctionCatalog
    from pnicer.photometry import _ColorSpaceBase

__all__ = ["ExtinctionPosterior", "PosteriorTerms", "component_terms"]

_LOG_2PI = float(np.log(2.0 * np.pi))
_CHUNK = 1 << 18


@dataclass
class PosteriorTerms:
    """Source-by-component geometry of the extinction posterior.

    For source i and mixture component k (Lombardi 2018, Eqs. 17-26):
    ``a_mean`` holds A_ik, ``a_var`` holds sigma_ik^2, ``log_z`` holds
    ln Z(W_ik) and ``c_quad`` holds C_ik. These depend only on the component
    means/covariances, not on the component weights, so posterior weights can
    be swapped cheaply (adaptive iterations, grid evaluation).
    """

    a_mean: np.ndarray  # (N, K)
    a_var: np.ndarray  # (N, K)
    log_z: np.ndarray  # (N, K)
    c_quad: np.ndarray  # (N, K)
    pattern_dim: np.ndarray  # (N,)

    def log_amplitudes(self, log_weights: np.ndarray) -> np.ndarray:
        """ln f_ik for given (per-source or global) component log-weights."""
        return (
            log_weights
            + self.log_z
            + 0.5 * (_LOG_2PI + np.log(self.a_var))
            - 0.5 * self.c_quad
        )

    def log_pdf_grid(
        self, a_grid: np.ndarray, log_weights: np.ndarray
    ) -> np.ndarray:
        """Unnormalized log posterior on an extinction grid, shape (N, G).

        ``log_weights`` may be (K,), (N, K) for per-source weights, or
        (G, K) for extinction-dependent weights inside the likelihood.
        """
        # (N, K, G) quadratic term
        dev = a_grid[None, None, :] - self.a_mean[..., None]
        log_kernel = (
            self.log_z[..., None]
            - 0.5 * self.c_quad[..., None]
            - 0.5 * dev**2 / self.a_var[..., None]
        )
        if log_weights.ndim == 2 and log_weights.shape[0] == a_grid.size:
            log_kernel = log_kernel + log_weights.T[None, :, :]
        else:
            weights = np.atleast_2d(log_weights)
            log_kernel = log_kernel + weights[..., None]
        return logsumexp(log_kernel, axis=1)


def component_terms(
    science: _ColorSpaceBase,
    means: np.ndarray,
    covariances: np.ndarray,
    min_dim: int = 1,
) -> PosteriorTerms:
    """Compute the per-source, per-component posterior geometry.

    Parameters
    ----------
    science : Photometry or Colors
    means : ndarray, shape (K, D)
    covariances : ndarray, shape (K, D, D)
    min_dim : int
        Minimum observed color-space dimensionality per source.
    """
    n_sources = science.n_sources
    n_components = means.shape[0]
    k_full = science.reddening_vector

    a_mean = np.full((n_sources, n_components), np.nan)
    a_var = np.full((n_sources, n_components), np.nan)
    log_z = np.full((n_sources, n_components), np.nan)
    c_quad = np.full((n_sources, n_components), np.nan)
    pattern_dim = np.zeros(n_sources, dtype=np.int64)

    for group in science.pattern_groups(min_dim=min_dim):
        proj = group.projection
        d = proj.shape[0]
        pattern_dim[group.indices] = d
        k_proj = proj @ k_full  # (d,)

        for start in range(0, group.n_sources, _CHUNK):
            sl = slice(start, start + _CHUNK)
            idx = group.indices[sl]
            colors = group.colors[sl]
            err_cov = group.covariances[sl]

            for j in range(n_components):
                obs_cov = proj @ covariances[j] @ proj.T + err_cov
                chol = batched_cholesky(obs_cov)
                logdet = 2.0 * np.sum(
                    np.log(np.diagonal(chol, axis1=-2, axis2=-1)), axis=-1
                )
                deviation = colors - proj @ means[j]
                # Forward substitution for both right-hand sides at once
                rhs = np.concatenate(
                    [
                        np.broadcast_to(k_proj, deviation.shape)[..., None],
                        deviation[..., None],
                    ],
                    axis=-1,
                )
                solved = solve_lower(chol, rhs)
                v = solved[..., 0]
                w = solved[..., 1]

                vv = np.sum(v * v, axis=-1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    var = 1.0 / vv
                    mean = var * np.sum(w * v, axis=-1)
                    quad = np.sum(w * w, axis=-1) - mean**2 * vv
                a_mean[idx, j] = mean
                a_var[idx, j] = var
                log_z[idx, j] = -0.5 * (d * _LOG_2PI + logdet)
                c_quad[idx, j] = quad

    return PosteriorTerms(
        a_mean=a_mean,
        a_var=a_var,
        log_z=log_z,
        c_quad=c_quad,
        pattern_dim=pattern_dim,
    )


class ExtinctionPosterior:
    """Extinction posteriors for a source catalog: one 1-d GMM per source.

    Attributes
    ----------
    means : ndarray, shape (N, K)
        Component means A_ik.
    variances : ndarray, shape (N, K)
        Component variances sigma_ik^2.
    log_weights : ndarray, shape (N, K)
        Normalized component log-weights; NaN rows mark sources without an
        estimate.
    log_evidence : ndarray, shape (N,)
        Log of the marginalized likelihood (Eq. 15 denominator). Comparable
        only between sources with the same missingness pattern
        (`pattern_dim` gives the observed dimensionality).
    """

    def __init__(
        self,
        means: np.ndarray,
        variances: np.ndarray,
        log_weights: np.ndarray,
        log_evidence: np.ndarray,
        pattern_dim: np.ndarray,
        coordinates: SkyCoord | None = None,
        extinction_vector: np.ndarray | None = None,
    ) -> None:
        self.means = means
        self.variances = variances
        self.log_weights = log_weights
        self.log_evidence = log_evidence
        self.pattern_dim = pattern_dim
        self.coordinates = coordinates
        self.extinction_vector = extinction_vector

    @property
    def n_sources(self) -> int:
        return self.means.shape[0]

    @property
    def n_components(self) -> int:
        return self.means.shape[1]

    @property
    def weights(self) -> np.ndarray:
        """Component weights per source, shape (N, K)."""
        return np.exp(self.log_weights)

    def mean(self) -> np.ndarray:
        """Posterior mean extinction per source (moment merge, Eq. 31)."""
        return np.sum(self.weights * self.means, axis=1)

    def variance(self) -> np.ndarray:
        """Posterior variance per source (moment merge, Eq. 32)."""
        weights = self.weights
        mean = np.sum(weights * self.means, axis=1)
        second = np.sum(weights * (self.variances + self.means**2), axis=1)
        return second - mean**2

    def pdf(self, a_values: np.ndarray) -> np.ndarray:
        """Posterior densities evaluated at `a_values`, shape (N, G)."""
        a_values = np.asarray(a_values, dtype=np.float64)
        dev = a_values[None, None, :] - self.means[..., None]
        log_norm = -0.5 * (_LOG_2PI + np.log(self.variances))
        log_pdf = (
            self.log_weights[..., None]
            + log_norm[..., None]
            - 0.5 * dev**2 / self.variances[..., None]
        )
        return np.exp(logsumexp(log_pdf, axis=1))

    def discretize(self) -> ExtinctionCatalog:
        """Collapse the posteriors to point estimates with variances."""
        from pnicer.catalog import ExtinctionCatalog

        return ExtinctionCatalog(
            extinction=self.mean(),
            variance=self.variance(),
            coordinates=self.coordinates,
            extinction_vector=self.extinction_vector,
        )

    def __repr__(self) -> str:
        n_good = int(np.isfinite(self.log_evidence).sum())
        return (
            f"{type(self).__name__}(n_sources={self.n_sources}, "
            f"n_components={self.n_components}, n_estimates={n_good})"
        )


def build_posterior(
    terms: PosteriorTerms,
    log_weights: np.ndarray,
    coordinates: SkyCoord | None,
    extinction_vector: np.ndarray | None,
) -> ExtinctionPosterior:
    """Assemble an `ExtinctionPosterior` from geometry terms and weights."""
    log_f = terms.log_amplitudes(log_weights)
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(log_f).any(axis=1)
    log_evidence = np.full(terms.a_mean.shape[0], np.nan)
    log_evidence[valid] = logsumexp(
        np.nan_to_num(log_f[valid], nan=-np.inf), axis=1
    )
    log_w = log_f - log_evidence[:, None]
    log_w[~valid] = np.nan
    return ExtinctionPosterior(
        means=terms.a_mean,
        variances=terms.a_var,
        log_weights=log_w,
        log_evidence=log_evidence,
        pattern_dim=terms.pattern_dim,
        coordinates=coordinates,
        extinction_vector=extinction_vector,
    )


def adaptive_log_weights(
    terms: PosteriorTerms,
    base_log_weights: np.ndarray,
    weights_at_extinction: np.ndarray,
    a_grid: np.ndarray,
    n_iter: int = 3,
) -> np.ndarray:
    """Per-source effective component log-weights (Lombardi 2018, Sect. 2.6).

    Iteratively re-weights the mixture components: the current posterior
    p(A) of each source, evaluated on `a_grid`, weights the
    extinction-dependent component weights ``weights_at_extinction``
    (shape (G, K)), giving effective weights for the next pass.

    Returns
    -------
    ndarray, shape (N, K)
        Final per-source component log-weights (normalized).
    """
    log_w = np.broadcast_to(
        base_log_weights, terms.a_mean.shape
    ).copy()  # (N, K)

    for _ in range(n_iter):
        # Current posterior on the grid (per source), normalized discretely
        log_pdf = terms.log_pdf_grid(a_grid, log_w)  # (N, G)
        with np.errstate(invalid="ignore"):
            log_pdf = log_pdf - logsumexp(
                np.nan_to_num(log_pdf, nan=-np.inf), axis=1, keepdims=True
            )
        prob = np.exp(log_pdf)
        # Effective weights: expectation of w_k(A) under the posterior
        eff = prob @ weights_at_extinction  # (N, K)
        log_w = np.log(eff + 1e-300)
        log_w -= logsumexp(log_w, axis=1, keepdims=True)

    return log_w
