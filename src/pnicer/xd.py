"""Extreme deconvolution: Gaussian mixtures fitted to noisy, incomplete data.

Implements the EM algorithm of Bovy, Hogg & Roweis (2011, Ann. Appl. Stat. 5,
1657) for a Gaussian mixture model observed through per-point Gaussian noise
and (possibly rank-deficient) linear projections. This recovers the underlying
("deconvolved") distribution, in contrast to a plain GMM fit which models the
noisy observations.

Everything is vectorized over sources within groups sharing one projection
(missingness pattern); densities are evaluated in log space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from pnicer.linalg import batched_cholesky, solve_lower, solve_upper
from pnicer.photometry import PatternGroup

__all__ = ["XDResult", "fit_xd", "xd_log_likelihood"]

_LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass
class XDResult:
    """Result of an extreme deconvolution fit.

    Attributes
    ----------
    weights : ndarray, shape (K,)
    means : ndarray, shape (K, D)
    covariances : ndarray, shape (K, D, D)
    log_likelihood : float
        Total observed-data log-likelihood at the solution.
    bic : float
        -2 log L + p log N with p = (K-1) + K D + K D(D+1)/2 free parameters.
    responsibilities : ndarray, shape (N, K)
        Posterior component probabilities per fitted source; NaN rows for
        sources that did not enter the fit.
    n_iter : int
    converged : bool
    """

    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    log_likelihood: float
    bic: float
    responsibilities: np.ndarray
    n_iter: int
    converged: bool

    @property
    def n_components(self) -> int:
        return self.weights.size


def _estep_group(
    group: PatternGroup,
    weights: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    compute_latent: bool,
):
    """E-step quantities for one pattern group, batched over components.

    Returns per-source log-likelihoods (n,), responsibilities (n, K) and,
    if requested, latent color posteriors b_ij (n, K, D) and second-moment
    matrices (B_ij + b_ij b_ij^T) (n, K, D, D).
    """
    proj = group.projection
    n, d = group.colors.shape
    n_components, n_dim = means.shape

    proj_means = means @ proj.T  # (K, d)
    vpt = covariances @ proj.T  # (K, D, d)
    pvp = np.einsum("da,kab->kdb", proj, vpt)  # (K, d, d)

    # Observed-space covariances and deviations, batched: (n, K, d, d)
    obs_cov = pvp[None, :, :, :] + group.covariances[:, None, :, :]
    deviation = group.colors[:, None, :] - proj_means[None, :, :]

    chol = batched_cholesky(obs_cov)
    solved = solve_lower(chol, deviation)
    logdet = 2.0 * np.sum(np.log(np.diagonal(chol, axis1=-2, axis2=-1)), axis=-1)
    maha = np.sum(solved**2, axis=-1)
    log_prob = (
        np.log(weights)[None, :]
        - 0.5 * (d * _LOG_2PI + logdet + maha)
    )

    latent_mean = None
    latent_m2 = None
    if compute_latent:
        # T^-1 (y - P b) via back-substitution of the Cholesky solve
        t_inv_dev = solve_upper(chol, solved)  # (n, K, d)
        latent_mean = means[None] + np.einsum("kDd,nkd->nkD", vpt, t_inv_dev)
        # B_ij = V - V P^T T^-1 P V
        pv = np.broadcast_to(
            np.swapaxes(vpt, -1, -2)[None], (n, n_components, d, n_dim)
        )
        t_inv_pv = solve_upper(chol, solve_lower(chol, pv))  # (n, K, d, D)
        b_cov = covariances[None] - np.einsum("kDd,nkdE->nkDE", vpt, t_inv_pv)
        latent_m2 = b_cov + np.einsum(
            "nka,nkb->nkab", latent_mean, latent_mean
        )

    log_norm = logsumexp(log_prob, axis=1)
    resp = np.exp(log_prob - log_norm[:, None])
    return log_norm, resp, latent_mean, latent_m2


def xd_log_likelihood(
    groups: list[PatternGroup],
    weights: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
) -> float:
    """Observed-data log-likelihood of an XD model on pattern groups."""
    total = 0.0
    for group in groups:
        log_norm, *_ = _estep_group(group, weights, means, covariances, False)
        total += float(np.sum(log_norm))
    return total


def _initial_parameters(
    groups: list[PatternGroup],
    n_dim: int,
    n_components: int,
    random_state: int | None,
    reg_covar: float,
):
    """Initialize from a plain GMM on the complete-pattern observations.

    The GMM is fitted to the noisy observed colors; its covariances are
    therefore broadened by the measurement errors. Subtracting the mean
    error covariance (clipped to stay positive definite) starts the EM close
    to the deconvolved solution, which substantially reduces the number of
    iterations needed.
    """
    from sklearn.mixture import GaussianMixture

    complete = [
        g for g in groups if g.projection.shape[0] == n_dim
    ]  # identity projection: full color space observed
    if complete:
        data = np.vstack([g.colors for g in complete])
        mean_err_cov = np.vstack(
            [g.covariances.mean(axis=0)[None] for g in complete]
        ).mean(axis=0)
    else:
        data = np.vstack([g.colors @ np.linalg.pinv(g.projection).T for g in groups])
        mean_err_cov = np.zeros((n_dim, n_dim))

    if data.shape[0] < 2 * n_components:
        raise ValueError(
            f"Too few usable sources ({data.shape[0]}) to initialize "
            f"{n_components} components"
        )

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        reg_covar=max(reg_covar, 1e-6),
        random_state=random_state,
        n_init=1,
    ).fit(data)

    # Deconvolve the initialization: clip eigenvalues to keep SPD
    covariances = gmm.covariances_ - mean_err_cov[None]
    eigval, eigvec = np.linalg.eigh(covariances)
    eigval = np.maximum(eigval, 1e-4)
    covariances = np.einsum(
        "kab,kb,kcb->kac", eigvec, eigval, eigvec
    )
    return gmm.weights_.copy(), gmm.means_.copy(), covariances


def fit_xd(
    groups: list[PatternGroup],
    n_dim: int,
    n_sources: int,
    n_components: int,
    *,
    random_state: int | None = None,
    reg_covar: float = 1e-6,
    tol: float = 1e-6,
    max_iter: int = 500,
    weight_floor: float = 1e-8,
) -> XDResult:
    """Fit a Gaussian mixture by extreme deconvolution.

    Parameters
    ----------
    groups : list of PatternGroup
        Observations grouped by missingness pattern.
    n_dim : int
        Dimensionality D of the underlying (color) space.
    n_sources : int
        Total number of sources in the parent catalog (defines the shape of
        the responsibility array; sources absent from `groups` get NaN).
    n_components : int
        Number of mixture components K.
    random_state : int, optional
        Seed for the initialization.
    reg_covar : float
        Ridge added to covariance diagonals in each M-step.
    tol : float
        Convergence threshold on the average per-point log-likelihood
        change between EM iterations.
    max_iter : int
        Maximum number of EM iterations.
    weight_floor : float
        Minimum component weight (renormalized after flooring).

    Returns
    -------
    XDResult
    """
    if not groups:
        raise ValueError("No usable pattern groups to fit")
    weights, means, covariances = _initial_parameters(
        groups, n_dim, n_components, random_state, reg_covar
    )

    n_used = sum(g.n_sources for g in groups)
    log_likelihood = -np.inf
    converged = False
    iteration = 0

    for iteration in range(1, max_iter + 1):
        # Accumulate expected sufficient statistics over all groups
        resp_sum = np.zeros(n_components)
        mean_sum = np.zeros((n_components, n_dim))
        m2_sum = np.zeros((n_components, n_dim, n_dim))
        ll_total = 0.0

        for group in groups:
            log_norm, resp, latent_mean, latent_m2 = _estep_group(
                group, weights, means, covariances, True
            )
            ll_total += float(np.sum(log_norm))
            resp_sum += resp.sum(axis=0)
            mean_sum += np.einsum("nj,njd->jd", resp, latent_mean)
            m2_sum += np.einsum("nj,njde->jde", resp, latent_m2)

        # M-step
        weights = resp_sum / n_used
        weights = np.maximum(weights, weight_floor)
        weights /= weights.sum()
        means = mean_sum / resp_sum[:, None]
        covariances = m2_sum / resp_sum[:, None, None] - np.einsum(
            "jd,je->jde", means, means
        )
        covariances = 0.5 * (covariances + np.swapaxes(covariances, -1, -2))
        covariances[:, np.arange(n_dim), np.arange(n_dim)] += reg_covar

        # Converge on the average per-point log-likelihood change (the
        # total can cross zero, which breaks purely relative criteria)
        change = (ll_total - log_likelihood) / n_used
        log_likelihood = ll_total
        if iteration > 1 and abs(change) <= tol:
            converged = True
            break

    # Final E-step for responsibilities at the solution
    responsibilities = np.full((n_sources, n_components), np.nan)
    log_likelihood = 0.0
    for group in groups:
        log_norm, resp, *_ = _estep_group(group, weights, means, covariances, False)
        responsibilities[group.indices] = resp
        log_likelihood += float(np.sum(log_norm))

    n_params = (n_components - 1) + n_components * n_dim * (n_dim + 3) // 2
    bic = -2.0 * log_likelihood + n_params * np.log(n_used)

    return XDResult(
        weights=weights,
        means=means,
        covariances=covariances,
        log_likelihood=log_likelihood,
        bic=bic,
        responsibilities=responsibilities,
        n_iter=iteration,
        converged=converged,
    )
