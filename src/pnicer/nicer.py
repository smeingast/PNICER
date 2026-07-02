"""The NICER extinction estimator (Lombardi & Alves 2001).

Closed-form generalization of the color-excess technique: the intrinsic color
distribution of the control field is described by its mean and covariance;
each source's extinction is the minimum-variance unbiased combination of its
color excesses (Eqs. 12-13 of the paper). Missing measurements are handled
exactly by projecting the problem onto each source's observed color subspace
(the sigma -> infinity limit of the down-weighting used in legacy PNICER).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pnicer.catalog import ExtinctionCatalog
    from pnicer.photometry import _ColorSpaceBase

__all__ = ["control_field_statistics", "nicer"]


def control_field_statistics(
    control: _ColorSpaceBase,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and covariance of the control-field colors.

    Uses all pairwise-complete observations (matching the legacy
    implementation based on masked-array statistics).

    Returns
    -------
    color0 : ndarray, shape (D,)
    cov : ndarray, shape (D, D)
    """
    colors = control.raw_colors  # (N, D), NaN = missing value
    color0 = np.nanmean(colors, axis=0)
    masked = np.ma.masked_invalid(colors.T)
    cov = np.ma.cov(masked)
    if colors.shape[1] == 1:
        cov = np.array([[float(cov)]])
    else:
        cov = np.asarray(cov)
    return color0, cov


def nicer(
    science: _ColorSpaceBase,
    control: _ColorSpaceBase | None = None,
    *,
    color0: np.ndarray | None = None,
    color0_cov: np.ndarray | None = None,
    min_dim: int = 1,
) -> ExtinctionCatalog:
    """Compute NICER extinction estimates for a science field.

    Parameters
    ----------
    science : Photometry or Colors
        Science-field data.
    control : Photometry or Colors, optional
        Extinction-free control field with the same color basis.
    color0 : ndarray, optional
        Intrinsic colors, shape (D,); alternative to a control field.
    color0_cov : ndarray, optional
        Intrinsic color covariance, shape (D, D) or (D,) diagonal; zero if
        omitted.
    min_dim : int
        Minimum number of observed colors required per source.

    Returns
    -------
    ExtinctionCatalog
    """
    from pnicer.catalog import ExtinctionCatalog

    ndim = science.n_colors
    if control is not None:
        if control.n_colors != ndim:
            raise ValueError("Science and control color spaces differ")
        intrinsic_mean, intrinsic_cov = control_field_statistics(control)
    elif color0 is not None:
        intrinsic_mean = np.asarray(color0, dtype=np.float64).ravel()
        if intrinsic_mean.size != ndim:
            raise ValueError(f"color0 must have {ndim} entries")
        if color0_cov is None:
            intrinsic_cov = np.zeros((ndim, ndim))
        else:
            intrinsic_cov = np.asarray(color0_cov, dtype=np.float64)
            if intrinsic_cov.ndim == 1:
                # 1-d input holds diagonal *variances* (documented semantics)
                intrinsic_cov = np.diag(intrinsic_cov)
    else:
        raise ValueError("Either a control field or intrinsic colors are required")

    if min_dim < 1:
        raise ValueError("min_dim must be at least 1")
    if min_dim > ndim:
        raise ValueError(f"Cannot require more than {ndim} colors")

    k_full = science.reddening_vector
    extinction = np.full(science.n_sources, np.nan)
    variance = np.full(science.n_sources, np.nan)

    for group in science.pattern_groups(min_dim=min_dim):
        proj = group.projection
        k = proj @ k_full
        # Total covariance per source: projected intrinsic + measurement
        cov = proj @ intrinsic_cov @ proj.T + group.covariances
        cov_inv = np.linalg.inv(cov)
        # b = C^-1 k / (k^T C^-1 k); A = b . (c - c0); Var = 1 / (k^T C^-1 k)
        upper = cov_inv @ k  # (n, d)
        lower = upper @ k  # (n,)
        deviation = group.colors - proj @ intrinsic_mean
        with np.errstate(divide="ignore", invalid="ignore"):
            extinction[group.indices] = np.sum(upper * deviation, axis=-1) / lower
            variance[group.indices] = 1.0 / lower

    return ExtinctionCatalog(
        extinction=extinction,
        variance=variance,
        coordinates=science.coordinates,
        extinction_vector=getattr(science, "extinction_vector", None),
    )
