"""Batched linear algebra for stacks of small matrices.

The estimators factor and solve millions of tiny (d <= ~8) symmetric systems.
Generic batched LAPACK routines pay a per-matrix dispatch cost that dominates
at these sizes, so Cholesky factorization and triangular solves are unrolled
into vectorized elementwise operations over the batch dimensions; LAPACK is
used as a fallback for larger dimensions.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "batched_cholesky",
    "solve_lower",
    "solve_upper",
]

_SMALL_DIM_MAX = 8


def _cholesky_small(matrices: np.ndarray) -> np.ndarray:
    """Unrolled Cholesky for stacks of small SPD matrices.

    Raises
    ------
    np.linalg.LinAlgError
        If any matrix in the stack is not positive definite.
    """
    d = matrices.shape[-1]
    chol = np.zeros_like(matrices)
    for i in range(d):
        s = matrices[..., i, i].copy()
        for k in range(i):
            s -= chol[..., i, k] ** 2
        if np.any(s <= 0) or not np.all(np.isfinite(s)):
            raise np.linalg.LinAlgError("Matrix stack not positive definite")
        diag = np.sqrt(s)
        chol[..., i, i] = diag
        for j in range(i + 1, d):
            s = matrices[..., j, i].copy()
            for k in range(i):
                s -= chol[..., j, k] * chol[..., i, k]
            chol[..., j, i] = s / diag
    return chol


def batched_cholesky(matrices: np.ndarray, max_tries: int = 4) -> np.ndarray:
    """Cholesky factorization of a stack of symmetric matrices.

    Falls back to adding progressively larger diagonal jitter (relative to
    the mean diagonal) if any matrix in the stack is not positive definite.

    Parameters
    ----------
    matrices : ndarray, shape (..., d, d)
    max_tries : int
        Number of jitter escalations before giving up.

    Returns
    -------
    ndarray, shape (..., d, d)
        Lower-triangular factors L with ``L @ L.T`` equal to the (possibly
        jittered) input.
    """
    small = matrices.shape[-1] <= _SMALL_DIM_MAX
    factorize = _cholesky_small if small else np.linalg.cholesky
    try:
        return factorize(matrices)
    except np.linalg.LinAlgError:
        pass
    d = matrices.shape[-1]
    scale = np.mean(np.abs(np.diagonal(matrices, axis1=-2, axis2=-1)))
    scale = scale if np.isfinite(scale) and scale > 0 else 1.0
    jitter = 1e-10 * scale
    for _ in range(max_tries):
        try:
            return factorize(matrices + jitter * np.eye(d))
        except np.linalg.LinAlgError:
            jitter *= 100.0
    raise np.linalg.LinAlgError(
        "Covariance stack not positive definite even after jitter"
    )


def solve_lower(chol: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve L x = rhs for stacks of lower-triangular L.

    Parameters
    ----------
    chol : ndarray, shape (..., d, d)
    rhs : ndarray, shape (..., d) or (..., d, r)
        Right-hand side(s); broadcastable against the batch dimensions.

    Returns
    -------
    ndarray with the shape of `rhs`.
    """
    d = chol.shape[-1]
    if d > _SMALL_DIM_MAX:
        if rhs.ndim == chol.ndim - 1:
            return np.linalg.solve(chol, rhs[..., None])[..., 0]
        return np.linalg.solve(chol, rhs)

    matrix_rhs = rhs.ndim == chol.ndim
    out = np.empty(np.broadcast_shapes(
        chol.shape[:-2] + (d,) + ((rhs.shape[-1],) if matrix_rhs else ()),
        rhs.shape,
    ))
    if matrix_rhs:
        for i in range(d):
            s = rhs[..., i, :] + 0.0
            for k in range(i):
                s = s - chol[..., i, k, None] * out[..., k, :]
            out[..., i, :] = s / chol[..., i, i, None]
    else:
        for i in range(d):
            s = rhs[..., i] + 0.0
            for k in range(i):
                s = s - chol[..., i, k] * out[..., k]
            out[..., i] = s / chol[..., i, i]
    return out


def solve_upper(chol: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve L^T x = rhs for stacks of lower-triangular L (backward pass)."""
    d = chol.shape[-1]
    if d > _SMALL_DIM_MAX:
        upper = np.swapaxes(chol, -1, -2)
        if rhs.ndim == chol.ndim - 1:
            return np.linalg.solve(upper, rhs[..., None])[..., 0]
        return np.linalg.solve(upper, rhs)

    matrix_rhs = rhs.ndim == chol.ndim
    out = np.empty(np.broadcast_shapes(
        chol.shape[:-2] + (d,) + ((rhs.shape[-1],) if matrix_rhs else ()),
        rhs.shape,
    ))
    if matrix_rhs:
        for i in reversed(range(d)):
            s = rhs[..., i, :] + 0.0
            for k in range(i + 1, d):
                s = s - chol[..., k, i, None] * out[..., k, :]
            out[..., i, :] = s / chol[..., i, i, None]
    else:
        for i in reversed(range(d)):
            s = rhs[..., i] + 0.0
            for k in range(i + 1, d):
                s = s - chol[..., k, i] * out[..., k]
            out[..., i] = s / chol[..., i, i]
    return out
