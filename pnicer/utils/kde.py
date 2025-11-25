# -----------------------------------------------------------------------------
# Import packages
import numpy as np
import multiprocessing

from itertools import repeat
# noinspection PyPackageRequirements
from sklearn.neighbors import KernelDensity


# -----------------------------------------------------------------------------
def mp_kde(grid, data, bandwidth, kernel="epanechnikov", norm=None, absolute=False, sampling=None):
    """
    Kernel density estimation with parallelisation.

    Parameters
    ----------
    grid
        Grid on which to evaluate the density.
    data
        Input data
    bandwidth : int, float
        Bandwidth of kernel (in data units).
    kernel : str, optional
        Name of kernel for KDE. e.g. 'epanechnikov' or 'gaussian'. Default is 'epanechnikov'.
    norm : str, optional
        Whether to normalize the result (density estimate from 0 to 1). Default is False.
    absolute : bool, optional
        Whether to return absolute numbers.
    sampling : int, optional
        Sampling of grid. Necessary only when absolute numbers should be returned.

    Returns
    -------
    np.ndarray

    """

    # If we want absolute values, we must specify the sampling
    if absolute:
        if not sampling:
            raise ValueError("For absolute values, sampling needs to be specified")

    # Dimensions of grid and data must match
    if len(grid.shape) != len(data.shape):
        raise ValueError("Data and Grid dimensions must match")

    # If only one dimension, extend
    if len(grid.shape) == 1:
        grid = grid[:, np.newaxis]
        data = data[:, np.newaxis]

    # Split for parallel processing
    grid_split = np.array_split(grid, multiprocessing.cpu_count(), axis=0)

    # Define kernel
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)

    # Run kernel density calculation
    with multiprocessing.Pool() as pool:
        mp = pool.starmap(_mp_kde, zip(repeat(kde), repeat(data), grid_split))

    # Create array
    mp = np.concatenate(mp)

    # If we want absolute numbers we have to evaluate the same thing for the grid
    if absolute:
        mp *= data.shape[0] / np.sum(mp) * sampling

    # Normalize if set
    if norm == "max":
        mp /= np.nanmax(mp)
    elif norm == "mean":
        mp /= np.nanmean(mp)
    elif norm == "sum":
        mp /= np.nansum(mp)

    # Return
    return mp


# -----------------------------------------------------------------------------
def _mp_kde(kde, data, grid):
    """
    Parallelisation routine for kernel density estimation.

    Parameters
    ----------
    kde
        KernelDensity instance from scikit learn
    data
        Input data
    grid
        Grid on which to evaluate the density.

    Returns
    -------
    np.ndarray

    """

    return np.exp(kde.fit(data).score_samples(grid))
