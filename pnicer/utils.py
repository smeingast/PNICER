# ----------------------------------------------------------------------
# Import stuff
import numpy as np
import multiprocessing

from matplotlib import pyplot as plt
from multiprocessing.pool import Pool
from itertools import combinations, repeat
# noinspection PyPackageRequirements
from sklearn.neighbors import KernelDensity


# ----------------------------------------------------------------------
def distance_sky(lon1, lat1, lon2, lat2, unit="radians"):
    """
    Returns the distance between two objects on a sphere along the connecting great circle. Also works with arrays.

    Parameters
    ----------
    lon1 : int, float, np.ndarray
        Longitude (e.g. Right Ascension) of first object
    lat1 : int, float, np.ndarray
        Latitude (e.g. Declination) of first object
    lon2 : int, float, np.ndarray
        Longitude of object to calculate the distance to.
    lat2 : int, float, np.ndarray
        Longitude of object to calculate the distance to.
    unit : str, optional
        The unit in which the coordinates are given. Either 'radians' or 'degrees'. Default is 'radians'. Output will
        be in the same units.

    Returns
    -------
    float, np.ndarray
        On-sky distances between given objects.

    """

    # Calculate distance on sphere
    if "rad" in unit:

        # Haversine distance (better for small numbers)
        dis = 2 * np.arcsin(np.sqrt(np.sin((lat1 - lat2) / 2.) ** 2 +
                                    np.cos(lat1) * np.cos(lat2) * np.sin((lon1 - lon2) / 2.) ** 2))

    elif "deg" in unit:

        # Haversine distance (better for small numbers)
        dis = 2 * np.degrees(np.arcsin(np.sqrt(np.sin((np.radians(lat1) - np.radians(lat2)) / 2.) ** 2 +
                                               np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
                                               np.sin((np.radians(lon1) - np.radians(lon2)) / 2.) ** 2)))

    # If given unit is not supported.
    else:
        raise ValueError("Unit {0:s} not supported".format(unit))

    # Return distance
    return dis


# ----------------------------------------------------------------------
def weighted_avg(values, weights):
    """
    Calculates weighted mean and standard deviation.

    Parameters
    ----------
    values : np.ndarray
        Data values.
    weights : np.ndarray
        Weights for each data point.

    Returns
    -------
    tuple(np.ndarray, np.ndarray)
        Weighted mean and variance.

    """

    # Calculate weighted average
    average = np.nansum(values * weights) / np.nansum(weights)

    # Calculate weighted variance
    variance = np.nansum((values - average) ** 2 * weights) / np.nansum(weights)

    # Return both
    return average, variance


# ----------------------------------------------------------------------
def get_covar(xi, yi):
    """
    Calculate sample covariance (can not contain NaNs!)

    Parameters
    ----------
    xi : np.ndarray
        X data.
    yi : np.ndarray
        Y data.

    Returns
    -------
    float
        Sample covariance.

    """

    return np.sum((xi - np.mean(xi)) * (yi - np.mean(yi))) / len(xi)


# ----------------------------------------------------------------------
def linear_model(vec, val):
    """
    Linear function model y = m*x + b.

    Parameters
    ----------
    vec : iterable
        Vector of the parameters.
    val : np.ndarray
        array of the current x values

    Returns
    -------

    """

    return vec[0] * val + vec[1]


# ----------------------------------------------------------------------
def axes_combinations(ndim, ax_size=None):
    """
    Creates a grid of axes to plot all combinations of data.

    Parameters
    ----------
    ndim : int
        Number of dimensions.
    ax_size : list, optional
        Single axis size. Default is [3, 3].

    Returns
    -------
    list
        List of axes which can be used for plotting.

    """

    if ax_size is None:
        ax_size = [3, 3]

    # Get all combinations to plot
    c = combinations(range(ndim), 2)

    # Create basic plot layout
    fig, axes = plt.subplots(ncols=ndim - 1, nrows=ndim - 1, figsize=[(ndim - 1) * ax_size[0], (ndim - 1) * ax_size[1]])

    if ndim == 2:
        axes = np.array([[axes], ])

    # Adjust plots
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0, hspace=0)

    axes_out = []
    for idx in c:

        # Get index of subplot
        x_idx, y_idx = ndim - idx[0] - 2, ndim - 1 - idx[1]

        # Get axes
        axes_out.append(axes[x_idx, y_idx])

        # Hide tick labels
        if x_idx < ndim - 2:
            axes_out[-1].axes.xaxis.set_ticklabels([])
        if y_idx > 0:
            axes_out[-1].axes.yaxis.set_ticklabels([])

        # Delete the other axes
        if ((idx[0] > 0) | (idx[1] - 1 > 0)) & (idx[0] != idx[1] - 1):
            fig.delaxes(axes[idx[0], idx[1] - 1])

    return fig, axes_out


def mp_kde(grid, data, bandwidth, shape=None, kernel="epanechnikov", norm=False, absolute=False, sampling=None):
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
    shape
        If set, shape out ouput.
    kernel : str, optional
        Name of kernel for KDE. e.g. 'epanechnikov' or 'gaussian'. Default is 'epanechnikov'.
    norm : bool, optional
        Whether to normalize the result (density estimate from 0 to 1). Default is False.
    absolute : bool, optional
        Whether to return absolute numbers.
    sampling : int, optional
        Sampling of grid. Necessary only when absolute numbers should be returned.

    Returns
    -------
    np.ndarray

    """

    # TODO: remove shape parameter

    # If we want absolute values, we must specify the sampling
    if absolute:
        assert sampling, "Sampling needs to be specified"

    # Dimensions of grid and data must match
    assert len(grid.shape) == len(data.shape), "Data and Grid dimensions must match"

    # If only one dimension, extend
    if len(grid.shape) == 1:
        grid = grid[:, np.newaxis]
        data = data[:, np.newaxis]

    # Split for parallel processing
    grid_split = np.array_split(grid, multiprocessing.cpu_count(), axis=0)

    # Define kernel according to Nyquist sampling
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)

    with Pool() as pool:
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

    # Unpack results and return
    if shape is None:
        return mp
    else:
        return mp.reshape(shape)


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
