# ----------------------------------------------------------------------
# Import stuff
import os
import sys
import importlib
import numpy as np
import multiprocessing

from astropy import wcs
from astropy.io import fits
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
    Calculate sample covariance (can not contain NaNs!).

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
def round_partial(data, precision):
    """
    Simple static method to round data to arbitrary precision.

    Parameters
    ----------
    data : float, np.ndarray
        Data to be rounded.
    precision : float, np.ndarray
        Desired precision. e.g. 0.2.

    Returns
    -------
    float, np.ndarray
        Rounded data.

    """

    return np.around(data / precision) * precision


# ----------------------------------------------------------------------
def caxes(ndim, ax_size=None, labels=None):
    """
    Creates a grid of axes to plot all combinations of data.

    Parameters
    ----------
    ndim : int
        Number of dimensions.
    ax_size : list, optional
        Single axis size. Default is [3, 3].
    labels : iterable, optional
        Optional list of feature names

    Returns
    -------
    tuple
        tuple containing the figure and a list of the axes.

    """

    if labels is not None:
        if len(labels) != ndim:
            raise ValueError("Number of provided labels must match dimensions")

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
        x_idx, y_idx = ndim - idx[0] - 2, ndim - idx[1] - 1

        # Grab axis
        ax = axes[x_idx, y_idx]

        # Hide tick labels
        if x_idx < ndim - 2:
            ax.axes.xaxis.set_ticklabels([])
        if y_idx > 0:
            ax.axes.yaxis.set_ticklabels([])

        # Add axis labels
        if labels is not None:
            if ax.get_position().x0 < 0.11:
                ax.set_ylabel("$" + labels[idx[0]] + "$")
            if ax.get_position().y0 < 0.11:
                ax.set_xlabel("$" + labels[idx[1]] + "$")

        # Append axes to return list
        axes_out.append(axes[x_idx, y_idx])

        # Delete not necessary axes
        if ((idx[0] > 0) | (idx[1] - 1 > 0)) & (idx[0] != idx[1] - 1):
            fig.delaxes(axes[idx[0], idx[1] - 1])

    return fig, axes_out


# ----------------------------------------------------------------------
def caxes_delete_ticklabels(axes, xfirst=False, xlast=False, yfirst=False, ylast=False):
    """
    Deletes tick labels from a combination axes list.

    Parameters
    ----------
    axes : iterable
        The combination axes list.
    xfirst : bool, optional
        Whether the first x label should be deleted.
    xlast : bool, optional
        Whether the last x label should be deleted.
    yfirst : bool, optional
        Whether the first y label should be deleted.
    ylast : bool, optional
        Whether the last y label should be deleted.


    """

    # Loop through the axes
    for ax, idx in zip(axes, combinations(range(len(axes)), 2)):

        # Modify x ticks
        if idx[0] == 0:

            # Grab ticks
            xticks = ax.xaxis.get_major_ticks()

            # Conditionally delete
            if xfirst:
                xticks[0].set_visible(False)
            if xlast:
                xticks[-1].set_visible(False)

        if idx[1] == np.max(idx):

            # Grab ticks
            yticks = ax.yaxis.get_major_ticks()

            # Conditionally delete
            if yfirst:
                yticks[0].set_visible(False)
            if ylast:
                yticks[-1].set_visible(False)


# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
def get_resource_path(package, resource):
    """
    Returns the path to an included resource.

    Parameters
    ----------
    package : str
        package name (e.g. astropype.resources.sextractor).
    resource : str
        Name of the resource (e.g. default.conv)

    Returns
    -------
    str
        Path to resource.

    """

    # Import package
    importlib.import_module(name=package)

    # Return path to resource
    return os.path.join(os.path.dirname(sys.modules[package].__file__), resource)


# ----------------------------------------------------------------------
def centroid_sphere(lon, lat, units="radian"):
    """
    Calcualte the centroid on a sphere. Strictly valid only for a unit sphere and for a coordinate system with latitudes
    from -90 to 90 degrees and longitudes from 0 to 360 degrees.

    Parameters
    ----------
    lon : list, np.array
        Input longitudes
    lat : list, np.array
        Input latitudes
    units : str, optional
        Input units. Either 'radian' or 'degree'. Default is 'radian'.

    Returns
    -------
    tuple
        Tuple with (lon, lat) of centroid

    """

    # Convert to radians if degrees
    if "deg" in units.lower():
        mlon, mlat = np.radians(lon), np.radians(lat)
    else:
        mlon, mlat = lon, lat

    # Convert to cartesian coordinates
    x, y, z = np.cos(mlat) * np.cos(mlon), np.cos(mlat) * np.sin(mlon), np.sin(mlat)

    # 3D centroid
    xcen, ycen, zcen = np.sum(x) / len(x), np.sum(y) / len(y), np.sum(z) / len(z)

    # Push centroid to triangle surface
    cenlen = np.sqrt(xcen**2 + ycen**2 + zcen**2)
    xsur, ysur, zsur = xcen / cenlen, ycen / cenlen, zcen / cenlen

    # Convert back to spherical coordinates and return
    outlon = np.arctan2(ysur, xsur)

    # Convert back to 0-2pi range if necessary
    if outlon < 0:
        outlon += 2 * np.pi
    outlat = np.arcsin(zsur)

    # Return
    if "deg" in units.lower():
        return np.degrees(outlon), np.degrees(outlat)
    else:
        return outlon, outlat


# ----------------------------------------------------------------------
def data2header(lon, lat, frame="icrs", proj_code="CAR", pixsize=1/3600, enlarge=1.05, **kwargs):
    """
    Create an astropy Header instance from a given dataset (longitude/latitude). The world coordinate system can be
    chosen between galactic and equatorial; all WCS projections are supported. Very useful for creating a quick WCS
    to plot data.

    Parameters
    ----------
    lon : list, np.array
        Input list or array of longitude coordinates in degrees.
    lat : list, np.array
        Input list or array of latitude coordinates in degrees.
    frame : str, optional
        World coordinate system frame of input data ('icrs' or 'galactic')
    proj_code : str, optional
        Projection code. (e.g. 'TAN', 'AIT', 'CAR', etc)
    pixsize : int, float, optional
        Pixel size of generated header in degrees. Not so important for plots, but still required.
    enlarge : float, optional
        Optional enlargement factor for calculated field size. Default is 1.05. Set to 1 if no enlargement is wanted.
    kwargs
        Additional projection parameters (e.g. pv2_1=-30)

    Returns
    -------
    astropy.fits.Header
        Astropy fits header instance.

    """

    # Define projection
    crval1, crval2 = centroid_sphere(lon=lon, lat=lat, units="degree")

    # Projection code
    if frame.lower() == "icrs":
        ctype1 = "RA{:->6}".format(proj_code)
        ctype2 = "DEC{:->5}".format(proj_code)
        frame = "equ"
    elif frame.lower() == "galactic":
        ctype1 = "GLON{:->4}".format(proj_code)
        ctype2 = "GLAT{:->4}".format(proj_code)
        frame = "gal"
    else:
        raise ValueError("Projection system {0:s} not supported".format(frame))

    # Build additional string
    additional = ""
    for key, value in kwargs.items():
        additional += ("{0: <8}= {1}\n".format(key.upper(), value))

    # Create preliminary header without size information
    header = fits.Header.fromstring("NAXIS   = 2" + "\n"
                                    "CTYPE1  = '" + ctype1 + "'\n"
                                    "CTYPE2  = '" + ctype2 + "'\n"
                                    "CRVAL1  = " + str(crval1) + "\n"
                                    "CRVAL2  = " + str(crval2) + "\n"
                                    "CUNIT1  = 'deg'" + "\n"
                                    "CUNIT2  = 'deg'" + "\n"
                                    "CDELT1  = -" + str(pixsize) + "\n"
                                    "CDELT2  = " + str(pixsize) + "\n"
                                    "COORDSYS= '" + frame + "'" + "\n" +
                                    additional,
                                    sep="\n")

    # Determine extent of data for this projection
    x, y = wcs.WCS(header).wcs_world2pix(lon, lat, 1)
    naxis1 = (np.ceil((x.max()) - np.floor(x.min())) * enlarge).astype(int)
    naxis2 = (np.ceil((y.max()) - np.floor(y.min())) * enlarge).astype(int)

    # Calculate pixel shift relative to centroid (caused by unisotropic distribution of sources)
    xdelta = (x.min() + x.max()) / 2
    ydelta = (y.min() + y.max()) / 2

    # Add size to header
    header["NAXIS1"], header["NAXIS2"] = naxis1, naxis2
    header["CRPIX1"], header["CRPIX2"] = naxis1 / 2 - xdelta, naxis2 / 2 + ydelta

    # Return Header
    return header