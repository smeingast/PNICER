# -----------------------------------------------------------------------------
# Import packages
import numpy as np
from astropy.units import Unit
from astropy.coordinates import SkyCoord


# -----------------------------------------------------------------------------
# Useful constants
std2fwhm = 2 * np.sqrt(2 * np.log(2))


# -----------------------------------------------------------------------------
def gauss_function(x, amp, x0, sigma, area=None):
    """
    Simple gauss function sampler.

    Parameters
    ----------
    x : ndarray
        X data range
    amp : int, float
        Amplitude of gaussian.
    x0 : int, float
        Mean of gaussian
    sigma : int, float
        Standard deviation of gaussian.
    area : int, float, optional
        If set, normalize gaussian to that area.

    Returns
    -------
    ndarray
        Y data range.

    """

    # Get samples
    gauss = amp * np.exp(-((x - x0) ** 2.0) / (2.0 * sigma**2.0))

    # Normalize
    if area is not None:
        # noinspection PyTypeChecker
        gauss /= np.trapezoid(gauss, x) / area

    # Return
    return gauss


# -----------------------------------------------------------------------------
def distance_sky(lon1, lat1, lon2, lat2, unit="radians"):
    """
    Returns the distance between two objects on a sphere along the connecting great
    circle. Also works with arrays.

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
        The unit in which the coordinates are given. Either 'radians' or 'degrees'.
        Default is 'radians'. Output will
        be in the same units.

    Returns
    -------
    float, np.ndarray
        On-sky distances between given objects.

    """

    l1, l2 = (
        np.radians(lon1) if "deg" in unit else lon1,
        np.radians(lon2) if "deg" in unit else lon2,
    )
    b1, b2 = (
        np.radians(lat1) if "deg" in unit else lat1,
        np.radians(lat2) if "deg" in unit else lat2,
    )

    # Return haversine distance
    dis = 2 * np.arcsin(
        np.sqrt(
            np.sin((b1 - b2) / 2.0) ** 2
            + np.cos(b1) * np.cos(b2) * np.sin((l1 - l2) / 2.0) ** 2
        )
    )
    if "deg" in unit:
        return np.rad2deg(dis)
    else:
        return dis


# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
def centroid_sphere(skycoord: SkyCoord) -> SkyCoord:
    """
    Calculate the centroid on a sphere. Strictly valid only for a unit sphere and for a
    coordinate system with latitudes from -90 to 90 degrees and longitudes from 0 to
    360 degrees.

    Parameters
    ----------
    skycoord : SkyCoord
        SkyCoord instance

    Returns
    -------
    SkyCoord
        Centroid as SkyCoord instance.

    """

    # Filter finite entries in arrays
    good = np.isfinite(skycoord.spherical.lon) & np.isfinite(skycoord.spherical.lat)

    # 3D mean
    mean_x = np.mean(skycoord[good].cartesian.x)
    mean_y = np.mean(skycoord[good].cartesian.y)
    mean_z = np.mean(skycoord[good].cartesian.z)

    # Push mean to triangle surface
    cenlen = np.sqrt(mean_x**2 + mean_y**2 + mean_z**2)
    xsur, ysur, zsur = mean_x / cenlen, mean_y / cenlen, mean_z / cenlen

    # Convert back to spherical coordinates and return
    outlon = np.arctan2(ysur, xsur)

    # Convert back to 0-2pi range if necessary
    if outlon < 0:
        outlon += 2 * np.pi * Unit("rad")
    outlat = np.arcsin(zsur)

    return SkyCoord(outlon, outlat, frame=skycoord.frame)


# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
def get_color_covar(magerr1, magerr2, magerr3, magerr4, name1, name2, name3, name4):
    """
    Calculate the error covariance matrix for color combinations of four magnitudes.
    x: (mag1 - mag2)
    y: (mag3 - mag4)

    Parameters
    ----------
    magerr1 : np.ndarray
        Source magnitudes in band 1.
    magerr2 : np.ndarray
        Source magnitudes in band 2.
    magerr3 : np.ndarray
        Source magnitudes in band 3.
    magerr4 : np.ndarray
        Source magnitudes in band 4.
    name1 : str, int
        Unique identifier string or index for band 1.
    name2 : str, int
        Unique identifier string or index for band 2.
    name3 : str, int
        Unique identifier string or index for band 3.
    name4 : str, int
        Unique identifier string or index for band 4.

    Returns
    -------
    np.ndarray
        Covariance matrix for color errors.

    """

    # Initialize matrix
    cmatrix = np.zeros(shape=(2, 2))

    # Calculate first entry
    cmatrix[0, 0] = np.mean(magerr1) ** 2 + np.mean(magerr2) ** 2

    # Calculate last entry
    cmatrix[1, 1] = np.mean(magerr3) ** 2 * np.mean(magerr4) ** 2

    # Initially set cross entries to 0
    cov = 0.0

    # Add first term
    if name1 == name3:
        cov += np.mean(magerr1) * np.mean(magerr3)

    # Add second term
    if name1 == name4:
        cov -= np.mean(magerr1) * np.mean(magerr4)

    # Add third term
    if name2 == name3:
        cov -= np.mean(magerr2) * np.mean(magerr3)

    # Add fourth term
    if name2 == name4:
        cov += np.mean(magerr2) * np.mean(magerr4)

    # Set entries in matrix
    cmatrix[1, 0] = cmatrix[0, 1] = cov

    # Return total covariance
    return cmatrix


# -----------------------------------------------------------------------------
def get_sample_covar(xi, yi):
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

    # Sample size must be equal
    if len(xi) != len(yi):
        raise ValueError("X and Y sample size must be equal.")

    # Check for NaNs
    if (np.sum(~np.isfinite(xi)) > 0) | (np.sum(~np.isfinite(yi)) > 0):
        raise ValueError("Sample contains NaN entries")

    return np.sum((xi - np.mean(xi)) * (yi - np.mean(yi))) / len(xi)
