# -----------------------------------------------------------------------------
# Import packages
import numpy as np

from itertools import repeat
from multiprocessing import Pool
from scipy.integrate import cumtrapz
# noinspection PyPackageRequirements
from sklearn.mixture import GaussianMixture
from pnicer.utils.algebra import gauss_function


# -----------------------------------------------------------------------------
def gmm_scale(gmm, shift=None, scale=None, reverse=False, params=None):
    """
    Apply scaling factors to GMM instances.

    Parameters
    ----------
    gmm : GaussianMixture
        GMM instance to be scaled.
    shift : int, float, optional
        Shift for the entire model. Default is 0 (no shift).
    scale : int, float, optional
        Scale for all components. Default is 1 (no scale).
    reverse : bool, optional
        Whether the GMM should be reversed.
    params
        GaussianMixture params for initialization of new instance.

    Returns
    -------
    GaussianMixture
        Modified GMM instance.

    """

    # Fetch parameters if not supplied
    if params is None:
        # noinspection PyUnresolvedReferences
        params = gmm.get_params()

    # Instantiate new GMM
    gmm_new = GaussianMixture(**params)

    # Create scaled fitted GMM model
    gmm_new.weights_ = gmm.weights_

    # Apply shift if set
    gmm_new.means_ = gmm.means_ + shift if shift is not None else gmm.means_

    # Apply scale
    if scale is not None:
        gmm_new.means_ /= scale

    gmm_new.covariances_ = gmm.covariances_ / scale ** 2 if scale is not None else gmm.covariances_
    gmm_new.precisions_ = np.linalg.inv(gmm_new.covariances_) if scale is not None else gmm.precisions_
    gmm_new.precisions_cholesky_ = np.linalg.cholesky(gmm_new.precisions_) if scale is not None \
        else gmm.precisions_cholesky_

    # Reverse if set
    if reverse:
        gmm_new.means_ *= -1

    # Add converged attribute if available
    if gmm.converged_:
        gmm_new.converged_ = gmm.converged_

    # Return scaled GMM
    return gmm_new


# -----------------------------------------------------------------------------
def gmm_sample_xy(gmm, kappa=3, sampling=10, nmin=100, nmax=100000):
    """
    Creates discrete values (x, y) for a given GMM instance.

    Parameters
    ----------
    gmm : GaussianMixture
        Fitted GaussianMixture model instance from which to draw samples.
    kappa : int, float
        Width of query range (in units of standard deviations).
    sampling : int
        Sampling factor for the smallest component standard deviation.
    nmin : int
        Minimum number of samples to draw.
    nmax : int
        Maximum number of samples to draw. Not recommended to set this to > 1E5 as it becomes very slow.

    Returns
    -------
    ndarray, ndarray
        Tuple holding x and y data arrays.

    """

    # Get GMM attributes
    s = np.sqrt(gmm.covariances_.ravel())
    m = gmm.means_.ravel()

    # Get min-max range options
    roptions = list(zip(*[(float(mm - kappa * ss), float(mm + kappa * ss)) for ss, mm in zip(s, m)]))

    # Determine min and max of range
    qmin, qmax = np.min(roptions[0]), np.max(roptions[1])

    # Determine number of samples (number of samples from smallest standard deviation with 'sampling' samples)
    nsamples = (qmax - qmin) / (np.min(s) / sampling)

    # Set min/max numer of samples
    nsamples = 100 if nsamples < nmin else nsamples
    nsamples = 100000 if nsamples > nmax else nsamples

    # Get query range
    xrange = np.linspace(start=qmin, stop=qmax, num=nsamples)

    # Score samples
    yrange = np.exp(gmm.score_samples(np.expand_dims(xrange, 1)))

    # Step
    return xrange, yrange


# -----------------------------------------------------------------------------
def gmm_sample_xy_components(gmm, **kwargs):
    """
    Gets samples for each GMM component individually.

    Parameters
    ----------
    gmm : GaussianMixture
        The GMM to be sampled.
    kwargs
        Any additional keyword arguments for drawing samples from GMM.

    Returns
    -------
    ndarray, iterable
        Tuple holding the x data range and a list of y data for each component of the GMM.

    """

    # Draw samples for entire range
    x, y = gmm_sample_xy(gmm=gmm, **kwargs)

    # Sample each component separately
    y_components = []
    for m, c, w in zip(gmm.means_.ravel(), gmm.covariances_.ravel(), gmm.weights_.ravel()):
        y_components.append(gauss_function(x=x, amp=1, x0=m, sigma=np.sqrt(c), area=w))

    # Return
    return x, y_components


# -----------------------------------------------------------------------------
def gmm_max(gmm, sampling=50):
    """
    Returns the coordinates of the maximum of the probability density distribution defined by the GMM.

    Parameters
    ----------
    gmm : GaussianMixture
        Input GMM for which the maximum should be determined.
    sampling : int, optional
        Sampling factor for GMM. The larger, the better the maximum will be determined. Default is 50.

    Returns
    -------
    iterable
        Maximum coordinate.

    """

    # Draw samples
    x, y = gmm_sample_xy(gmm=gmm, kappa=1, sampling=sampling)

    # Return maximum
    return x[np.argmax(y)]


# -----------------------------------------------------------------------------
def gmm_expected_value(gmm, sampling=50):
    """
    Returns the coordinates of the expected value of the probability density distribution defined by the GMM.

    Parameters
    ----------
    gmm : GaussianMixture
        Input GMM for which the expected value should be determined.
    sampling : int, optional
        Sampling factor for GMM. The larger, the better the expected value will be determined. Default is 50.

    Returns
    -------
    iterable
        Expected value coordinate.

    """

    # Draw samples
    xrange, yrange = gmm_sample_xy(gmm=gmm, kappa=10, sampling=sampling)

    # Return expected value
    return np.trapz(xrange * yrange, xrange)


# -----------------------------------------------------------------------------
def gmm_confidence_interval(gmm, level=0.9, sampling=50):
    """
    Returns the confidence interval for a given gaussian mixture model and for a given confidence level.

    Parameters
    ----------
    gmm : GaussianMixture
        Input GMM for which the interval should be determined
    level : float, optional
        Confidence level (between 0 and 1). Default is 90%
    sampling : int
        Sampling factor for GMM. The larger, the better the interval will be determined. Default is 50.

    Returns
    -------
    tuple
        Confidence interval (min, max)

    """

    # Draw samples
    xrange, yrange = gmm_sample_xy(gmm=gmm, kappa=10, sampling=sampling)

    # Cumulative integral
    cumint = cumtrapz(y=yrange, x=xrange, initial=0)

    # Return interval
    return tuple(np.interp([(1 - level) / 2, level + (1 - level) / 2], cumint, xrange))


# -----------------------------------------------------------------------------
def gmm_population_variance(gmm):
    """
    Determine the population variance of the probability density distribution given by a GMM.

    Parameters
    ----------
    gmm : GaussianMixture
        Input Gaussian Mixture Model.

    Returns
    -------
    float
        Population variance for GMM.

    """

    # Get expected value
    ev = gmm_expected_value(gmm=gmm)

    # Get query range
    xrange, yrange = gmm_sample_xy(gmm=gmm, kappa=10, sampling=50)

    # Return population variance
    return np.trapz(np.power(xrange, 2) * yrange, xrange) - ev**2


# -----------------------------------------------------------------------------
def gmm_confidence_interval_value(gmm, value, level=0.95):
    # TODO: Check if this works as intended

    # Get query ranges
    gmm_x, gmm_y = gmm_sample_xy(gmm=gmm, kappa=5, sampling=100, nmin=1000, nmax=100000)

    # Get dx
    dx = np.ediff1d(gmm_x)[0]

    # Find position of 'value'
    value_idx = np.argmin(np.abs(gmm_x - value))

    # Integrate as long as necessary
    for i in range(len(gmm_x)):

        # Current index on both sides
        lidx = value_idx - i if value_idx - i >= 0 else 0
        ridx = value_idx + i if value_idx + i < len(gmm_x) else len(gmm_x) - 1

        # Need to separate left and right integral due to asymmetry
        lint = np.trapz(gmm_y[lidx:value_idx], dx=dx)
        rint = np.trapz(gmm_y[value_idx:ridx], dx=dx)

        # Sum of both sides
        # integral = np.trapz(gmm_y[lidx:ridx], dx=dx)
        integral = lint + rint

        # Break if confidence level reached
        if integral > level:
            break

    # Choose final index for confidence interval
    # noinspection PyUnboundLocalVariable
    ci_half_size = value - gmm_x[lidx] if value_idx - lidx > ridx - value_idx else gmm_x[ridx] - value

    # Return interval
    return value - ci_half_size, value + ci_half_size


# -----------------------------------------------------------------------------
def gmm_components(data, max_components, n_per_component=20):
    """
    Simple estimator for number of components.

    Parameters
    ----------
    data : ndarray
        Data array.
    max_components : int
        Maximum number of components.
    n_per_component : int, optional
        The minimum number of data points per component. Defaults to 20

    Returns
    -------
    int
        Number of components.

    """

    if data is None:
        return None

    # Determine number of components for GMM
    n_components = np.round(len(data.ravel()) / n_per_component, decimals=0).astype(int)
    n_components = max_components if n_components > max_components else n_components
    n_components = 1 if n_components < 1 else n_components

    # Return
    return n_components


# -----------------------------------------------------------------------------
def mp_gmm(data, max_components, parallel=True, **kwargs):
    """
    Gaussian mixture model fitting with parallelisation. The parallelisation only works when mutliple sets need to be
    fit.

    Parameters
    ----------
    data : iterable
        Iterable (list) of data vectors to be fit
    max_components : int, iterable
        Maximum number of components for all models.
    parallel : bool, optional
        Whether to use parallelisation.
    kwargs
        Additional keyword arguments for GaussianMixture class.

    Returns
    -------
    iterable
        List of results of fitting.

    """

    # TODO: Make parallelisation a user choice

    # Determine number of components for each data vector
    n_components = [gmm_components(data=d, max_components=max_components) for d in data]

    """
    The parallelisation can break in some cases. See the following page for more info
    http://stackoverflow.com/questions/19705200/multiprocessing-with-numpy-makes-python-quit-unexpectedly-on-osx

    To solve this on macOS, compile numpy (and maybe scipy) with e.g. openBLAS:

    sudo port install py36-numpy +gfortran +openblas
    sudo port install py36-scipy +gfortran +gcc6 +openblas
    """

    # Fit models with parallelisation
    if parallel:
        with Pool() as pool:
            return pool.starmap(_mp_gmm, zip(data, n_components, repeat(kwargs)))

    # or without
    else:
        return [GaussianMixture(n_components=n, **kwargs).fit(X=d) if d is not None else None
                for n, d in zip(n_components, data)]


# -----------------------------------------------------------------------------
def _mp_gmm(data, n_components, kwargs):
    """
    Gaussian mixture model fitting helper routine.

    Parameters
    ----------
    data : np.array
        Data to be fit.
    n_components : int
        Number of components for model.
    kwargs
        Additional keyword arguments for GaussianMixture class.

    Returns
    -------
        GaussianMixture class or NaN in case the fitting procedure did not converge or was not possible.

    """

    # If no data is given
    if data is None:
        return None

    # Fit model
    gmm = GaussianMixture(n_components=n_components, **kwargs).fit(X=data)

    # Check for convergence and return
    if gmm.converged_:
        return gmm
    else:
        return None
