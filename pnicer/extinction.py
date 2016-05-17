# ----------------------------------------------------------------------
# Import stuff
import wcsaxes
import warnings
import numpy as np

from astropy.io import fits
from itertools import repeat
from multiprocessing.pool import Pool

from pnicer.common import Coordinates
from pnicer.utils import distance_sky


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# noinspection PyProtectedMember
class Extinction:

    # Useful constants
    std2fwhm = 2 * np.sqrt(2 * np.log(2))

    def __init__(self, coordinates, extinction, variance=None, color0=None):
        """
        Class for extinction measurements.

        Parameters
        ----------
        coordinates : SkyCoord
            Astropy SkyCoord instance.
        extinction : np.ndarray
            Extinction data.
        variance : np.ndarray, optional
            Variance in extinction.
        color0 : np.ndarray, optional
            Intrisic color set for each source.

        """

        # Set attributes
        self.coordinates = Coordinates(coordinates=coordinates)
        self.extinction = extinction
        self.variance = np.zeros_like(extinction) if variance is None else variance
        self.color0 = np.zeros_like(extinction) if color0 is None else color0

        # Sanity checks
        if len(self.extinction) != len(self.variance):
            raise ValueError("Extinction and variance arrays must have equal length")

    # ---------------------------------------------------------------------- #
    #                              Magic methods                             #
    # ---------------------------------------------------------------------- #

    # ----------------------------------------------------------------------
    def __len__(self):
        return len(self.extinction)

    # ----------------------------------------------------------------------
    def __str__(self):
        return str(self.extinction)

    # ---------------------------------------------------------------------- #
    #                               Properties                               #
    # ---------------------------------------------------------------------- #

    # ----------------------------------------------------------------------
    @property
    def _clean_index(self):
        """
        Index of finite extinction measurements.

        Returns
        -------
        np.ndarray

        """

        return np.isfinite(self.extinction)

    # TODO: Move this to DataBase routine
    # # ----------------------------------------------------------------------
    # @property
    # def features_dered(self):
    #     """
    #     Dereddened features.
    #
    #     Returns
    #     -------
    #     list
    #
    #     """
    #
    #     return [f - self.extinction * v for f, v in zip(self.db.features, self.db.extvec.extvec)]

    # ---------------------------------------------------------------------- #
    #                            Instance methods                            #
    # ---------------------------------------------------------------------- #

    # ----------------------------------------------------------------------
    def build_map(self, bandwidth, metric="median", sampling=2, nicest=False, use_fwhm=False):
        """
        Method to build an extinction map.

        Parameters
        ----------
        bandwidth : int, float
            Resolution of output map.
        metric : str, optional
            Metric to be used. One of 'median', 'gaussian', 'epanechnikov', 'uniform, 'triangular'. Default is 'median'.
        sampling : int, optional
            Sampling of data. i.e. how many pixels per bandwidth. Default is 2.
        nicest : bool, optional
            Whether to activate the NICEST correction factor.
        use_fwhm : bool, optional
            If set, the bandwidth parameter represents the gaussian FWHM instead of its standard deviation. Only
            available when using a gaussian weighting.

        Returns
        -------
        ExtinctionMap
            ExtinctionMap instance.

        """

        # Sampling must be an integer
        if not isinstance(sampling, int):
            raise ValueError("Sampling factor must be an integer")

        # FWHM can only be used with a gaussian metric
        if use_fwhm & (metric != "gaussian"):
            raise ValueError("FWHM only valid for gaussian kernel")

        # Determine pixel size
        pixsize = bandwidth / sampling

        # Create WCS grid
        grid_header, (grid_lon, grid_lat) = self.coordinates.build_wcs_grid(proj_code="CAR", pixsize=pixsize)

        # Adjust bandwidth in case FWHM is to be used
        if use_fwhm:
            grid_header["FWHM"] = (bandwidth, "FWHM of gaussian (degrees)")
            bandwidth /= self.std2fwhm
        elif metric == "gaussian":
            grid_header["FWHM"] = (bandwidth * self.std2fwhm, "FWHM of gaussian (degrees)")

        # Add bandwidth to header
        grid_header["BWIDTH"] = (bandwidth, "Bandwidth of kernel (degrees)")

        # Run extinction mapping for each pixel
        with Pool() as pool:
            mp = pool.starmap(get_extinction_pixel,
                              zip(grid_lon.ravel(), grid_lat.ravel(), repeat(self.coordinates.lon[self._clean_index]),
                                  repeat(self.coordinates.lat[self._clean_index]),
                                  repeat(self.extinction[self._clean_index]), repeat(self.variance[self._clean_index]),
                                  repeat(bandwidth), repeat(metric), repeat(nicest)))

        # Unpack results
        map_ext, map_var, map_num, map_rho = list(zip(*mp))

        # reshape
        map_ext = np.array(map_ext).reshape(grid_lon.shape).astype(np.float32)
        map_var = np.array(map_var).reshape(grid_lon.shape).astype(np.float32)
        map_num = np.array(map_num).reshape(grid_lon.shape).astype(np.uint32)
        map_rho = np.array(map_rho).reshape(grid_lon.shape).astype(np.float32)

        # Return extinction map instance
        return ExtinctionMap(ext=map_ext, var=map_var, num=map_num, rho=map_rho, header=grid_header, metric=metric)

    # ----------------------------------------------------------------------
    def save_fits(self, path):
        """
        Write the extinction data to a FITS table file.

        Parameters
        ----------
        path : str
            File path; e.g. "/path/to/table.fits"

        """

        # Create FITS columns
        col1 = fits.Column(name="Lon", format="D", array=self.coordinates.lon)
        col2 = fits.Column(name="Lat", format="D", array=self.coordinates.lat)
        col3 = fits.Column(name="Extinction", format="E", array=self.extinction)
        col4 = fits.Column(name="Variance", format="E", array=self.variance)

        # Column definitions
        cols = fits.ColDefs([col1, col2, col3, col4])

        # Create binary table object
        tbhdu = fits.BinTableHDU.from_columns(cols)

        # Write to file
        tbhdu.writeto(path, clobber=True)


# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #
class ExtinctionMap:

    def __init__(self, ext, var, header, metric=None, num=None, rho=None):
        """
        Extinction map class.

        Parameters
        ----------
        ext : np.ndarray
            2D Extintion map.
        var : np.ndarray
            2D Extinction variance map.
        header : astropy.fits.Header
            Header of grid from which extinction map was built.
        metric : str, optional
            Metric used to create the map.
        num : np.ndarray, optional
            2D source count map.
        rho : np.ndarray, optional
            2D source density map.

        """

        # Set instance attributes
        self.map = ext
        self.var = var
        self.num = np.full_like(self.map, fill_value=np.nan, dtype=np.uint32) if num is None else num
        self.rho = np.full_like(self.map, fill_value=np.nan, dtype=np.float32) if num is None else rho

        # Other parameters
        self.metric = metric
        self.shape = self.map.shape
        self.fits_header = header

        # Sanity check
        if (self.map.ndim != 2) | (self.var.ndim != 2) | (self.num.ndim != 2) | (self.rho.ndim != 2):
            raise TypeError("Input must be 2D arrays")

    @staticmethod
    def _get_vlim(data, percentiles, r=10):
        vmin = np.floor(np.percentile(data[np.isfinite(data)], percentiles[0]) * r) / r
        vmax = np.ceil(np.percentile(data[np.isfinite(data)], percentiles[1]) * r) / r
        return vmin, vmax

    # ----------------------------------------------------------------------
    def plot_map(self, path=None, figsize=10):
        """
        Method to plot extinction map.

        Parameters
        ----------
        path : str, optional
            File path if it should be saved. e.g. "/path/to/image.png". Default is None.
        figsize : int, float, optional
            Figure size for plot. Default is 10.

        """

        # Import
        from matplotlib import pyplot as plt
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=[figsize, 3 * 0.9 * figsize * (self.shape[0] / self.shape[1])])
        grid = GridSpec(ncols=2, nrows=3, bottom=0.1, top=0.9, left=0.1, right=0.9, hspace=0.08, wspace=0,
                        height_ratios=[1, 1, 1, 1], width_ratios=[1, 0.05])

        for idx in range(0, 6, 2):

            ax = plt.subplot(grid[idx], projection=wcsaxes.WCS(self.fits_header))
            cax = plt.subplot(grid[idx + 1])

            # Plot Extinction map
            if idx == 0:
                vmin, vmax = self._get_vlim(data=self.map, percentiles=[1, 99], r=10)
                im = ax.imshow(self.map, origin="lower", interpolation="nearest", cmap="binary", vmin=vmin, vmax=vmax)
                fig.colorbar(im, cax=cax, label="Extinction (mag)")

            # Plot error map
            elif idx == 2:
                vmin, vmax = self._get_vlim(data=np.sqrt(self.var), percentiles=[1, 90], r=100)
                im = ax.imshow(np.sqrt(self.var), origin="lower", interpolation="nearest", cmap="binary", vmin=vmin,
                               vmax=vmax)
                if self.metric == "median":
                    fig.colorbar(im, cax=cax, label="MAD (mag)")
                else:
                    fig.colorbar(im, cax=cax, label="Error (mag)")

            # Plot source count map
            elif idx == 4:
                vmin, vmax = self._get_vlim(data=self.num, percentiles=[1, 99], r=1)
                im = ax.imshow(self.num, origin="lower", interpolation="nearest", cmap="binary", vmin=vmin, vmax=vmax)
                fig.colorbar(im, cax=cax, label="N")

            # Grab axes
            lon, lat = ax.coords[0], ax.coords[1]

            # Add axes labels
            if idx == 4:
                lon.set_axislabel("Longitude")
            lat.set_axislabel("Latitude")

            # Hide tick labels
            if idx != 4:
                lon.set_ticklabel_position("")

        # Save or show figure
        if path is None:
            plt.show()
        else:
            plt.savefig(path, bbox_inches="tight")
        plt.close()

    # ----------------------------------------------------------------------
    def save_fits(self, path):
        """
        Save extinciton map as FITS file.

        Parameters
        ----------
        path : str
            File path; e.g. "/path/to/table.fits"

        """
        # TODO: Add some header information

        # Create HDU list
        # noinspection PyTypeChecker
        hdulist = fits.HDUList([fits.PrimaryHDU(),
                                fits.ImageHDU(data=self.map, header=self.fits_header),
                                fits.ImageHDU(data=self.var, header=self.fits_header),
                                fits.ImageHDU(data=self.num, header=self.fits_header),
                                fits.ImageHDU(data=self.rho, header=self.fits_header)])

        # Write
        hdulist.writeto(path, clobber=True)


# ----------------------------------------------------------------------
def _get_weight_func(metric, bandwidth):
    # TODO: Add docstrings.

    if metric == "uniform":
        def wfunc(wdis):
            """
            Returns
            -------
            float, np.ndarray
            """
            return np.ones_like(wdis)

    elif metric == "triangular":
        def wfunc(wdis):
            return 1 - np.abs(wdis / bandwidth)

    elif metric == "gaussian":
        def wfunc(wdis):
            return np.exp(-0.5 * (wdis / bandwidth) ** 2)

    elif metric == "epanechnikov":
        def wfunc(wdis):
            return 1 - (wdis / bandwidth) ** 2

    else:
        raise TypeError("metric {0:s} not implemented".format(metric))

    return wfunc


# ----------------------------------------------------------------------
def get_extinction_pixel(lon_grid, lat_grid, lon_sources, lat_sources, ext, var, bandwidth, metric, nicest):
    """
    Calculate extinction for a given grid point.

    Parameters
    ----------
    lon_grid : int, float
        X grid point (longitude).
    lat_grid : int, float
        Y grid point (latitude).
    lon_sources : np.ndarray
        X data (longitudes for all sources).
    lat_sources : np.ndarray
        Y data (latitudes for all source).
    ext : np.ndarray
        Extinction data for each source.
    var : np.ndarray
        Variance data for each source.
    bandwidth : int, float
        Bandwidth of kernel.
    metric : str
        Method to be used. e.g. 'median', 'gaussian', 'epanechnikov', 'uniform', 'triangular'.
    nicest : bool
        Wether or not to use NICEST weight adjustment.

    Returns
    -------
    tuple

    """

    # Define bad pixel return
    bad_return = (np.nan, np.nan, 0, np.nan)

    # In case the average or median is to be calculated, set the truncation scale equal to the bandwidth
    trunc_scale = bandwidth if (metric == "average") | (metric == "median") else 5 * bandwidth

    # Truncate input data to a more managable size
    index = (lon_sources > lon_grid - trunc_scale) & (lon_sources < lon_grid + trunc_scale) & \
            (lat_sources > lat_grid - trunc_scale) & (lat_sources < lat_grid + trunc_scale)

    # If we have nothing here, immediately return
    if np.sum(index) == 0:
        return bad_return

    # Apply pre-filtering
    ext, var, lon_sources, lat_sources = ext[index], var[index], lon_sources[index], lat_sources[index]

    # Calculate the distance to the grid point in a spherical metric
    dis = distance_sky(lon1=lon_sources, lat1=lat_sources, lon2=lon_grid, lat2=lat_grid, unit="degrees")

    # Get sources within truncation scale
    index = dis < trunc_scale / 2

    # There must be at least two sources within the truncation scale which have extinction data
    if np.sum(np.isfinite(ext[index])) < 2:
        return bad_return

    # Calulate number of sources left over after truncation
    npixel = np.sum(index)

    # If nothing remains, return empty pixel
    if npixel == 0:
        return bad_return

    # Get data within truncation radius
    ext, var, dis = ext[index], var[index], dis[index]

    # Based on chosen metric calculate extinction or spatial weights
    if metric == "average":
        return np.nanmean(ext), np.sqrt(np.nansum(var)) / npixel, npixel, np.nan

    elif metric == "median":
        pixel_ext = np.nanmedian(ext)
        return pixel_ext, np.median(np.abs(ext - pixel_ext)), npixel, np.nan

    # If not median or average, fetch weight function
    else:
        wfunc = _get_weight_func(metric=metric, bandwidth=bandwidth)

    # Set parameters for density correction
    # TODO: This needs to be generalised
    alpha, k_lambda = 0.33, 1
    beta = np.log(10) * alpha * k_lambda

    # Get spatial weights:
    weights_spatial = wfunc(wdis=dis)

    weights_spatial[weights_spatial < 0] = 0

    # Get approximate integral and normalize weights
    dummy = np.arange(-100, 100, 0.01)
    weights_spatial_norm = np.divide(weights_spatial, np.trapz(y=wfunc(dummy), x=dummy))

    # TODO: Check if NICEST should modify this
    # Get density map
    rho = np.sum(weights_spatial_norm)

    # Calculate total weight including the variance of the extinction measurement
    weights = weights_spatial / var

    # Modify weights for NICEST
    if nicest:
        weights *= 10 ** (alpha * k_lambda * ext)

    # Ignore warnings for calculating the extinction and variance
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Get extinction for this pixel based on weights
        pixel_ext = np.nansum(weights * ext) / np.nansum(weights)

        # Calculate variance with NICEST weights
        if nicest:

            # Calculate correction factor (Equ. 34 in NICEST paper)
            cor = beta * np.nansum(weights * var) / np.nansum(weights)

            # Calculate error for NICEST (private communication with M. Lombardi)
            # TODO: Check if this makes sense
            pixel_var = (np.sum((weights**2 * np.exp(2*beta*ext) * (1 + beta + ext)**2) / var) /
                         np.sum(weights * np.exp(beta * ext) / var)**2)

        # Without NICEST the variance is a normal weighted error
        else:
            pixel_var = np.nansum(weights ** 2 * var) / np.nansum(weights) ** 2
            cor = 0.

    # Return
    return pixel_ext - cor, pixel_var, npixel, rho
