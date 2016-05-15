# ----------------------------------------------------------------------
# Import stuff
import warnings

import wcsaxes
import numpy as np

from astropy.io import fits
from itertools import repeat
from matplotlib import pyplot as plt
from multiprocessing.pool import Pool
from matplotlib.gridspec import GridSpec

from pnicer.utils import distance_sky


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# noinspection PyProtectedMember
class Extinction:

    def __init__(self, db, extinction, variance=None, color0=None):
        """
        Class for extinction measurements.

        Parameters
        ----------
        db : pnicer.common.DataBase
            Base class from which the extinction was derived.
        extinction : np.ndarray
            Extinction data.
        variance : np.ndarray, optional
            Variance in extinction.
        color0 : np.ndarray, optional
            Intrisic color set for each source.

        """

        # Avoid circular import
        from pnicer.common import DataBase

        # Check if db is really a DataBase instance
        if not isinstance(db, DataBase):
            raise ValueError("passed instance is not DataBase class")

        # Define inititial attributes
        self.db = db
        self.extinction = extinction

        # Set variance to 0 if not given.
        self.variance = variance
        if self.variance is None:
            self.variance = np.zeros_like(extinction)

        # Set intrinsic colors to 0 if not given.
        self.color0 = color0
        if self.color0 is None:
            self.color0 = np.zeros_like(extinction)

        # Index with clean extinction data
        self.clean_index = np.isfinite(self.extinction)

        # Extinction and variance must have same length
        if len(self.extinction) != len(self.variance):
            raise ValueError("Extinction and variance arrays must have equal length")

        # Calculate de-reddened features
        self.features_dered = [f - self.extinction * v for f, v in zip(self.db.features, self.db.extvec.extvec)]

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
    #                            Instance methods                            #
    # ---------------------------------------------------------------------- #

    # ----------------------------------------------------------------------
    def build_map(self, bandwidth, metric="median", frame="galactic", sampling=2, nicest=False, use_fwhm=False):
        """
        Method to build an extinction map.

        Parameters
        ----------
        bandwidth : int, float
            Resolution of output map.
        metric : str, optional
            Metric to be used. e.g. 'median', 'gaussian', 'epanechnikov', 'uniform, 'triangular'. Default is 'median'.
        frame : str, optional
            Reference frame; 'galactic' or 'equatorial'. Default is 'galactic'
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
        assert isinstance(sampling, int), "sampling must be an integer"
        assert (frame == "galactic") | (frame == "equatorial"), "frame must be either 'galactic' or 'equatorial'"

        # Determine pixel size
        pixsize = bandwidth / sampling

        # In case of a gaussian, we can use the fwhm instead
        if use_fwhm:
            # TODO: Check if assertions works in installed software
            assert metric == "gaussian", "FWHM only valid for gaussian kernel"
            bandwidth /= 2 * np.sqrt(2 * np.log(2))

        # First let's get a grid
        grid_header, grid_lon, grid_lat = self.db._build_wcs_grid(frame=frame, pixsize=pixsize)

        # Set some header keywords
        grid_header["BWIDTH"] = (bandwidth, "Bandwidth of kernel (degrees)")
        if use_fwhm:
            grid_header["FWHM"] = (bandwidth * 2 * np.sqrt(2 * np.log(2)), "FWHM of gaussian (degrees)")

        # Run extinction mapping for each pixel
        with Pool() as pool:
            # Submit tasks
            mp = pool.starmap(get_extinction_pixel,
                              zip(grid_lon.ravel(), grid_lat.ravel(),
                                  repeat(self.db._lon[self.clean_index]), repeat(self.db._lat[self.clean_index]),
                                  repeat(self.extinction[self.clean_index]), repeat(self.variance[self.clean_index]),
                                  repeat(bandwidth), repeat(metric), repeat(use_fwhm), repeat(nicest)))

        # Unpack results
        map_ext, map_var, map_num, map_rho = list(zip(*mp))

        # reshape
        map_ext = np.array(map_ext).reshape(grid_lon.shape).astype(np.float32)
        map_var = np.array(map_var).reshape(grid_lon.shape).astype(np.float32)
        map_num = np.array(map_num).reshape(grid_lon.shape)
        map_rho = np.array(map_rho).reshape(grid_lon.shape).astype(np.float32)

        # Return extinction map instance
        return ExtinctionMap(ext=map_ext, var=map_var, num=map_num, dens=map_rho, header=grid_header, metric=metric)

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
        col1 = fits.Column(name="Lon", format='D', array=self.db._lon)
        col2 = fits.Column(name="Lat", format='D', array=self.db._lat)
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

    def __init__(self, ext, var, header, metric=None, num=None, dens=None):
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

        Returns
        -------

        """

        self.map = ext
        self.var = var
        # Number map for each pixel
        if num is None:
            self.num = np.full_like(self.map, fill_value=np.nan, dtype=np.float32)
        else:
            self.num = num
        # Density map from kernel estimation
        if num is None:
            self.rho = np.full_like(self.map, fill_value=np.nan, dtype=np.float32)
        else:
            self.rho = dens

        # Other parameters
        self.metric = metric
        self.shape = self.map.shape
        self.fits_header = header

        # Input must be 2D
        if (len(self.map.shape) != 2) | (len(self.var.shape) != 2) | (len(self.num.shape) != 2):
            raise TypeError("Input must be 2D arrays")

    # ----------------------------------------------------------------------
    def plot_map(self, path=None, figsize=5):
        """
        Method to plot extinction map.

        Parameters
        ----------
        path : str, optional
            File path if it should be saved. e.g. "/path/to/image.png". Default is None.
        figsize : int, float, optional
            Figure size for plot. Default is 5.

        """

        fig = plt.figure(figsize=[figsize, 3 * 0.9 * figsize * (self.shape[0] / self.shape[1])])
        grid = GridSpec(ncols=2, nrows=3, bottom=0.1, top=0.9, left=0.1, right=0.9, hspace=0.08, wspace=0,
                        height_ratios=[1, 1, 1], width_ratios=[1, 0.05])

        for idx in range(0, 6, 2):

            ax = plt.subplot(grid[idx], projection=wcsaxes.WCS(self.fits_header))
            cax = plt.subplot(grid[idx + 1])

            # Plot Extinction map
            if idx == 0:
                im = ax.imshow(self.map, origin="lower", interpolation="nearest", cmap="binary",
                               vmin=np.floor(np.percentile(self.map[np.isfinite(self.map)], 1) * 10) / 10,
                               vmax=np.ceil(np.percentile(self.map[np.isfinite(self.map)], 99) * 10) / 10)
                fig.colorbar(im, cax=cax, label="Extinction (mag)")

            # Plot variance map
            if idx == 2:
                im = ax.imshow(self.var, origin="lower", interpolation="nearest", cmap="binary",
                               vmin=np.floor(np.percentile(self.var[np.isfinite(self.var)], 1) * 10) / 10,
                               vmax=np.ceil(np.percentile(self.var[np.isfinite(self.var)], 2) * 100) / 100)
                if self.metric == "median":
                    fig.colorbar(im, cax=cax, label="MAD")
                else:
                    fig.colorbar(im, cax=cax, label="Variance")

            if idx == 4:
                im = ax.imshow(self.num, origin="lower", interpolation="nearest", cmap="binary",
                               vmin=np.floor(np.percentile(self.num[np.isfinite(self.num)], 1) * 10) / 10,
                               vmax=np.ceil(np.percentile(self.num[np.isfinite(self.num)], 99) * 10) / 10)
                fig.colorbar(im, cax=cax, label="N")

            # Add axes labels
            lon = ax.coords[0]
            lat = ax.coords[1]
            if idx == 4:
                lon.set_axislabel("Longitude")
            lat.set_axislabel("Latitude")

            # Hide tick labels
            if (idx == 0) | (idx == 2):
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
        # Create and save
        # noinspection PyTypeChecker
        hdulist = fits.HDUList([fits.PrimaryHDU(),
                                fits.ImageHDU(data=self.map, header=self.fits_header),
                                fits.ImageHDU(data=self.var, header=self.fits_header),
                                fits.ImageHDU(data=self.num, header=self.fits_header),
                                fits.ImageHDU(data=self.rho, header=self.fits_header)])
        hdulist.writeto(path, clobber=True)


# ----------------------------------------------------------------------
def get_extinction_pixel(xgrid, ygrid, xdata, ydata, ext, var, bandwidth, metric, use_fwhm, nicest=False):
    """
    Calculate extinction for a given grid point.

    Parameters
    ----------
    xgrid : int, float
        X grid point (longitude).
    ygrid : int, float
        Y grid point (latitude).
    xdata : np.ndarray
        X data (longitudes for all sources).
    ydata : np.ndarray
        Y data (latitudes for all source).
    ext : np.ndarray
        Extinction data for each source.
    var : np.ndarray
        Variance data for each source.
    bandwidth : int, float
        Bandwidth of kernel.
    metric : str
        Method to be used. e.g. 'median', 'gaussian', 'epanechnikov', 'uniform', 'triangular'.
    use_fwhm : bool
        If set, then the bandwidth was specified as FWHM. Here this is used to preserve the truncation.
    nicest : bool, optional
        Wether or not to use NICEST weight adjustment.

    Returns
    -------
    tuple

    """

    # In case the average or median is to be calculated, I set bandwidth == truncation scale
    if (metric == "average") | (metric == "median"):
        trunc = 1
    else:
        trunc = 6
        # In case the bandwidth was specified as FWHM, set truncation scale back to standard deviation
        if use_fwhm:
            trunc *= 2 * np.sqrt(2 * np.log(2))

    # Truncate input data to a more managable size (this step does not detemrmine the final source number)
    index = (xdata > xgrid - trunc * bandwidth) & (xdata < xgrid + trunc * bandwidth) & \
            (ydata > ygrid - trunc * bandwidth) & (ydata < ygrid + trunc * bandwidth)

    # If we have nothing here, immediately return
    if np.sum(index) == 0:
        return np.nan, np.nan, 0, np.nan

    # Apply pre-filtering
    ext, var, xdata, ydata = ext[index], var[index], xdata[index], ydata[index]

    # Calculate the distance to the grid point in a spherical metric
    dis = distance_sky(lon1=xdata, lat1=ydata, lon2=xgrid, lat2=ygrid, unit="degrees")

    # There must be at least three sources within the truncation scale which have extinction data
    if np.sum(np.isfinite(ext[dis < trunc * bandwidth])) < 3:
        return np.nan, np.nan, 0, np.nan

    # Now we truncate the data to the truncation scale (i.e. a circular patch on the sky)
    index = dis < trunc * bandwidth / 2

    # If nothing remains, return
    if np.sum(index) == 0:
        return np.nan, np.nan, 0, np.nan

    # Get data within truncation radius
    ext, var, dis, xdata, ydata = ext[index], var[index], dis[index], xdata[index], ydata[index]

    # Calulate number of sources left over after truncation
    # TODO: Somehow this number is always very high! Check if it is ok in Aladin
    npixel = np.sum(index)

    # Based on chosen metric calculate extinction or spatial weights
    # TODO: For average and median I still return the number of sources in each pixel
    if metric == "average":
        pixel_ext = np.nanmean(ext)
        pixel_var = np.sqrt(np.nansum(var)) / npixel
        return pixel_ext, pixel_var, npixel, np.nan

    elif metric == "median":
        pixel_ext = np.nanmedian(ext)
        pixel_mad = np.median(np.abs(ext - pixel_ext))
        return pixel_ext, pixel_mad, npixel, np.nan

    elif metric == "uniform":
        def wfunc(wdis):
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
        raise TypeError("metric not implemented")

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

    # Get density map
    rho = np.sum(weights_spatial_norm)

    # Calculate total weight including the variance of the extinction measurement
    weights = weights_spatial / var

    # Modify weights for NICEST
    if nicest:
        weights *= 10 ** (alpha * k_lambda * ext)

    # Assertion to not raise editor warnings
    assert isinstance(weights, np.ndarray)

    # Get extinction based on weights
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pixel_ext = np.nansum(weights * ext) / np.nansum(weights)

    # Return
    if nicest:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Calculate correction factor (Equ. 34 in NICEST)
            cor = beta * np.nansum(weights * var) / np.nansum(weights)
            # Calculate error for NICEST
            # TODO: Check if this even makes sense
            pixel_var = (np.sum((weights**2 * np.exp(2*beta*ext) * (1 + beta + ext)**2) / var) /
                         np.sum(weights * np.exp(beta * ext) / var)**2)
        # Return
        return pixel_ext - cor, pixel_var, npixel, rho
    else:
        # Error without NICEST is simply a weighted error
        pixel_var = np.nansum(weights ** 2 * var) / np.nansum(weights) ** 2
        # Return
        return pixel_ext, pixel_var, npixel, rho
