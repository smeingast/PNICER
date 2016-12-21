# -----------------------------------------------------------------------------
# Import stuff
import sys
import time
import warnings
import numpy as np

from copy import copy
from astropy import wcs
from astropy.io import fits
from itertools import repeat
from astropy.table import Table
from multiprocessing.pool import Pool
# noinspection PyPackageRequirements
from sklearn.mixture import GaussianMixture

from pnicer.common import Coordinates
from pnicer.user import Magnitudes, Colors, ApparentMagnitudes, ApparentColors
from pnicer.utils import distance_sky, std2fwhm, centroid_sphere, weighted_avg


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
class IntrinsicProbability(object):

    def __init__(self, features, models, models_norm, sources_index, sources_zp):

        # Set instance attributes
        self.features = features
        self.models = models
        self.models_norm = models_norm
        self.sources_index = sources_index
        self.sources_zp = sources_zp

        # Mask for bad sources
        self._sources_mask = self.sources_index < self.features.n_data

        self.get_extinction(metric="bla")

    # ----------------------------------------------------------------------------- #
    #                             Model attribute tools                             #
    # ----------------------------------------------------------------------------- #

    # -----------------------------------------------------------------------------
    def __get_models_attributes(self, attr):
        """
        Helper method to return GMM attributes.

        Parameters
        ----------
        attr : str
            Attribute name to return.

        Returns
        -------
        iterable
            List of attributes for each unique gaussian mixture model.

        """

        return [getattr(gmm, attr) if isinstance(gmm, GaussianMixture) else np.nan for gmm in self.models]

    # -----------------------------------------------------------------------------
    @property
    def _models_means(self):
        """
        Fetches the means of all gaussian mixture models.

        Returns
        -------
        iterable
            List of means of models.
        """

        return self.__get_models_attributes(attr="means_")

    # -----------------------------------------------------------------------------
    @property
    def _models_variances(self):
        """
        Fetches the variances for all gaussian mixture models.

        Returns
        -------
        iterable
            List of variances of models.
        """

        return self.__get_models_attributes(attr="covariances_")

    # -----------------------------------------------------------------------------
    @property
    def _models_weights(self):
        """
        Fetches the weights for all gaussian mixture models.

        Returns
        -------
        iterable
            List of weights of models.
        """

        return self.__get_models_attributes(attr="weights_")

    # ----------------------------------------------------------------------------- #
    #                            Model evaluation tools                             #
    # ----------------------------------------------------------------------------- #

    # -----------------------------------------------------------------------------
    @property
    def _models_intrinsic_mean(self):
        """
        Returns the intrisic feature values for all models corresponding to the mean of the probability density
        distribution.

        Returns
        -------
        iterable
            List of means for each unique model.

        """

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return [weighted_avg(values=v.ravel(), weights=w.ravel())[0] for v, w in
                    zip(self._models_means, self._models_weights)]

    # -----------------------------------------------------------------------------
    @property
    def _models_intrinsic_max(self):
        """
        Returns the intrisic feature values for all models corresponding to the maximum of the probability density
        distribution.

        Returns
        -------
        iterable
            List of maxima for each unique model.

        """

        # Determine query range
        qrange = [np.min(self._models_means) - 3 * self._models_variances[0][0],
                  np.max(self._models_means) + 3 * self._models_variances[0][0]]

        # Get number of samples (range oversampled by 20 times the variance)
        nsamples = np.uint32(np.ceil(20 * (qrange[1] - qrange[0]) / self._models_variances[0][0]))[0][0]

        # At least 100 samples, maximum 1E5
        nsamples = 100 if nsamples < 100 else nsamples
        nsamples = 100000 if nsamples > 1E5 else nsamples

        # Create query range
        qrange = np.linspace(start=qrange[0][0], stop=qrange[1][0], num=nsamples)

        # Determine maximum and return
        return [qrange[np.argmax(np.exp(g.score_samples(np.expand_dims(qrange, 1))))] for g in self.models]

    # ----------------------------------------------------------------------------- #
    #                               Extinction tools                                #
    # ----------------------------------------------------------------------------- #

    # -----------------------------------------------------------------------------
    def get_extinction(self, metric="mean"):
        """
        Calculates extinction from the probability density distributions based on a given metric.

        Parameters
        ----------
        metric : str, optional
            Metric to use to evaluate the intrinsic feature. Possible are 'mean' (default) and 'max'.

        Returns
        -------
        np.ndarray
            Array of extinction values for each source.

        """

        if metric.lower() == "mean":
            attr = "_models_intrinsic_mean"
        elif metric.lower() == "max":
            attr = "_models_intrinsic_max"
        else:
            raise ValueError("Metric '{0}' not supported".format(metric))

        return ((self.sources_zp[self._sources_mask] -
                 np.array(getattr(self, attr))[self.sources_index[self._sources_mask]]) /
                self.models_norm[self.sources_index[self._sources_mask]])

    # ----------------------------------------------------------------------------- #
    #                                     Misc                                      #
    # ----------------------------------------------------------------------------- #

    # -----------------------------------------------------------------------------
    @property
    def _get_intrinsic_class(self):
        # TODO: Add docstring

        if isinstance(self.features, ApparentMagnitudes):
            return IntrinsicMagnitudes
        elif isinstance(self.features, ApparentColors):
            return IntrinsicColors
        else:
            raise NotImplementedError("'{0}' not supported".format(self.features.__class__))


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# noinspection PyProtectedMember
class Intrinsic(object):

    # -----------------------------------------------------------------------------
    def __init__(self, coordinates, extinction, variance=None, extvec=None):
        """
        Class for Intrisic features and extinction.

        Parameters
        ----------
        coordinates : SkyCoord
            Astropy SkyCoord instance.
        extinction : np.ndarray
            Extinction data.
        variance : np.ndarray, optional
            Variance in extinction.
        extvec : ExtinctionVector, optional
            Extinction Vector instance.

        """

        # Set attributes
        self.coordinates = Coordinates(coordinates=coordinates)
        self.extinction = extinction
        self.variance = np.zeros_like(extinction) if variance is None else variance
        self.extvec = extvec

        # Sanity checks
        if len(self.extinction) != len(self.variance):
            raise ValueError("Extinction and variance arrays must have equal length")

    # -----------------------------------------------------------------------------
    def __len__(self):
        return len(self.extinction)

    # -----------------------------------------------------------------------------
    def __str__(self):
        return Table([np.around(self.extinction, 3), np.around(np.sqrt(self.variance), 3)],
                     names=("Extinction", "Error")).__str__()

    # -----------------------------------------------------------------------------
    def __iter__(self):
        for x in self.extinction:
            yield x

    # -----------------------------------------------------------------------------
    @property
    def _clean_index(self):
        """
        Index of finite extinction measurements.

        Returns
        -------
        np.ndarray

        """

        return np.isfinite(self.extinction)

    # -----------------------------------------------------------------------------
    @staticmethod
    def __build_map_print(silent):
        if not silent:
            print("{:<45}".format("Building extinction map"))
            print("{:-<45}".format(""))
            print("{0:<15}{1:<15}{2:<15}".format("Progress (%)", "Elapsed (s)", "ETA (s)"))
        else:
            pass

    # -----------------------------------------------------------------------------
    def build_map(self, bandwidth, metric="median", sampling=2, nicest=False, alpha=1/3, use_fwhm=False, silent=False,
                  **kwargs):
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
        alpha : float, optional
            The slope in the number counts (NICEST equation 2). Default is 1/3.
        use_fwhm : bool, optional
            If set, the bandwidth parameter represents the gaussian FWHM instead of its standard deviation. Only
            available when using a gaussian weighting.
        silent : bool, optional
            Whether information on the progress should be printed.

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

        # Adjust bandwidth in case FWHM is to be used
        if use_fwhm:
            bandwidth /= std2fwhm

        # Set default projection
        if "proj_code" not in kwargs:
            kwargs["proj_code"] = "TAN"

        # Set k_lambda for nicest
        k_lambda = np.max(self.extvec) if self.extvec is not None else 1

        # Create WCS grid
        grid_header, (grid_lon, grid_lat) = self.coordinates.build_wcs_grid(pixsize=pixsize, **kwargs)

        # Create pixel grid
        grid_x, grid_y = wcs.WCS(grid_header).wcs_world2pix(grid_lon, grid_lat, 0)

        # Grid shape
        grid_shape = grid_x.shape

        # Get pixel coordinates of sources
        sources_x, sources_y = wcs.WCS(grid_header).wcs_world2pix(self.coordinates.lon, self.coordinates.lat, 0)

        # Split into a minimum of ~1x1 deg2 patches
        n = np.ceil(np.min(grid_shape) * pixsize)

        # Determine patch size in pixels
        patch_size = np.ceil(np.min(grid_shape) / n).astype(np.int)

        # Set minimum to 100 pix for effective parallelisation
        if patch_size < 100:
            patch_size = 100

        # But it can't exceed 5 degrees to keep a reasonable number of sources per patch
        if patch_size * pixsize > 5:
            patch_size = 5 / pixsize

        # Number of patches in each dimension of the grid
        np0, np1 = np.int(np.ceil(grid_shape[0] / patch_size)), np.int(np.ceil(grid_shape[1] / patch_size))

        # Create patches
        grid_x_patches = [np.array_split(p, np1, axis=1) for p in np.array_split(grid_x, np0, axis=0)]
        grid_y_patches = [np.array_split(p, np1, axis=1) for p in np.array_split(grid_y, np0, axis=0)]
        grid_lon_patches = [np.array_split(p, np1, axis=1) for p in np.array_split(grid_lon, np0, axis=0)]
        grid_lat_patches = [np.array_split(p, np1, axis=1) for p in np.array_split(grid_lat, np0, axis=0)]

        # Determine total number of patches
        n_patches = len([item for sublist in grid_x_patches for item in sublist])

        # Print info
        self.__build_map_print(silent)

        # Loop over patch rows
        tstart, stack, i, progress = time.time(), [], 0, 0
        for row_x, row_y, row_lon, row_lat in zip(grid_x_patches, grid_y_patches, grid_lon_patches, grid_lat_patches):

            # Loop over patch columns in row
            row = []
            for px, py, plon, plat in zip(row_x, row_y, row_lon, row_lat):

                # Patch shape
                pshape = px.shape

                # Get center of patch
                center_plon, center_plat = centroid_sphere(lon=plon, lat=plat, units="degree")

                # Get maximum distance in patch grid
                pmax = np.max(distance_sky(lon1=center_plon, lat1=center_plat, lon2=plon, lat2=plat, unit="degree"))

                # Extend with kernel bandwidth
                pmax += 3.01 * bandwidth     # This scale connects to the truncation scale in _get_extinction_pixel!!!

                # Get all sources within patch range
                pfil = distance_sky(lon1=center_plon, lat1=center_plat,
                                    lon2=self.coordinates.lon[self._clean_index],
                                    lat2=self.coordinates.lat[self._clean_index], unit="degree") < pmax

                # Filter data
                splon = self.coordinates.lon[self._clean_index][pfil]
                splat = self.coordinates.lat[self._clean_index][pfil]
                sx, sy = sources_x[self._clean_index][pfil], sources_y[self._clean_index][pfil]
                pext, pvar = self.extinction[self._clean_index][pfil], self.variance[self._clean_index][pfil]

                # Run extinction estimation for each pixel
                with Pool() as pool:
                    mp = pool.starmap(_get_extinction_pixel,
                                      zip(plon.ravel(), plat.ravel(), px.ravel(), py.ravel(), repeat(pixsize),
                                          repeat(splon), repeat(splat), repeat(sx), repeat(sy), repeat(pext),
                                          repeat(pvar), repeat(bandwidth), repeat(metric), repeat(nicest),
                                          repeat(alpha), repeat(k_lambda)))

                # Reshape and convert
                row.append([np.array(x).reshape(pshape).astype(d) for
                            x, d in zip(list(zip(*mp)), [np.float32, np.float32, np.uint32, np.float32])])

                # Calcualte ETA
                i += 1
                progress, telapsed = i / n_patches, time.time() - tstart
                remaining = 1 - progress
                tremaining = (telapsed / progress) * remaining

                # Report progress
                try:
                    # Test for notebook
                    # noinspection PyUnresolvedReferences
                    if get_ipython().__class__.__name__ == "ZMQInteractiveShell":

                        # noinspection PyPackageRequirements
                        from IPython.display import display, clear_output
                        clear_output(wait=True)
                        self.__build_map_print(silent)
                        if not silent:
                            print("{0:<15.1f}{1:<15.1f}{2:<15.1f}".format(100 * progress, telapsed, tremaining))
                        sys.stdout.flush()
                    else:
                        raise NameError
                except NameError:
                    end = "" if i < n_patches else "\n"
                    if not silent:
                        print("\r{0:<15.1f}{1:<15.1f}{2:<15.1f}".format(100 * progress, telapsed, tremaining), end=end)

            # Stack into full row
            stack.append([np.hstack(x) for x in list(zip(*row))])

        # Stack into full maps
        full = [np.vstack(x) for x in list(zip(*stack))]

        # Create primary header for map with metric information
        phdr = self._make_prime_header(bandwidth=bandwidth, metric=metric, sampling=sampling, nicest=nicest)

        # Final print
        if not silent:
            print("All done in {0:0.1f}s".format(time.time() - tstart))

        # Return extinction map instance
        return ExtinctionMap(ext=full[0], var=full[1], num=full[2], rho=full[3],
                             map_header=grid_header, prime_header=phdr)

    # -----------------------------------------------------------------------------
    @staticmethod
    def _make_prime_header(bandwidth, metric, sampling, nicest):
        """
        Creates a header with information about extinction mapping.

        Parameters
        ----------
        bandwidth : int, float
            Bandwidth used to create the map.
        metric : str
            Metric used to create the map.
        sampling : int
            Sampling factor of map.
        nicest : bool
            Whether NICEST tuning was activated.

        Returns
        -------
        fits.Header
            Astropy Fits header instance.

        """

        # Create empty header
        header = fits.Header()

        # Add keywords
        header["BWIDTH"] = (bandwidth, "Bandwidth of kernel (degrees)")
        header["METRIC"] = (metric, "Metric used to create this map")
        header["SAMPLING"] = (sampling, "Sampling factor of map")
        header["NICEST"] = (nicest, "Whether NICEST was activated")

        return header

    # -----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
class IntrinsicMagnitudes(Magnitudes, Intrinsic):

    # -----------------------------------------------------------------------------
    def __init__(self, magnitudes, errors, extinction, extinction_variance, extvec, coordinates=None, names=None):
        """
        Intrinsic magnitude data class.

        Parameters
        ----------
        magnitudes : list
            List of magnitude arrays. All arrays must have the same length.
        errors : list
            List off magnitude error arrays.
        extinction : np.ndarray
            Extinction data.
        extinction_variance : np.ndarray, optional
            Variance in extinction.
        extvec : list
            List holding the extinction components for each magnitude.
        coordinates : SkyCoord, optional
            Astropy SkyCoord instance.
        names : list, optional
            List of magnitude (feature) names.

        """

        # Call the constructors explicitly without super
        Magnitudes.__init__(self, magnitudes=magnitudes, errors=errors, extvec=extvec, coordinates=coordinates,
                            names=names)
        Intrinsic.__init__(self, coordinates=coordinates, extinction=extinction, variance=extinction_variance,
                           extvec=extvec)


# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
class IntrinsicColors(Intrinsic, Colors):

    # -----------------------------------------------------------------------------
    def __init__(self, colors, errors, extinction, extinction_variance, extvec, coordinates=None, names=None):
        """
        Intrinsic color data class.

        Parameters
        ----------
        colors : list
            List of magnitude arrays. All arrays must have the same length.
        errors : list
            List off magnitude error arrays.
        extinction : np.ndarray
            Extinction data.
        extinction_variance : np.ndarray, optional
            Variance in extinction.
        extvec : list
            List holding the extinction components for each magnitude.
        coordinates : SkyCoord, optional
            Astropy SkyCoord instance.
        names : list, optional
            List of magnitude (feature) names.

        """

        # Call the constructors explicitly without super
        Colors.__init__(self, colors=colors, errors=errors, extvec=extvec, coordinates=coordinates, names=names)
        Intrinsic.__init__(self, coordinates=coordinates, extinction=extinction, variance=extinction_variance,
                           extvec=extvec)


# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
class ExtinctionMap:

    # -----------------------------------------------------------------------------
    def __init__(self, ext, var, map_header, prime_header=None, num=None, rho=None):
        """
        Extinction map class.

        Parameters
        ----------
        ext : np.ndarray
            2D Extintion map.
        var : np.ndarray
            2D Extinction variance map.
        map_header : astropy.fits.Header
            Header of grid from which extinction map was built.
        num : np.ndarray, optional
            2D source count map.
        rho : np.ndarray, optional
            2D source density map.

        """

        # Set instance attributes
        self.map, self.var = ext, var
        self.num = np.full_like(self.map, fill_value=np.nan, dtype=np.uint32) if num is None else num
        self.rho = np.full_like(self.map, fill_value=np.nan, dtype=np.float32) if num is None else rho
        self.prime_header = fits.Header() if prime_header is None else prime_header
        self.map_header = map_header

        # Sanity check for dimensions
        if (self.map.ndim != 2) | (self.var.ndim != 2) | (self.num.ndim != 2) | (self.rho.ndim != 2):
            raise TypeError("Input must be 2D arrays")

    # -----------------------------------------------------------------------------
    @property
    def shape(self):
        return self.map.shape

    # -----------------------------------------------------------------------------
    @staticmethod
    def _get_vlim(data, percentiles, r=10):
        vmin = np.floor(np.percentile(data[np.isfinite(data)], percentiles[0]) * r) / r
        vmax = np.ceil(np.percentile(data[np.isfinite(data)], percentiles[1]) * r) / r
        return vmin, vmax

    # -----------------------------------------------------------------------------
    # noinspection PyUnresolvedReferences
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
        import matplotlib
        from matplotlib import pyplot as plt

        # If the density should be plotted, set to 4
        nfig = 3

        fig = plt.figure(figsize=[figsize, nfig * 0.9 * figsize * (self.shape[0] / self.shape[1])])
        grid = matplotlib.gridspec.GridSpec(ncols=2, nrows=nfig, bottom=0.1, top=0.9, left=0.1, right=0.9, hspace=0.08,
                                            wspace=0, height_ratios=[1] * nfig, width_ratios=[1, 0.05])

        # Set cmap
        cmap = copy(matplotlib.cm.binary)
        cmap.set_bad("#DC143C", 1.)

        for idx in range(0, nfig * 2, 2):

            ax = plt.subplot(grid[idx], projection=wcs.WCS(self.map_header))
            cax = plt.subplot(grid[idx + 1])

            # Plot Extinction map
            if idx == 0:
                vmin, vmax = self._get_vlim(data=self.map, percentiles=[0.1, 90], r=100)
                im = ax.imshow(self.map, origin="lower", interpolation="nearest", vmin=vmin, vmax=vmax, cmap=cmap)
                fig.colorbar(im, cax=cax, label="Extinction (mag)")

            # Plot error map
            elif idx == 2:
                vmin, vmax = self._get_vlim(data=np.sqrt(self.var), percentiles=[1, 90], r=100)
                im = ax.imshow(np.sqrt(self.var), origin="lower", interpolation="nearest", vmin=vmin, vmax=vmax,
                               cmap=cmap)
                if self.prime_header["METRIC"] == "median":
                    fig.colorbar(im, cax=cax, label="MAD (mag)")
                else:
                    fig.colorbar(im, cax=cax, label="Error (mag)")

            # Plot source count map
            elif idx == 4:
                vmin, vmax = self._get_vlim(data=self.num, percentiles=[1, 99], r=1)
                im = ax.imshow(self.num, origin="lower", interpolation="nearest", vmin=vmin, vmax=vmax, cmap=cmap)
                fig.colorbar(im, cax=cax, label="N")

            elif idx == 6:
                vmin, vmax = self._get_vlim(data=self.rho, percentiles=[1, 99], r=1)
                im = ax.imshow(self.rho, origin="lower", interpolation="nearest", vmin=vmin, vmax=vmax, cmap=cmap)
                fig.colorbar(im, cax=cax, label=r"$\rho$")

            # Grab axes
            lon, lat = ax.coords[0], ax.coords[1]

            # Add axes labels
            if idx == (nfig - 1) * 2:
                lon.set_axislabel("Longitude")
            lat.set_axislabel("Latitude")

            # Hide tick labels
            if idx != (nfig - 1) * 2:
                lon.set_ticklabel_position("")

        # Save or show figure
        if path is None:
            plt.show()
        else:
            plt.savefig(path, bbox_inches="tight")
        plt.close()

    # -----------------------------------------------------------------------------
    def save_fits(self, path, clobber=True):
        """
        Save extinciton map as FITS file.

        Parameters
        ----------
        path : str
            File path. e.g. "/path/to/table.fits".
        clobber : bool, optional
            Whether to overwrite exisiting files. Default is True.

        """

        # Create HDU list
        # noinspection PyTypeChecker
        hdulist = fits.HDUList([fits.PrimaryHDU(header=self.prime_header),
                                fits.ImageHDU(data=self.map, header=self.map_header),
                                fits.ImageHDU(data=self.var, header=self.map_header),
                                fits.ImageHDU(data=self.num, header=self.map_header),
                                fits.ImageHDU(data=self.rho, header=self.map_header)])

        # Write
        hdulist.writeto(path, clobber=clobber)


# -----------------------------------------------------------------------------
def _get_weight_func(metric, bandwidth):
    """
    Defines the weight function for extinction mapping.

    Parameters
    ----------
    metric : str
        The mtric to be used. One of 'uniform', 'triangular', 'gaussian', 'epanechnikov'
    bandwidth : int, float
        Bandwidth of metric (kernel).

    """

    if metric == "uniform":
        def wfunc(wdis):
            """
            Returns
            -------
            float, np.ndarray

            """
            return np.ones_like(wdis)

    elif metric == "triangular":
        # noinspection PyUnresolvedReferences
        def wfunc(wdis):
            val = 1 - np.abs(wdis / bandwidth)
            val[val < 0] = 0
            return val

    elif metric == "gaussian":
        def wfunc(wdis):
            return np.exp(-0.5 * (wdis / bandwidth) ** 2)

    elif metric == "epanechnikov":
        # noinspection PyUnresolvedReferences
        def wfunc(wdis):
            val = 1 - (wdis / bandwidth) ** 2
            val[val < 0] = 0
            return val

    else:
        raise TypeError("metric {0:s} not implemented".format(metric))

    return wfunc


# -----------------------------------------------------------------------------
# noinspection PyTypeChecker
def _get_extinction_pixel(lon_grid, lat_grid, x_grid, y_grid, pixsize, lon_sources, lat_sources, x_sources, y_sources,
                          extinction, variance, bandwidth, metric, nicest, alpha, k_lambda):
    """
    Calculate extinction for a given grid point.

    Parameters
    ----------
    lon_grid : int, float
        X grid point (longitude).
    lat_grid : int, float
        Y grid point (latitude).
    x_grid : int, float
        X grid point (x coordinate in grid).
    y_grid : int, float
        Y grid point (Y coordinate in grid).
    pixsize : int, float
        Pixel size in degrees.
    lon_sources : np.ndarray
        X data (longitudes for all sources).
    lat_sources : np.ndarray
        Y data (latitudes for all source).
    x_sources : np.array
        X data (X coordinates in grid).
    y_sources : np.array
        Y data (Y coordinates in grid).
    extinction : np.ndarray
        Extinction data for each source.
    variance : np.ndarray
        Variance data for each source.
    bandwidth : int, float
        Bandwidth of kernel.
    metric : str
        Method to be used. e.g. 'median', 'gaussian', 'epanechnikov', 'uniform', 'triangular'.
    nicest : bool
        Wether or not to use NICEST weight adjustment.
    alpha : int, float
        Slope of source counts (NICEST equation 2).
    k_lambda : int, float
        Extinction law in considered band (NICEST equation 2).

    Returns
    -------
    tuple

    """

    # Set truncation scale
    trunc_deg = bandwidth if (metric == "average") | (metric == "median") else 6 * bandwidth
    trunc_pix = trunc_deg / pixsize / 2

    # Truncate input sources to a more manageable size
    idx = ((x_sources < x_grid + trunc_pix) & (x_sources > x_grid - trunc_pix) &
           (y_sources < y_grid + trunc_pix) & (y_sources > y_grid - trunc_pix) & np.isfinite(extinction))

    # Return if no data
    if np.sum(idx) == 0:
        return np.nan, np.nan, 0, np.nan

    # Apply pre-filtering to sky coordinates
    lon, lat, ext, var = lon_sources[idx], lat_sources[idx], extinction[idx], variance[idx]

    # Calculate the distance to the grid point on a sphere for the filtered sources
    dis = distance_sky(lon1=lon, lat1=lat, lon2=lon_grid, lat2=lat_grid, unit="degrees")

    # Get sources within truncation scale on the sky
    idx = dis < trunc_deg / 2

    # Calulate remaining number of sources after truncation
    nsources = np.sum(idx)

    # Return if there are less than 2 sources after filtering
    if nsources < 2:
        return np.nan, np.nan, 0, np.nan

    # Get data within truncation radius on sky
    ext, var, dis = ext[idx], var[idx], dis[idx]

    # Conditional choice for different metrics
    if metric == "average":

        # TODO: Check error calculations
        # return np.nanmean(ext), np.nanvar(ext), nsources, np.nan

        # 3 sig filter
        sigfil = np.abs(ext - np.mean(ext)) < 3 * np.std(ext)
        nsources = np.sum(sigfil)

        # Return
        return np.mean(ext[sigfil]), np.sqrt(np.sum(var[sigfil])) / nsources, nsources, np.nan

    elif metric == "median":
        pixel_ext = np.median(ext)
        pixel_mad = np.median(np.abs(ext - pixel_ext))
        return pixel_ext, pixel_mad, nsources, np.nan

    # If not median or average, fetch weight function
    else:
        wfunc = _get_weight_func(metric=metric, bandwidth=bandwidth)

    # Set parameters for density correction
    beta = np.log(10) * alpha * k_lambda

    # Get weights:
    w_theta = wfunc(wdis=dis)   # Spatial weight only
    w_total = w_theta / var     # Weighted by variance of sources

    # Get approximate integral and normalize weights
    w_theta = np.divide(w_theta, np.trapz(y=wfunc(np.arange(-100, 100, 0.01)), x=np.arange(-100, 100, 0.01)))

    # Modify weights for NICEST
    if nicest:
        w_theta *= 10 ** (alpha * k_lambda * ext)
        w_total *= 10 ** (alpha * k_lambda * ext)

    # Do sigma clipping in extinction
    pixel_ext = np.sum(w_total * ext) / np.sum(w_total)
    sigfil = np.abs(ext - pixel_ext) < 3 * np.std(ext)

    # Apply sigma clipping to all variables
    ext, var, w_theta, w_total, nsources = ext[sigfil], var[sigfil], w_theta[sigfil], w_total[sigfil], np.sum(sigfil)

    # Get final extinction
    pixel_ext = np.sum(w_total * ext) / np.sum(w_total)

    # Get density
    rho = np.sum(w_theta)

    # Get variance
    if nicest:

        # Correction factor (Equ. 34 in NICEST paper)
        cor = beta * np.sum(w_total * var) / np.sum(w_total)

        # Calculate error for NICEST (private communication with M. Lombardi)
        pixel_var = (np.sum((w_total ** 2 * np.exp(2 * beta * ext) * (1 + beta * ext) ** 2) / var) /
                     np.sum(w_total * np.exp(beta * ext) / var) ** 2)

    # Without NICEST the variance is calculated as a normal weighted error
    else:
        pixel_var = np.sum(w_total ** 2 * var) / np.sum(w_total) ** 2
        cor = 0.

    # Return
    return pixel_ext - cor, pixel_var, nsources, rho
