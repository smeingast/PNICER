# -----------------------------------------------------------------------------
# Import stuff
import numpy as np

from copy import copy
from astropy import wcs
from astropy.io import fits
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d

from pnicer.utils.algebra import round_partial
from pnicer.utils.gmm import gmm_expected_value, gmm_population_variance, mp_gmm_score_samples_absolute, gmm_query_range


# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
class ExtinctionMap:

    # -----------------------------------------------------------------------------
    def __init__(self, map_ext, map_header=None, prime_header=None,):

        # Map
        self.map_ext = map_ext

        # Headers
        self.prime_header = fits.Header() if prime_header is None else prime_header
        self.map_header = fits.Header() if map_header is None else map_header

    # -----------------------------------------------------------------------------
    @property
    def map_shape(self):
        """ Shape of map array. """
        return self.map_ext.shape

    # -----------------------------------------------------------------------------
    # noinspection PyTypeChecker
    @property
    def map_mask(self):
        """ Mask for bad entries in map. """
        try:
            # TODO: Check this
            return np.isnan(self.map_ext)
        except TypeError:
            return np.equal(self.map_ext, None)


# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
class ContinuousExtinctionMap(ExtinctionMap):

    # -----------------------------------------------------------------------------
    def __init__(self, map_models, map_header, prime_header=None):

        # Set instance attributes
        super(ContinuousExtinctionMap, self).__init__(map_ext=map_models, map_header=map_header,
                                                      prime_header=prime_header)

    # -----------------------------------------------------------------------------
    @property
    def _models(self):
        """ All models in the map. """
        return self.map_ext[~self.map_mask]

    # -----------------------------------------------------------------------------
    @property
    def _models_components_means(self):
        """ Means of all components of all models. """
        return [m.means_ for m in self._models]

    # -----------------------------------------------------------------------------
    @property
    def _models_components_variances(self):
        """ Means of all components of all models. """
        return [m.covariances_ for m in self._models]

    # -----------------------------------------------------------------------------
    @property
    def _models_components_weights(self):
        """ Means of all components of all models. """
        return [m.weights_ for m in self._models]

    # -----------------------------------------------------------------------------
    def _extinction_range(self, kappa=3):
        """
        Determine the full significant ranges of extinction the map covers.

        Parameters
        ----------
        kappa : int, float, optional
            The query width in standard deviations of the components. Default is 3.

        Returns
        -------
        tuple
            Range of the extinction in the map (min, max).

        """

        # Query all models for ranges
        test = [gmm_query_range(gmm=g, kappa=kappa, means=m, variances=v) for g, m, v in
                zip(self._models, self._models_components_means, self._models_components_variances)]

        # Repack results
        qmin, qmax = zip(*test)

        # Return min and max ranges
        return np.min(qmin), np.max(qmax)

    # -----------------------------------------------------------------------------
    def _models_set_expected_value(self):
        """ Set expected value for all models as attribute. """
        for gmm in self.map_ext.ravel():
            if gmm is not None:
                setattr(gmm, "expected_value", gmm_expected_value(gmm=gmm, method="weighted"))

    # -----------------------------------------------------------------------------
    def _models_set_population_variance(self):
        """ Set variance for all models as attribute. """
        for gmm in self.map_ext.ravel():
            if gmm is not None:
                setattr(gmm, "population_variance", gmm_population_variance(gmm=gmm, method="weighted"))

    # ----------------------------------------------------------------------------- #
    #                               Map contructors                                 #
    # ----------------------------------------------------------------------------- #

    # -----------------------------------------------------------------------------
    def __map_attr(self, attr, dtype=None):
        """
        Build map from model attributes

        Parameters
        ----------
        attr : str
            Model attribute to build the map from.
        dtype : optional
            Output data type

        Returns
        -------
        np.ndarray
            Map array built from attribute

        """

        # Initialize empty map
        map_attr = np.full_like(self.map_ext, fill_value=0, dtype=dtype)

        # Fill map
        for idx in range(self.map_ext.size):

            if self.map_ext.ravel()[idx] is not None:
                map_attr.ravel()[idx] = getattr(self.map_ext.ravel()[idx], attr)
            else:
                map_attr.ravel()[idx] = 0

        return map_attr

    # -----------------------------------------------------------------------------
    @property
    def map_num(self):
        """ Map with number of sources used for each pixel. """
        return self.__map_attr(attr="n_models", dtype=np.uint32)

    # -----------------------------------------------------------------------------
    @property
    def map_ncomponents(self):
        """ Map with number of GMM components in each pixel. """
        return self.__map_attr(attr="n_components", dtype=np.uint32)

    # -----------------------------------------------------------------------------
    @property
    def map_expected_value(self):
        """ Map with expected value of models. """
        try:
            return self.__map_attr(attr="expected_value", dtype=np.uint32)
        except AttributeError:
            self._models_set_expected_value()
            return self.__map_attr(attr="expected_value", dtype=np.float32)

    # -----------------------------------------------------------------------------
    @property
    def map_variance(self):
        """ Map with population variances of models. """
        try:
            return self.__map_attr(attr="population_variance", dtype=np.float32)
        except AttributeError:
            self._models_set_population_variance()
            return self.__map_attr(attr="population_variance", dtype=np.float32)

    # ----------------------------------------------------------------------------- #
    #                               Cube contructors                                #
    # ----------------------------------------------------------------------------- #

    # -----------------------------------------------------------------------------
    def _map2cube_header(self, naxis3, crval3, crpix3, cdelt3, ctype3=None):
        """
        Creates a valid cube FITS header based on the given extinction map header.

        Parameters
        ----------
        naxis3 : int
            Length of third axis.
        crval3 : int, float
            CRVAL of third axis.
        crpix3 : int, float
            CRPIX of third axis.
        cdelt3 : int, float
            CDELT of third axis.
        ctype3 : str, optional
            CTYPE of third axis.

        Returns
        -------
        Header
            New FITS header instance for cube.

        """

        # Get map header and modify NAXIS
        cube_header = self.map_header.copy()
        cube_header["NAXIS"] = 3

        # Add cards
        cube_header.insert("NAXIS2", fits.Card(keyword="NAXIS3", value=naxis3), after=True)
        cube_header.insert("CRVAL2", fits.Card(keyword="CRVAL3", value=crval3), after=True)
        cube_header.insert("CRPIX2", fits.Card(keyword="CRPIX3", value=crpix3), after=True)
        cube_header.insert("CDELT2", fits.Card(keyword="CDELT3", value=cdelt3), after=True)
        if ctype3 is not None:
            cube_header.insert("CTYPE2", fits.Card(keyword="CTYPE3", value=ctype3), after=True)

        return cube_header

    # -----------------------------------------------------------------------------
    _cube_prob_dens_full = None

    def __cube_prob_dens_full(self, qstep=0.05):
        """
        Constructs a cube of probability densities across the entire significant extinction range.

        Parameters
        ----------
        qstep : float, optional
            Cube step size in extinction, Default is 0.05.

        Returns
        -------
        ExtinctionCube

        """

        # Fast return if already set
        if isinstance(self._cube_prob_dens_full, ExtinctionCube):
            return self._cube_prob_dens_full

        # Get full query range of GMM map
        qmin, qmax = round_partial(np.array(self._extinction_range(kappa=3)), precision=qstep)

        # Score samples for given range
        samples = mp_gmm_score_samples_absolute(gmms=self._models, xmin=qmin, xmax=qmax, xstep=qstep)
        nsamples = len(samples[0])

        # Construct cube FITS header (CRPIX3 since FITS convention starts with pixel 1 not 0)
        header = self._map2cube_header(naxis3=nsamples, crval3=qmin, crpix3=1, cdelt3=qstep, ctype3="extinction")

        # Create cube with probability densities
        cube = np.full((self.map_ext.size, nsamples), fill_value=np.nan, dtype=np.float32)
        cube[np.nonzero(~self.map_mask.ravel()), :] = np.vstack(samples)
        cube = cube.reshape(*self.map_shape, -1)
        cube = np.rollaxis(cube, 2, 0)

        # Save the cube
        self._cube_prob_dens_full = ExtinctionCube(cube=cube, header=header)
        return self._cube_prob_dens_full

    # -----------------------------------------------------------------------------
    def __cube_probability_density(self, ext_min=0, ext_max=2, ext_step=0.1):
        """
        Constructs a cube of probability densities for each pixel in the map.

        Parameters
        ----------
        ext_min : int, float, optional
            Minimum extinction to sample.
        ext_max : int, float, optional
            Maximum extinction to sample.
        ext_step : int, float, optional
            Step size in extinction.

        Returns
        -------
        ExtinctionCube

        """

        # Get full query range of GMM map
        cube_pd = self.__cube_prob_dens_full()

        # Construct interpolator for probability density
        f = interp1d(cube_pd.crange3, cube_pd.cube, axis=0, fill_value=0)

        # Get actual query range
        cube = f(np.arange(ext_min, ext_max + ext_step / 2, step=ext_step))

        # Modify reference value in header
        header = cube_pd.header.copy()
        header["CRPIX3"] = 1
        header["CRVAL3"] = ext_min
        header["CDELT3"] = ext_step

        # Return
        return ExtinctionCube(cube=cube, header=header)

    # -----------------------------------------------------------------------------
    def __cube_probability_max(self, ext_min=0, ext_max=2, ext_step=0.1):
        """
        Constructs a cube of probabilities for the maximum extinction. Each pixel then shows the probability of having
        a certain amount of maximum extinction. The extinction is represented in the third axis.

        Parameters
        ----------
        ext_min : int, float, optional
            Minimum extinction to sample.
        ext_max : int, float, optional
            Maximum extinction to sample.
        ext_step : int, float, optional
            Step size in extinction.

        Returns
        -------
        ExtinctionCube

        """

        # Get full query range of GMM map
        cube_pd = self.__cube_prob_dens_full()

        # Construct cumulative integral
        # noinspection PyTypeChecker
        cube = cumtrapz(cube_pd.cube, dx=cube_pd.header["CDELT3"], axis=0, initial=0)

        # Construct interpolator for probability density
        f = interp1d(cube_pd.crange3, cube, axis=0, fill_value=0)

        # Get actual query range
        cube = f(np.arange(ext_min, ext_max + ext_step / 2, step=ext_step))

        # Modify reference value in header
        header = cube_pd.header.copy()
        header["CRPIX3"] = 1
        header["CRVAL3"] = ext_min
        header["CDELT3"] = ext_step

        # Return
        return ExtinctionCube(cube=cube, header=header)

    # -----------------------------------------------------------------------------
    def __cube_probability_min(self, ext_min=0, ext_max=2, ext_step=0.1):
        """
        Constructs a cube of probabilities for the minimum extinction. Each pixel then shows the probability of having
        a certain amount of minimum extinction. The extinction is represented in the third axis.

        Parameters
        ----------
        ext_min : int, float, optional
            Minimum extinction to sample.
        ext_max : int, float, optional
            Maximum extinction to sample.
        ext_step : int, float, optional
            Step size in extinction.

        Returns
        -------
        ExtinctionCube

        """

        cube = self.__cube_probability_max(ext_min=ext_min, ext_max=ext_max, ext_step=ext_step)
        return ExtinctionCube(cube=1 - cube.cube, header=cube.header)

    # -----------------------------------------------------------------------------
    def map2cube(self, mode="probability density", ext_min=0, ext_max=2, ext_step=0.1):
        """
        Frontend user method to constuct FITS cubes of various kinds.

        Parameters
        ----------
        mode
        ext_min : int, float, optional
            Minimum extinction to sample.
        ext_max : int, float, optional
            Maximum extinction to sample.
        ext_step : int, float, optional
            Step size in extinction.

        Returns
        -------
        ExtinctionCube

        """

        if mode == "probability density":
            return self.__cube_probability_density(ext_min=ext_min, ext_max=ext_max, ext_step=ext_step)
        elif mode == "probability max":
            return self.__cube_probability_max(ext_min=ext_min, ext_max=ext_max, ext_step=ext_step)
        elif mode == "probability min":
            return self.__cube_probability_min(ext_min=ext_min, ext_max=ext_max, ext_step=ext_step)
        else:
            raise ValueError("Mode {0} not implemented".format(mode))


# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
class ExtinctionCube:

    # -----------------------------------------------------------------------------
    def __init__(self, cube, header=None):
        self.cube = cube
        self.header = header if header is not None else fits.Header(self.cube)

        # Get third axis range
        if self.header is not None:
            self.crange3 = (np.arange(self.cube.shape[0]) * self.header["CDELT3"] +
                            self.header["CRVAL3"] - (self.header["CRPIX3"] - 1) * self.header["CDELT3"])
        else:
            self.crange3 = np.arange(self.cube.shape[0])

    # -----------------------------------------------------------------------------
    @property
    def shape(self):
        return self.cube.shape

    # -----------------------------------------------------------------------------
    def save_fits(self, path, overwrite=True):
        hdu = fits.PrimaryHDU(self.cube, header=self.header)
        hdu.writeto(path, overwrite=overwrite)


# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
class DiscreteExtinctionMap(ExtinctionMap):

    # -----------------------------------------------------------------------------
    def __init__(self, map_ext, map_var, map_header, prime_header=None, map_num=None, map_rho=None):
        """
        Extinction map class.

        Parameters
        ----------
        map_ext : np.ndarray
            2D Extintion map.
        map_var : np.ndarray
            2D Extinction variance map.
        map_header : astropy.fits.Header
            Header of grid from which extinction map was built.
        map_num : np.ndarray, optional
            2D source count map.
        map_rho : np.ndarray, optional
            2D source density map.

        """

        # Set instance attributes
        super(DiscreteExtinctionMap, self).__init__(map_ext=map_ext, map_header=map_header, prime_header=prime_header)
        self.map_var = map_var
        self.map_num = np.full_like(self.map_ext, fill_value=np.nan, dtype=np.uint32) if map_num is None else map_num
        self.map_rho = np.full_like(self.map_ext, fill_value=np.nan, dtype=np.float32) if map_num is None else map_rho

        # Sanity check for dimensions
        if (self.map_ext.ndim != 2) | (self.map_var.ndim != 2) | (self.map_num.ndim != 2) | (self.map_rho.ndim != 2):
            raise TypeError("Input must be 2D arrays")

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
        import matplotlib.pyplot as plt

        # If the density should be plotted, set to 4
        nfig = 3

        fig = plt.figure(figsize=[figsize, nfig * 0.9 * figsize * (self.map_shape[0] / self.map_shape[1])])
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
                vmin, vmax = self._get_vlim(data=self.map_ext, percentiles=[0.1, 90], r=100)
                im = ax.imshow(self.map_ext, origin="lower", interpolation="nearest", vmin=vmin, vmax=vmax, cmap=cmap)
                fig.colorbar(im, cax=cax, label="Extinction (mag)")

            # Plot error map
            elif idx == 2:
                vmin, vmax = self._get_vlim(data=np.sqrt(self.map_var), percentiles=[1, 90], r=100)
                im = ax.imshow(np.sqrt(self.map_var), origin="lower", interpolation="nearest", vmin=vmin, vmax=vmax,
                               cmap=cmap)
                if self.prime_header["METRIC"] == "median":
                    fig.colorbar(im, cax=cax, label="MAD (mag)")
                else:
                    fig.colorbar(im, cax=cax, label="Error (mag)")

            # Plot source count map
            elif idx == 4:
                vmin, vmax = self._get_vlim(data=self.map_num, percentiles=[1, 99], r=1)
                im = ax.imshow(self.map_num, origin="lower", interpolation="nearest", vmin=vmin, vmax=vmax, cmap=cmap)
                fig.colorbar(im, cax=cax, label="N")

            elif idx == 6:
                vmin, vmax = self._get_vlim(data=self.map_rho, percentiles=[1, 99], r=1)
                im = ax.imshow(self.map_rho, origin="lower", interpolation="nearest", vmin=vmin, vmax=vmax, cmap=cmap)
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
                                fits.ImageHDU(data=self.map_ext, header=self.map_header),
                                fits.ImageHDU(data=self.map_var, header=self.map_header),
                                fits.ImageHDU(data=self.map_num, header=self.map_header),
                                fits.ImageHDU(data=self.map_rho, header=self.map_header)])

        # Write
        hdulist.writeto(path, clobber=clobber)
