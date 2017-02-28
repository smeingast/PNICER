# -----------------------------------------------------------------------------
# Import stuff
import sys
import time
import warnings
import numpy as np

from astropy import wcs
from astropy.io import fits
from itertools import repeat
from astropy.table import Table
from multiprocessing.pool import Pool
# noinspection PyPackageRequirements
from sklearn.mixture import GaussianMixture
# noinspection PyPackageRequirements
from sklearn.neighbors import NearestNeighbors

from pnicer.utils.plots import finalize_plot
from pnicer.extinction_map import ContinuousExtinctionMap, DiscreteExtinctionMap
from pnicer.utils.algebra import centroid_sphere, distance_sky, std2fwhm, round_partial
from pnicer.utils.gmm import gmm_scale, gmm_expected_value, gmm_sample_xy, gmm_max, gmm_confidence_interval, \
    gmm_population_variance, gmm_sample_xy_components, mp_gmm_combine, gmm_combine


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# noinspection PyProtectedMember
class Extinction:

    def __init__(self, features):
        self.features = features

    # -----------------------------------------------------------------------------
    def __len__(self):
        return self.features.n_data

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
    def query_position(self, skycoord, bandwidth, mode="average", metric="gaussian", use_fwhm=False, nicest=False,
                       alpha=1/3):
        # TODO: Add docstring

        # FWHM can only be used with a gaussian metric
        if use_fwhm & (metric != "gaussian"):
            raise ValueError("FWHM only valid for gaussian kernel")

        # Adjust bandwidth in case FWHM is to be used
        if use_fwhm:
            bandwidth /= std2fwhm

        # Calculate distances to specified coordinates
        dis = skycoord.separation(self.features.coordinates).degree

        # Get spatial weights
        w_spatial = self._get_weights(distances=dis, metric=metric, bandwidth=bandwidth)

        # Mask sources outside of truncation scale
        trunc_radius = bandwidth if (metric == "average") | (metric == "median") else 3 * bandwidth
        w_spatial[dis > trunc_radius / 2] = np.nan

        # Return based on object instance
        if isinstance(self, DiscreteExtinction):

            if mode.lower() == "model":
                return self._get_extinction_model(nbrs_idx=np.arange(self.features.n_data), w_spatial=w_spatial,
                                                  nicest=nicest, alpha=alpha)
            elif mode.lower() == "average":
                return self._get_extinction_average(nbrs_idx=np.arange(self.features.n_data), w_spatial=w_spatial,
                                                    metric=metric, nicest=nicest, alpha=alpha)[:-1]
            else:
                raise ValueError("Mode {0} not implemented".format(mode))

        if isinstance(self, ContinuousExtinction):

            if mode.lower() == "model":
                return self._get_extinction_model(nbrs_idx=np.arange(self.features.n_data), w_spatial=w_spatial,
                                                  nicest=nicest, alpha=alpha)
            elif mode.lower() == "average":

                return self._get_extinction_average(nbrs_idx=np.arange(self.features.n_data), w_spatial=w_spatial,
                                                    nicest=nicest, alpha=alpha)[:-1]
            else:
                raise ValueError("Mode {0} not implemented".format(mode))
        else:
            raise ValueError("Extinction query not supported for {0}".format(self.__class__))

    # -----------------------------------------------------------------------------
    def build_map(self, bandwidth, mode="average", metric="gaussian", use_fwhm=False,
                  nicest=False, alpha=1/3, sampling=2, **kwargs):
        # TODO: Add docstring

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

        # Set truncation scale based on metric
        trunc_radius = bandwidth if (metric == "average") | (metric == "median") else 3 * bandwidth

        # Create WCS grid
        map_hdr, map_wcs = self.features._build_wcs_grid(pixsize=pixsize, return_skycoord=True, **kwargs)

        # Create primary header for map with metric information
        p_hdr = self._make_prime_header(bandwidth=bandwidth, metric=metric, sampling=sampling, nicest=nicest)

        # Get shape of map
        map_shape = map_wcs.data.shape

        # Determine number of nearest neighbors to query from 10 random samples
        ridx = np.random.choice(len(self.features._lon_deg), size=10, replace=False)

        d = distance_sky(self.features._lon_deg[ridx].reshape(-1, 1), self.features._lat_deg[ridx].reshape(-1, 1),
                         self.features._lon_deg, self.features._lat_deg, unit="degree")
        nnbrs = np.ceil(np.median(np.sum(d < trunc_radius / 2, axis=1))).astype(np.int)

        # Maximum of 500 nearest neighbors
        nnbrs = 500 if nnbrs > 500 else nnbrs
        # TODO: Issue warning when the farthest neighbor is outside the truncation radius.

        # Convert coordinates to 3D cartesian
        sci_car = self.features.coordinates.cartesian.xyz.value.T
        map_car = map_wcs.cartesian.xyz.reshape(3, -1).value.T

        # Find nearest neighbors of grid points to all sources
        nbrs = NearestNeighbors(n_neighbors=nnbrs, algorithm="auto", metric="euclidean", n_jobs=-1).fit(sci_car)
        nbrs_idx = nbrs.kneighbors(map_car, return_distance=False)

        # Calculate all distances in grid (in float32 to save some memory)
        map_dis = distance_sky(map_wcs.spherical.lon.degree.ravel().astype(np.float32),
                               map_wcs.spherical.lat.degree.ravel().astype(np.float32),
                               self.features._lon_deg[nbrs_idx].T.astype(np.float32),
                               self.features._lat_deg[nbrs_idx].T.astype(np.float32), unit="degree")

        # Get spatial weights
        w_spatial = self._get_weights(distances=map_dis, metric=metric, bandwidth=bandwidth)

        # Mask sources outside of truncation scale
        w_spatial[map_dis > trunc_radius / 2] = np.nan

        # Build map for discretized extinction
        if isinstance(self, DiscreteExtinction):

            # In case an average extinction map should be built
            if mode == "average":

                # Calculate values for all map pixels
                ext, var, num, rho = self._get_extinction_average(nbrs_idx=nbrs_idx.T, w_spatial=w_spatial,
                                                                  metric=metric, nicest=nicest, alpha=alpha)

                # Reshape to map and return
                return DiscreteExtinctionMap(map_ext=ext.reshape(map_shape), map_var=var.reshape(map_shape),
                                             map_num=num.reshape(map_shape), map_rho=rho.reshape(map_shape),
                                             map_header=map_hdr, prime_header=p_hdr)

            if mode == "model":

                # Build new GMMs for each pixel
                gmms = [self._get_extinction_model(nbrs_idx=n, w_spatial=w, nicest=nicest, alpha=alpha)
                        for n, w in zip(nbrs_idx, w_spatial.T)]

                # Return continuous extinction map
                return ContinuousExtinctionMap(map_models=np.array(gmms).reshape(map_shape),
                                               map_header=map_hdr, prime_header=p_hdr)

            # Raise error if mode is not recognized
            else:
                raise ValueError("Mode '{0}' not implemented. Use either 'model' or 'average'.".format(mode))

        # ...or continous extinction
        elif isinstance(self, ContinuousExtinction):

            # In case an average extinction map should be built
            if mode == "average":

                ext, var, num, rho = self._get_extinction_average(nbrs_idx=nbrs_idx.T, w_spatial=w_spatial,
                                                                  nicest=nicest, alpha=alpha)

                # Reshape to map and return
                return DiscreteExtinctionMap(map_ext=ext.reshape(map_shape), map_var=var.reshape(map_shape),
                                             map_num=num.reshape(map_shape), map_rho=rho.reshape(map_shape),
                                             map_header=map_hdr, prime_header=p_hdr)

            # In case extinction map with full models should be built
            elif mode == "model":

                # Get combined models for
                gmms = self._get_extinction_model(nbrs_idx=nbrs_idx.T, w_spatial=w_spatial, nicest=nicest, alpha=alpha)

                # Return continuous extinction map
                return ContinuousExtinctionMap(map_models=np.array(gmms).reshape(map_shape),
                                               map_header=map_hdr, prime_header=p_hdr)

            else:
                raise ValueError("Mode '{0}' not implemented. Use either 'model' or 'average'.".format(mode))

        # Or raise an Error
        else:
            raise NotImplementedError

    # -----------------------------------------------------------------------------
    @staticmethod
    def _get_weights(distances, metric, bandwidth):
        """
        Defines the weight function for extinction mapping.

        Parameters
        ----------
        metric : str
            The metric to be used. One of 'uniform', 'triangular', 'gaussian', 'epanechnikov'
        bandwidth : int, float
            Bandwidth of metric (kernel).

        """

        if metric == "uniform" or metric == "average" or metric == "median":
            def wfunc(wdis):
                """
                Returns
                -------
                float, np.ndarray

                """
                return np.ones_like(wdis)

        elif metric == "gaussian":
            def wfunc(wdis):
                return np.exp(-0.5 * (wdis / bandwidth) ** 2)

        elif metric == "epanechnikov":
            # noinspection PyUnresolvedReferences
            def wfunc(wdis):
                val = 1 - (wdis / bandwidth) ** 2
                val[val < 0] = 0
                return val

        elif metric == "triangular":
            # noinspection PyUnresolvedReferences
            def wfunc(wdis):
                val = 1 - np.abs(wdis / bandwidth)
                val[val < 0] = 0
                return val

        else:
            raise TypeError("metric {0:s} not implemented".format(metric))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            weights = wfunc(distances)

            return np.divide(weights, np.trapz(y=wfunc(np.arange(-100, 100, 0.01)), x=np.arange(-100, 100, 0.01)))
            # return weights / np.nanmax(weights, axis=0)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
class ContinuousExtinction(Extinction):

    def __init__(self, features, models, index, zp):

        # Input checks
        if (len(index) != features.n_data) | (len(zp) != features.n_data):
            raise ValueError("Incompatible input for InstrinsicProbability")

        # Set instance attributes
        self.models = models
        self.index = index
        self.zp = zp
        super(ContinuousExtinction, self).__init__(features=features)

    # -----------------------------------------------------------------------------
    @property
    def _sources_mask(self):
        """
        Mask for bad sources where no intrinsic probability distribution could be derived. The mask contains True
        for good sources and False for bad.

        Returns
        -------
        ndarray
            Numpy array with boolean entries.

        """

        return self.index < self.features.n_data

    # -----------------------------------------------------------------------------
    @property
    def _n_models(self):
        """
        Number of unique models in instance.

        Returns
        -------
        int

        """

        return len(self.models)

    # -----------------------------------------------------------------------------
    @property
    def _n_sources_models(self):
        """
        Number of sources in the science field assigned to each unique model.

        Returns
        -------
        ndarray
            Numpy array with number of sources for each unique model.

        """

        return np.histogram(self.index, bins=np.arange(-0.5, stop=self._n_models + 0.5, step=1))[0]

    # ----------------------------------------------------------------------------- #
    #                             Model attribute tools                             #
    # ----------------------------------------------------------------------------- #

    # -----------------------------------------------------------------------------
    @property
    def _model_params(self):
        """
        GMM model parameters (shared for all unique instances)

        Returns
        -------
        dict
            GMM model parameter dictionary.

        """

        return self.models[0].get_params()

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

        return [getattr(gmm, attr) for gmm in self.models]

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

    # -----------------------------------------------------------------------------
    @property
    def _models_precision_cholesky(self):
        """
        Fetches precisions for all gaussian mixture models.

        Returns
        -------
        iterable
            List of precisions of models.
        """

        return self.__get_models_attributes(attr="precisions_cholesky_")

    # ----------------------------------------------------------------------------- #
    #                            Model evaluation tools                             #
    # ----------------------------------------------------------------------------- #

    # -----------------------------------------------------------------------------
    def _models_sample_xy(self, **kwargs):
        """
        Creates discrete values (x, y) for all probability density distributions.

        Parameters
        ----------
        kwargs
            Additional parameters (kappa, sigma, nmin, nmax)

        Returns
        -------
        iterable
            List of tuples (xarr, yarr)

        """

        return [gmm_sample_xy(gmm=gmm, **kwargs) for gmm in self.models]

    # -----------------------------------------------------------------------------
    @property
    def _models_expected_value(self, method="weighted"):
        """
        Calculates the expected value for all model probability density distributions.

        Parameters
        ----------
        method : str, optional
            Method to use to calculate the expected value. Either 'weighted' (default) or 'integral'.

        Returns
        -------
        iterable
            List of expected values.

        """

        return [gmm_expected_value(gmm=gmm, method=method) for gmm in self.models]

    # -----------------------------------------------------------------------------
    @property
    def _models_max(self):
        """
        Returns the coordinates of the maximum of the probability density distribution.

        Returns
        -------
        iterable
            List of coordinates for maxima for each model.

        """

        return [gmm_max(gmm=gmm, sampling=100) for gmm in self.models]

    # -----------------------------------------------------------------------------
    @property
    def _models_population_variance(self, method="weighted"):
        """
        Determine the population variance of the probability density distributions for all unique models.

        Parameters
        ----------
        method : str, optional
            Method to use to calculate the variance. Either 'weighted' (default) or 'integral'.

        Returns
        -------
        iterable
            List of population variances for all GMMs.

        """

        return [gmm_population_variance(gmm=gmm, method=method) for gmm in self.models]

    # -----------------------------------------------------------------------------
    def _models_confidence_interval(self, levels):
        """
        Calculates the confidence intervals for all models at specified confidence levels.

        Parameters
        ----------
        levels : float, list
            Confidence levels to evaluate. Either a single value for all models, or a list of levels for each model.

        Returns
        -------
        iterable
            List of confidence intervals for given confidence levels.

        """

        # Convert to lists if parameters are given as single values
        if isinstance(levels, float):
            levels = [levels for _ in range(self._n_models)]
        if isinstance(levels, list):
            if len(levels) != self._n_models:
                raise ValueError("Levels must be either specified by a single value or a list of values for each model")

        return [gmm_confidence_interval(gmm=gmm, level=l) for gmm, l in zip(self.models, levels)]

    # ----------------------------------------------------------------------------- #
    #                               Extinction tools                                #
    # ----------------------------------------------------------------------------- #

    # -----------------------------------------------------------------------------
    @property
    def _models_extinction(self):
        """
        Creates the extinction probability density distribution for each source.

        Returns
        -------
        iterable
            List of Gaussian Mixture model instances for each source.

        """

        model_params = self._model_params
        sources_mask = self._sources_mask

        return [gmm_scale(gmm=self.models[midx], shift=self.zp[sidx], params=model_params) if mask else None
                for sidx, midx, mask in zip(range(self.features.n_data), self.index, sources_mask)]

    # -----------------------------------------------------------------------------
    def _model_extinction_source(self, idx):
        """
        Creates a Gaussiam Mixture Model probability density distribution describing the extinction for a specific
        source (as given by the index).

        Parameters
        ----------
        idx : int
            Index of source.

        Returns
        -------
        GaussianMixture
            GMM instance describing the extinction for the given source.

        """

        midx = self.index[idx]
        return gmm_scale(gmm=self.models[midx], shift=self.zp[idx], scale=None, reverse=False)

    # -----------------------------------------------------------------------------
    def __discretize(self, metric="expected value"):
        """
        Discretize extinction from probability density distributions.

        Parameters
        ----------
        metric : str, optional
            Metric to be used. Either 'expected value' or 'max'.

        Returns
        -------
        DiscreteExtinction
            DiscreteExtinction object with the discretized values and errors.

        """
        # TODO: This does not correctly reflect the average extinction (error estimate is wrong)

        if "expect" in metric.lower():
            attr = "_models_expected_value"
        elif "max" in metric.lower():
            attr = "_models_intrinsic_max"
        else:
            raise ValueError("Metric '{0}' not supported".format(metric))

        idx = self.index[self._sources_mask]

        # Determine extinction based on chosen metric
        ext = np.full_like(self.zp, fill_value=np.nan, dtype=np.float32)
        ext[self._sources_mask] = np.array(getattr(self, attr))[idx] + self.zp[self._sources_mask]

        # Get population variance of models
        var = np.full_like(self.zp, fill_value=np.nan, dtype=np.float32)
        var[self._sources_mask] = np.array(self._models_population_variance)[idx]

        # Return
        return DiscreteExtinction(features=self.features, extinction=ext, variance=var)

    # ----------------------------------------------------------------------------- #
    #                                Extinction map                                 #
    # ----------------------------------------------------------------------------- #

    # -----------------------------------------------------------------------------
    def _get_extinction_average(self, nbrs_idx, w_spatial, nicest=False, alpha=1/3):
        """
        Calcualtes the average extinction given a set of models as defined by the nbrs_index and the spatial weights.

        Parameters
        ----------
        nbrs_idx : np.array
            Array containing the model indices for all neighbors.
        w_spatial : np.array
            Array with the weights for all sources.
        nicest : bool, optional
            Whether the NICEST correction factor should be used. Default is False.
        alpha : int, float, optional
            Slope in luminosity function for NICEST. Default is 1/3 (for NIR bands).

        Returns
        -------
        np.array, np.array, np.array, np.array
            Tuple holing the average extinction, the variance, the number of sources used, and the density of sources.

        """

        # Ignore Runtime warnings (due to NaNs) for the following steps
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            # Get model index for all neighbors
            model_idx = self.index[nbrs_idx]

            # Mask bad neighbors
            model_idx[~np.isfinite(w_spatial)] = self.features.n_data + 1
            good_idx = model_idx < self.features.n_data   # type: np.ndarray
            model_idx[~good_idx] = 0

            # Get GMM attributes for neighbors
            nbrs_popvar = np.array(self._models_population_variance)[model_idx]
            nbrs_zp = self.zp[nbrs_idx]

            # Adjust weights with variance
            var_weights = np.array(self._models_population_variance)[model_idx]
            w_total = w_spatial / var_weights   # type: np.ndarray

            # Modify weights with NICEST factor
            if nicest:
                k_lambda = np.max(self.features.extvec.extvec) if self.features.extvec.extvec is not None else 1
                w_total *= 10**(alpha * k_lambda * nbrs_zp)

            # Calculate weighted mean
            m = np.nansum(w_total * nbrs_zp, axis=0) / np.nansum(w_total, axis=0)
            v = np.nansum(w_total ** 2. * nbrs_popvar, axis=0) / np.nansum(w_total, axis=0)**2

            # Get source density
            rho = np.nansum(w_spatial, axis=0)

        # Return
        return m, v, np.nansum(np.isfinite(w_total), axis=0), rho

    # -----------------------------------------------------------------------------
    def _get_extinction_model(self, nbrs_idx, w_spatial, nicest=False, alpha=1/3):
        # TODO: Add docstring

        idx = self.index[nbrs_idx]
        idx[~np.isfinite(w_spatial)] = self.features.n_data + 1
        good_idx = idx < self.features.n_data   # type: np.ndarray
        idx[~good_idx] = 0

        # Get model index for all neighbors
        nbrs_models = np.array(self.models)[idx]

        # Get GMM attributes for neighbors
        nbrs_means = np.array(self._models_means)[idx]
        nbrs_variances = np.array(self._models_variances)[idx]
        nbrs_weights = np.array(self._models_weights)[idx]
        nbrs_zp = self.zp[nbrs_idx]

        # Adjust weights with variance
        var_weights = np.array(self._models_population_variance)[idx]
        w_total = w_spatial / var_weights

        # Modify weights with NICEST factor
        if nicest:
            k_lambda = np.max(self.features.extvec.extvec) if self.features.extvec.extvec is not None else 1
            w_total *= 10**(alpha * k_lambda * nbrs_zp)

        # Build combined Models
        params = self.models[0].get_params()

        # In case multiple coordinates are queried
        if isinstance(nbrs_models.T[0], list):
            return mp_gmm_combine(gmms=nbrs_models.T, weights=w_total.T, params=params, good_idx=good_idx.T,
                                  gmms_means=nbrs_means.T, gmms_variances=nbrs_variances.T,
                                  gmms_weights=nbrs_weights.T, gmms_zps=nbrs_zp.T)

        # Otherwise just combine all models in the list
        else:
            return gmm_combine(gmms=nbrs_models, weights=w_total, params=params, good_idx=good_idx,
                               gmms_means=nbrs_means, gmms_variances=nbrs_variances,
                               gmms_weights=nbrs_weights, gmms_zps=nbrs_zp)

    # ----------------------------------------------------------------------------- #
    #                               Plotting methods                                #
    # ----------------------------------------------------------------------------- #

    # -----------------------------------------------------------------------------
    def _plot_models(self, path=None, ax_size=None, confidence_level=0.9, draw_components=True):
        """
        Creates a plot of all unique Gaussian Mixture Models.

        Parameters
        ----------
        path : str, optional
            Figure file path.
        ax_size : list, optional
            Size of axis for a single model (e.g. [5, 4]). Defaults to [4, 4].
        confidence_level : float, optional
            Confidence interval level to be plotted. Default is 0.9. If no interval should be plotted, set to None.

        """

        # Import
        import matplotlib.pyplot as plt
        from matplotlib.pyplot import GridSpec
        from matplotlib.ticker import AutoMinorLocator

        # Set axis size
        if ax_size is None:
            ax_size = [4, 4]

        # Determine layout
        nrows = int(np.floor(np.sqrt(self._n_models)))
        ncols = int(np.ceil(self._n_models / nrows))

        # Generate plot grid
        plt.figure(figsize=[ax_size[0] * ncols, ax_size[1] * nrows])
        grid = GridSpec(ncols=ncols, nrows=nrows, bottom=0.05, top=0.95, left=0.05, right=0.95,
                        hspace=0.15, wspace=0.15)

        plot_range = []
        for idx in range(self._n_models):

            # Grab GMM and scale
            gmm = self.models[idx]

            # Adjust so that distribution is centered on 0 fpr plots
            # gmm.means_ -= np.mean(gmm.means_)

            # Add axis
            ax = plt.subplot(grid[idx])

            # Get plot range and values
            x, y = gmm_sample_xy(gmm=gmm, kappa=4, sampling=10, nmin=100, nmax=5000)

            # Draw entire GMM
            ax.plot(x, y, color="black", lw=2)

            # Draw individual components if set
            if draw_components:
                xc, yc = gmm_sample_xy_components(gmm=gmm, kappa=4, sampling=10, nmin=100, nmax=5000)
                for y in yc:
                    ax.plot(xc, y, color="black", linestyle="dashed", lw=1)

            # Add confidence interval if requested
            if confidence_level is not None:

                # Check input
                if not isinstance(confidence_level, float):
                    raise ValueError("Specified confidence level must be >0 and <1!")

                # Draw upper and lower limit
                for l in gmm_confidence_interval(gmm=gmm, level=confidence_level):
                    ax.axvline(l, ymin=0, ymax=2, color="black", linestyle="dotted", lw=2, alpha=0.7)

            # Add expected value and max
            ax.axvline(gmm_max(gmm=gmm), color="crimson", alpha=0.8, linestyle="dashed", lw=2)
            ax.axvline(gmm_expected_value(gmm=gmm), color="#2A52BE", alpha=0.8, linestyle="dashed", lw=2)

            # Annotate
            if idx % ncols == 0:
                ax.set_ylabel("Probability Density")
            if idx >= self._n_models - ncols:
                ax.set_xlabel("Extinction + ZP")

            # Add number of sources for each model
            ax.annotate("N = " + str(self._n_sources_models[idx]), xy=[0.5, 1.02], xycoords="axes fraction",
                        va="bottom", ha="center")

            # Add number of components for each model
            ax.annotate("NC = " + str(gmm.get_params()["n_components"]), xy=[0.03, 0.97], xycoords="axes fraction",
                        va="top", ha="left")

            # Ticks
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())

            # Save plot range
            plot_range.append(np.max(np.abs(x)))

        # Set common xrange
        xr = round_partial(np.mean(plot_range) - 0.1, precision=0.1)
        for idx in range(self._n_models):
            ax = plt.subplot(grid[idx])

            # Set symmetric range
            ax.set_xlim(-xr, xr)

        # Save or show figure
        finalize_plot(path=path, dpi=150)

    # -----------------------------------------------------------------------------
    def _plot_model_extinction_source(self, idx, path=None, ax_size=8, confidence_level=None, **kwargs):
        """
        Plot the full extinction model for a specific source.

        Parameters
        ----------
        idx : int
            Index of source to be plotted.
        path : str, optional
            Figure file path
        ax_size : int
            Size of figure axis
        confidence_level : float, optional
            If set, also plot confidence interval at given level.
        kwargs
            Any additional keyword arguments for the plot function which draws the distribution (e.g. color, lw, etc.).

        """

        # Import
        import matplotlib.pyplot as plt
        from matplotlib.ticker import AutoMinorLocator

        # Fetch extinction model for source
        egmm = self._model_extinction_source(idx=idx)

        # Get samples
        x, y = gmm_sample_xy(gmm=egmm, kappa=3, sampling=50)

        # Make figure
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[ax_size, 0.7 * ax_size])

        # Draw
        ax.plot(x, y, **kwargs)

        # Add confidence interval if requested
        if confidence_level is not None:

            # Check input
            if not isinstance(confidence_level, float):
                raise ValueError("Specified confidence level must be >0 and <1!")

            # Draw upper and lower limit
            for l in gmm_confidence_interval(gmm=egmm, level=confidence_level):
                ax.axvline(l, ymin=0, ymax=2, color="black", linestyle="dashed", lw=2, alpha=0.7)

        # Annotate
        ax.set_xlabel("Extinction")
        ax.set_ylabel("Probability Density")

        # Ticks
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        # Save or show figure
        finalize_plot(path=path)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# noinspection PyProtectedMember
class DiscreteExtinction(Extinction):

    # -----------------------------------------------------------------------------
    def __init__(self, features, extinction, variance=None):
        """
        Class for Intrisic features and extinction.

        Parameters
        ----------
        features
            Features instance from which the extinction was created.
        extinction : np.ndarray
            Extinction data.
        variance : np.ndarray, optional
            Variance in extinction.

        """

        # Set attributes
        self.extinction = extinction
        self.variance = np.zeros_like(extinction) if variance is None else variance
        super(DiscreteExtinction, self).__init__(features=features)

        # Sanity checks
        if len(self.extinction) != len(self.variance):
            raise ValueError("Extinction and variance arrays must have equal length")

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

    # ----------------------------------------------------------------------------- #
    #                               Extinction map                                  #
    # ----------------------------------------------------------------------------- #

    # -----------------------------------------------------------------------------
    def _get_extinction_average(self, nbrs_idx, w_spatial, metric, nicest, alpha):
        """
        Calcualtes the average extinction defined by the nbrs_index and the spatial weights.

        Parameters
        ----------
        nbrs_idx : np.array
            Array containing the extinction indices for all neighbors.
        w_spatial : np.array
            Array with the weights for all sources.
        nicest : bool, optional
            Whether the NICEST correction factor should be used. Default is False.
        alpha : int, float, optional
            Slope in luminosity function for NICEST. Default is 1/3 (for NIR bands).

        Returns
        -------
        np.array, np.array, np.array, np.array
            Tuple holing the average extinction, the variance, the number of sources used, and the density of sources.

        """

        # Ignore Runtime warnings (due to NaNs) for the following steps
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            # Get extinction and variance for all good neighbours
            bad_idx = ~np.isfinite(w_spatial)
            nbrs_ext, nbrs_var = self.extinction[nbrs_idx], self.variance[nbrs_idx]
            nbrs_ext[bad_idx], nbrs_var[bad_idx] = np.nan, np.nan

            # Conditional choice for different metrics
            if metric == "average":

                # 3 sig filter
                ext = np.nanmean(nbrs_ext, axis=0)
                sigfil = (np.abs(nbrs_ext - ext) > 3 * np.nanstd(nbrs_ext, axis=0))

                # Apply filter
                nbrs_ext[sigfil], nbrs_var[sigfil] = np.nan, np.nan

                # Make final maps
                ext = np.nanmean(nbrs_ext, axis=0)
                num = np.sum(np.isfinite(nbrs_ext), axis=0).astype(np.uint32)
                var = np.nansum(nbrs_var, axis=0) / num**2
                rho = np.full_like(ext, fill_value=np.nan)

            elif metric == "median":

                # Make final maps
                ext = np.nanmedian(nbrs_ext, axis=0)
                var = np.nanmedian(np.abs(nbrs_ext - ext), axis=0)   # MAD
                num = np.sum(np.isfinite(nbrs_ext), axis=0).astype(np.uint32)
                rho = np.full_like(ext, fill_value=np.nan)

            # If not median or average, fetch weight function
            else:

                # Total weight with variance
                w_total = w_spatial / nbrs_var  # type: np.ndarray

                # Do sigma clipping in extinction
                ext = np.nansum(w_total * nbrs_ext, axis=0) / np.nansum(w_total, axis=0)
                sigfil = (np.abs(nbrs_ext - ext) > 3 * np.nanstd(nbrs_ext, axis=0))

                # Apply sigma clipping
                nbrs_ext[sigfil], nbrs_var[sigfil], w_spatial[sigfil], w_total[sigfil] = np.nan, np.nan, np.nan, np.nan

                # TODO: Check and possibly fix this
                # Get approximate integral and normalize weights
                # w_spatial = np.divide(w_spatial, np.trapz(y=wfunc(np.arange(-100, 100, 0.01)),
                #                                           x=np.arange(-100, 100, 0.01)))

                # Modify weights for NICEST and calculate variance
                if nicest:

                    # Set parameters for density correction
                    k_lambda = np.max(self.features.extvec.extvec) if self.features.extvec.extvec is not None else 1
                    beta = np.log(10) * alpha * k_lambda

                    # Modify weights
                    w_spatial *= 10 ** (alpha * k_lambda * nbrs_ext)
                    w_total *= 10 ** (alpha * k_lambda * nbrs_ext)

                    # Correction factor (Equ. 34 in NICEST paper)
                    map_cor = beta * np.nansum(w_total * nbrs_var, axis=0) / np.nansum(w_total, axis=0)

                    # Calculate error for NICEST (private communication with M. Lombardi)
                    var = np.nansum((w_total**2*np.exp(2*beta*nbrs_ext) * (1+beta*nbrs_ext)**2) / nbrs_var, axis=0)
                    var /= np.nansum(w_total * np.exp(beta * nbrs_ext) / nbrs_var, axis=0) ** 2

                # Variance without NICEST
                else:

                    var = np.divide(np.nansum(w_total ** 2. * nbrs_var, axis=0), np.nansum(w_total, axis=0) ** 2)
                    map_cor = np.full_like(var, fill_value=0.)

                # Determine weighted extinction
                ext = np.nansum(w_total * nbrs_ext, axis=0) / np.nansum(w_total, axis=0) - map_cor

                # Determine the number of sources used for each pixel
                num = np.sum(np.isfinite(w_total), axis=0).astype(np.uint32)

                # Get source density
                rho = np.nansum(w_spatial, axis=0)

        # Return extinction map
        return ext, var, num, rho

    # -----------------------------------------------------------------------------
    def _get_extinction_model(self, nbrs_idx, w_spatial, nicest, alpha):
        # TODO: Add docstring

        # Get extinction and variance for all good neighbours
        good_idx = np.isfinite(w_spatial) & np.isfinite(self.extinction) & np.isfinite(self.variance)
        nbrs_ext, nbrs_var = self.extinction[good_idx], self.variance[good_idx]
        w_total = w_spatial[good_idx] / nbrs_var

        # Number of components for new GMM
        n_components = len(nbrs_ext)

        # Modify weights with NICEST factor
        if nicest:
            k_lambda = np.max(self.features.extvec.extvec) if self.features.extvec.extvec is not None else 1
            w_total *= 10 ** (alpha * k_lambda * nbrs_ext)

        # Create new GMM from extinction estimates
        gmm = GaussianMixture(n_components=n_components)

        # Modify instance attributes of GMM
        gmm.means_ = nbrs_ext.reshape(n_components, 1)
        gmm.covariances_ = nbrs_var.reshape(n_components, 1, 1)
        gmm.weights_ = w_total / np.sum(w_total)
        gmm.precisions_ = np.linalg.inv(gmm.covariances_)
        gmm.precisions_cholesky_ = np.linalg.cholesky(gmm.precisions_)

        # Return new GaussianMixture instance
        return gmm

    # -----------------------------------------------------------------------------
    def _build_map_(self, bandwidth, metric="median", sampling=2, nicest=False, alpha=1/3, use_fwhm=False, **kwargs):
        """ Obsolete code """

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

        # Set truncation scale based on metric
        trunc_radius = bandwidth if (metric == "average") | (metric == "median") else 3 * bandwidth

        # Create primary header for map with metric information
        p_hdr = self._make_prime_header(bandwidth=bandwidth, metric=metric, sampling=sampling, nicest=nicest)

        # Create WCS grid
        map_hdr, map_wcs = self.features._build_wcs_grid(pixsize=pixsize, return_skycoord=True, **kwargs)

        # Determine number of nearest neighbors to query from 10 random samples
        ridx = np.random.choice(len(self.features._lon_deg), size=10, replace=False)

        d = distance_sky(self.features._lon_deg[ridx].reshape(-1, 1), self.features._lat_deg[ridx].reshape(-1, 1),
                         self.features._lon_deg, self.features._lat_deg, unit="degree")
        nnbrs = np.ceil(np.median(np.sum(d < trunc_radius / 2, axis=1))).astype(np.int)

        # Maximum of 500 nearest neighbors
        nnbrs = 500 if nnbrs > 500 else nnbrs

        # Convert coordinates to 3D cartesian
        sci_car = np.array(self.features.coordinates.cartesian.xyz).T
        map_car = np.array(map_wcs.cartesian.xyz.reshape(3, -1).T)

        # Find nearest neighbors of grid points to all sources
        nbrs = NearestNeighbors(n_neighbors=nnbrs, algorithm="auto", metric="euclidean", n_jobs=-1).fit(sci_car)
        nbrs_idx = nbrs.kneighbors(map_car, return_distance=False)

        # Calculate all distances in grid (in float32 to save some memory)
        a = map_wcs.spherical.lon.degree.ravel().astype(np.float32)
        b = map_wcs.spherical.lat.degree.ravel().astype(np.float32)
        c = self.features._lon_deg[nbrs_idx].T.astype(np.float32)
        d = self.features._lat_deg[nbrs_idx].T.astype(np.float32)
        map_dis = distance_sky(a, b, c, d, unit="degree").T

        # Index to mask sources outside range
        bad_idx = map_dis > trunc_radius / 2

        # Get extinction and variance for all good neighbours
        nbrs_ext, nbrs_var = self.extinction[nbrs_idx], self.variance[nbrs_idx]
        nbrs_ext[bad_idx], nbrs_var[bad_idx] = np.nan, np.nan

        # Conditional choice for different metrics
        if metric == "average":

            # 3 sig filter
            map_ext = np.nanmean(nbrs_ext, axis=1)
            sigfil = (np.abs(nbrs_ext.T - map_ext) > 3 * np.nanstd(nbrs_ext, axis=1)).T

            # Apply filter
            nbrs_ext[sigfil], nbrs_var[sigfil] = np.nan, np.nan

            # Make final maps
            map_ext = np.nanmean(nbrs_ext, axis=1).reshape(map_wcs.data.shape)
            map_num = np.sum(np.isfinite(nbrs_ext), axis=1).reshape(map_wcs.data.shape).astype(np.uint32)
            map_var = np.sqrt(np.nansum(nbrs_var, axis=1)).reshape(map_wcs.data.shape) / map_num
            map_rho = np.full_like(map_ext, fill_value=np.nan)

        elif metric == "median":

            # Make final maps
            map_ext = np.nanmedian(nbrs_ext, axis=1)
            map_var = np.nanmedian(np.abs(nbrs_ext.T - map_ext).T, axis=1).reshape(map_wcs.data.shape)   # MAD
            map_ext = map_ext.reshape(map_wcs.data.shape)
            map_num = np.sum(np.isfinite(nbrs_ext), axis=1).reshape(map_wcs.data.shape).astype(np.uint32)
            map_rho = np.full_like(map_ext, fill_value=np.nan)

        # If not median or average, fetch weight function
        else:
            wfunc = _get_weight_func(metric=metric, bandwidth=bandwidth)

            # Ignore Runtime warnings (due to NaNs) for the following steps
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)

                # Get spatial weights:
                w_theta = wfunc(wdis=map_dis)
                w_theta = (w_theta.T / np.nanmax(w_theta, axis=1)).T

                # Total weight with variance
                w_total = w_theta / nbrs_var   # type: np.ndarray

                # Do sigma clipping in extinction
                map_ext = np.nansum(w_total * nbrs_ext, axis=1) / np.nansum(w_total, axis=1)
                sigfil = (np.abs(nbrs_ext.T - map_ext) > 3 * np.nanstd(nbrs_ext, axis=1)).T

                # Apply sigma clipping
                nbrs_ext[sigfil], nbrs_var[sigfil], w_theta[sigfil], w_total[sigfil] = np.nan, np.nan, np.nan, np.nan

                # Get approximate integral and normalize weights
                w_theta = np.divide(w_theta, np.trapz(y=wfunc(np.arange(-100, 100, 0.01)),
                                                      x=np.arange(-100, 100, 0.01)))

                # Modify weights for NICEST and calculate variance
                if nicest:

                    # Set parameters for density correction
                    k_lambda = np.max(self.features.extvec.extvec) if self.features.extvec.extvec is not None else 1
                    beta = np.log(10) * alpha * k_lambda

                    # Modify weights
                    w_theta *= 10 ** (alpha * k_lambda * nbrs_ext)
                    w_total *= 10 ** (alpha * k_lambda * nbrs_ext)

                    # Correction factor (Equ. 34 in NICEST paper)
                    map_cor = beta * np.nansum(w_total * nbrs_var, axis=1) / np.nansum(w_total, axis=1)

                    # Calculate error for NICEST (private communication with M. Lombardi)
                    map_var = np.nansum((w_total**2*np.exp(2*beta*nbrs_ext) * (1+beta*nbrs_ext)**2) / nbrs_var, axis=1)
                    map_var /= np.nansum(w_total * np.exp(beta * nbrs_ext) / nbrs_var, axis=1) ** 2
                    map_var = map_var.reshape(map_wcs.data.shape)

                # Variance without NICEST
                else:

                    map_var = np.divide(np.nansum(w_total ** 2. * nbrs_var, axis=1), np.nansum(w_total, axis=1) ** 2)
                    map_cor = np.full_like(map_var, fill_value=0.)
                    map_var = map_var.reshape(map_wcs.data.shape)

                # Determine weighted extinction
                map_ext = np.nansum(w_total * nbrs_ext, axis=1) / np.nansum(w_total, axis=1) - map_cor
                map_ext = map_ext.reshape(map_wcs.data.shape)

                # Determine the number of sources used for each pixel
                map_num = np.sum(np.isfinite(w_total), axis=1).reshape(map_wcs.data.shape).astype(np.uint32)

                # Get source density
                map_rho = np.nansum(w_theta, axis=1).reshape(map_wcs.data.shape)

        # Return extinction map
        return DiscreteExtinctionMap(map_ext=map_ext, map_var=map_var, map_num=map_num, map_rho=map_rho,
                                     map_header=map_hdr, prime_header=p_hdr)

    # -----------------------------------------------------------------------------
    def _build_map_old(self, bandwidth, metric="median", sampling=2, nicest=False, alpha=1/3, use_fwhm=False,
                       silent=False, **kwargs):
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
            Whether to activate the NICEST correction factor. Default is False.
        alpha : float, optional
            The slope in the number counts (NICEST equation 2). Default is 1/3.
        use_fwhm : bool, optional
            If set, the bandwidth parameter represents the gaussian FWHM instead of its standard deviation. Only
            available when using a gaussian weighting.
        silent : bool, optional
            Whether information on the progress should be printed. Default is False.

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
        k_lambda = np.max(self.features.extvec.extvec) if self.features.extvec.extvec is not None else 1

        # Create WCS grid
        grid_header, (grid_lon, grid_lat) = self.features._build_wcs_grid(pixsize=pixsize, **kwargs)

        # Create pixel grid
        grid_x, grid_y = wcs.WCS(grid_header).wcs_world2pix(grid_lon, grid_lat, 0)

        # Grid shape
        grid_shape = grid_x.shape

        # Get pixel coordinates of sources
        sources_x, sources_y = wcs.WCS(grid_header).wcs_world2pix(self.features._lon_deg, self.features._lat_deg, 0)

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
                                    lon2=self.features._lon_deg[self._clean_index],
                                    lat2=self.features._lat_deg[self._clean_index], unit="degree") < pmax

                # Filter data
                splon = self.features._lon_deg[self._clean_index][pfil]
                splat = self.features._lat_deg[self._clean_index][pfil]
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
        return DiscreteExtinctionMap(map_ext=full[0], map_var=full[1], map_num=full[2], map_rho=full[3],
                                     map_header=grid_header, prime_header=phdr)

    # -----------------------------------------------------------------------------
    def save_fits(self, path, overwrite=True):
        """
        Write the extinction data to a FITS table file.

        Parameters
        ----------
        path : str
            File path; e.g. "/path/to/table.fits"
        overwrite : bool, optional
            Whether to overwrite an existing file.

        """

        # Create FITS columns
        col1 = fits.Column(name="Lon", format="D", array=self.features._lon_deg)
        col2 = fits.Column(name="Lat", format="D", array=self.features._lat_deg)
        col3 = fits.Column(name="Extinction", format="E", array=self.extinction)
        col4 = fits.Column(name="Variance", format="E", array=self.variance)

        # Column definitions
        cols = fits.ColDefs([col1, col2, col3, col4])

        # Create binary table object
        tbhdu = fits.BinTableHDU.from_columns(cols)

        # Write to file
        tbhdu.writeto(path, overwrite=overwrite)


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
    """ Obsolete code """

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
