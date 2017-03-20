# -----------------------------------------------------------------------------
# Import stuff
import numpy as np

from astropy import wcs
from itertools import combinations
from pnicer.utils.kde import mp_kde
from pnicer.utils.wcs import data2grid
from pnicer.utils.auxiliary import flatten_lol
from pnicer.utils.gmm import mp_gmm, gmm_scale, gmm_expected_value, gmm_population_variance
from pnicer.utils.plots import caxes, caxes_delete_ticklabels, finalize_plot
from pnicer.utils.algebra import round_partial, centroid_sphere, distance_sky

# noinspection PyPackageRequirements
from sklearn.neighbors import NearestNeighbors
# noinspection PyPackageRequirements
from sklearn.mixture import GaussianMixture


# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# noinspection PyProtectedMember
class Features:

    # -----------------------------------------------------------------------------
    def __init__(self, features, feature_err, feature_extvec, feature_names=None, feature_coordinates=None):
        """
        Basic Data class which provides the foundation for extinction measurements.

        Parameters
        ----------
        features : iterable
            List of feature arrays. All arrays must have the same length!
        feature_err : iterable
            List off feature error arrays.
        feature_extvec : iterable
            List holding the extinction components for each feature (extinction vector).
        feature_coordinates : astropy.coordinates.SkyCoord, optional
            Astropy SkyCoord instance.
        feature_names : list
            List of feature names.

        """

        # Set features
        self.features = features
        self.features_err = feature_err
        self.features_names = feature_names
        self.extvec = ExtinctionVector(extvec=feature_extvec)

        # Set coordinate attributes
        self.coordinates = feature_coordinates

        # Generate simple names for the magnitudes if not set
        if self.features_names is None:
            self.features_names = ["Mag" + str(idx + 1) for idx in range(self.n_features)]

        # Add data dictionary
        self.dict = {}
        for i in range(self.n_features):
            self.dict[self.features_names[i]] = self.features[i]
            self.dict[self.features_names[i] + "_err"] = self.features_err[i]

        # -----------------------------------------------------------------------------
        # Do some input checks

        # Dimensions of extinction vector must be equal to dimensions of data
        if self.extvec.n_dimensions != self.n_features:
            raise ValueError("Dimensions of extinction vector must be equal to number of features")

        # There must be at least one feature
        if self.n_features < 1:
            raise ValueError("There must be at least two features")

        # All input lists must have equal length
        if len(set([len(l) for l in [self.features, self.features_err, self.features_names]])) != 1:
            raise ValueError("Input lists must have equal length")

        # Input data must also have the same size
        if len(set([x.size for x in self.features])) != 1:
            raise ValueError("Input arrays must have equal size")

        # Coordinates must be supplied for all data if set
        if feature_coordinates is not None:
            if len(self.coordinates) != len(self.features[0]):
                raise ValueError("Input coordinates do not match to data!")

    # -----------------------------------------------------------------------------
    def __len__(self):
        return self.features_err.__len__()

    # -----------------------------------------------------------------------------
    # def __str__(self):
    #     return Table([np.around(x, 3) for x in self.features], names=self.features_names).__str__()

    # -----------------------------------------------------------------------------
    def __iter__(self):
        for x in self.features:
            yield x

    # ----------------------------------------------------------------------------- #
    #                            Some useful properties                             #
    # ----------------------------------------------------------------------------- #

    # -----------------------------------------------------------------------------
    @property
    def n_features(self):
        """
        Number of features.

        Returns
        -------
        int

        """

        return len(self.features)

    # -----------------------------------------------------------------------------
    @property
    def n_data(self):
        """
        Number of provided sources.

        Returns
        -------
        int

        """

        return self.features[0].size

    # -----------------------------------------------------------------------------
    @property
    def _n_data_strict_mask(self):
        """
        Number of sources which have data in all features.

        Returns
        -------
        int

        """

        return np.sum(self._strict_mask)

    # ----------------------------------------------------------------------------- #
    #                                     Masks                                     #
    # ----------------------------------------------------------------------------- #

    # -----------------------------------------------------------------------------
    @property
    def _features_masks(self):
        """
        Provides a list with masks for each given feature. True (1) entries are good, False (0) are bad.


        Returns
        -------
        iterable
            List of masks.

        """

        return [np.isfinite(m) & np.isfinite(e) for m, e in zip(self.features, self.features_err)]

    # -----------------------------------------------------------------------------
    def _loose_mask(self, max_bad_features):
        """
        Returns a mask where entries are masked when given 'max_bad_features' bad entries. When e.g. set to 0, no bad
        features are allowed at all (same as strict mask). When set to 2, then all entries are masked which have 2 or
        more bad measurements.

        Parameters
        ----------
        max_bad_features : int
            Maximum number of bad measurements for each source.

        Returns
        -------
        np.ndarray
            Loose mask.

        """
        return self.n_features - np.sum(np.vstack(self._features_masks), axis=0) <= max_bad_features

    # -----------------------------------------------------------------------------
    @property
    def _strict_mask(self):
        """
        Combines all feature masks into a single mask. Any entry that has a NaN in any band will be masked.

        Returns
        -------
        np.ndarray
            Combined mask.

        """

        return np.prod(np.vstack(self._features_masks), axis=0, dtype=bool)

        # Would be the same:
        # return self._loose_mask(max_bad_features=0)

    # -----------------------------------------------------------------------------
    def _custom_strict_mask(self, idx=None, names=None):
        """
        Creates a custom mask for a given set of combined features. Any entry that has a single NaN in any of the
        specified bands will be masked.

        Parameters
        ----------
        idx : iterable, optional
            List of indices to create the mask for
        names : iterable, optional
            List of names for the features for which a combined mask should be made


        Returns
        -------
        np.ndarray
            Combined mask.

        Raises
        ------
        ValueError
            If no argument is given.

        """

        if (idx is None) & (names is None):
            raise ValueError("Either indices or names must be specified")

        # Grab indices for given names
        if names:
            idx = [self.features_names.index(key) for key in names]

        # Return combined mask
        return np.prod(np.vstack([self._features_masks[i] for i in idx]), axis=0, dtype=bool)

    # -----------------------------------------------------------------------------
    @staticmethod
    def _mask2index(mask):
        """
        Converts a mask to an index array.

        Parameters
        ----------
        mask : np.array
            Inut mask to convert to index

        Returns
        -------
        np.array
            Index array of mask.

        """

        return np.where(mask)[0]

    # -----------------------------------------------------------------------------
    @property
    def _strict_mask_index(self):
        """
        Converts the strict mask to and index array.

        Returns
        -------
        np.array
            Index array of strict mask of instance.
        """

        return self._mask2index(self._strict_mask)

    # ----------------------------------------------------------------------------- #
    #                                 Helper methods                                #
    # ----------------------------------------------------------------------------- #

    # -----------------------------------------------------------------------------
    def _index2name(self, idx):
        """
        Returns the name (or names) of features based on their index entry.

        Parameters
        ----------
        idx : int, list
            Index (or indices) for which to get the names.

        Returns
        -------
            Name or Names

        """

        if isinstance(idx, int):
            return self.features_names[idx]
        elif (isinstance(idx, list)) | isinstance(idx, tuple):
            return [self.features_names[i] for i in idx]
        else:
            raise ValueError("Index must be integer or list")

    # -----------------------------------------------------------------------------
    def _name2index(self, name):
        """
        Returns the index (or indices) of features based on their name.

        Parameters
        ----------
        name : str, list
            Name (or names) of features

        Returns
        -------
            Index or indices.

        """

        if isinstance(name, str):
            return self.features_names.index(name)
        elif (isinstance(name, list)) | (isinstance(name, tuple)):
            return [self.features_names.index(n) for n in name]
        else:
            raise ValueError("Name must be given as a string or a list")

    # -----------------------------------------------------------------------------
    @staticmethod
    def _build_feature_grid(data, precision):
        """
        Static method to build a grid of unique positons from given input data rounded to arbitrary precision.

        Parameters
        ----------
        data : np.ndarray
            Data from which to build grid (at least 2D).
        precision : float, optional
            Desired precision, i.e. pixel scale.

        Returns
        -------
        np.ndarray
            Grid built from input data.

        """

        # Round data to requested precision.
        grid_data = round_partial(data=data, precision=precision).T

        # Get unique positions for coordinates
        dummy = np.ascontiguousarray(grid_data).view(np.dtype((np.void, grid_data.dtype.itemsize * grid_data.shape[1])))
        _, idx = np.unique(dummy, return_index=True)

        return grid_data[np.sort(idx)].T

    # -----------------------------------------------------------------------------
    @classmethod
    def _check_class(cls, ccls):
        """
        Checks another instance for compatability in the PNICER methods.

        Parameters
        ----------
        ccls
            Instance to check.

        Raises
        ------
        ValueError
            If the classes do not match.

        """

        if cls != ccls.__class__:
            raise ValueError("Instance and control class do not match")

    # ----------------------------------------------------------------------------- #
    #                                Instance methods                               #
    # ----------------------------------------------------------------------------- #

    # -----------------------------------------------------------------------------
    def _rotate(self):
        """
        Method to rotate data space with the given extinction vector. Only finite data are transmitted intp the new
        data space.

        Returns
        -------
            New instance with rotated data.

        """

        # Apply strict masks (no NaN can be present!)
        data = np.vstack(self.features).T[self._strict_mask].T
        err = np.vstack(self.features_err).T[self._strict_mask].T

        # Rotate data
        rotdata = self.extvec._rotmatrix.dot(data)

        # Rotate extinction vector
        extvec = self.extvec._extvec_rot

        # In case no coordinates are supplied they need to be masked
        if self.coordinates is not None:
            coordinates = self.coordinates[self._strict_mask]
        else:
            coordinates = None

        # Return
        # noinspection PyTypeChecker
        return self.__class__([rotdata[idx, :] for idx in range(self.n_features)],
                              [err[idx, :]for idx in range(self.n_features)],
                              extvec, coordinates, [x + "_rot" for x in self.features_names])

    # -----------------------------------------------------------------------------
    def _all_combinations(self, idxstart):
        """
        Method to get all combinations of input features

        Parameters
        ----------
        idxstart : int
            Minimun number of features required. Used to exclude single magnitudes for univariate PNICER.

        Returns
        -------
        iterable
            List of instances with all combinations from input features.

        """

        all_c = [item for sublist in [combinations(range(self.n_features), p)
                                      for p in range(idxstart, self.n_features + 1)]
                 for item in sublist]

        # Import
        from pnicer.user import ApparentColors, ApparentMagnitudes

        combination_instances = []
        for c in all_c:
            cdata, cerror = [self.features[idx] for idx in c], [self.features_err[idx] for idx in c]
            cnames = [self.features_names[idx] for idx in c]
            extvec = [self.extvec.extvec[idx] for idx in c]

            if isinstance(self, ApparentMagnitudes):
                combination_instances.append(self.__class__(magnitudes=cdata, errors=cerror, extvec=extvec,
                                                            coordinates=self.coordinates, names=cnames))
            elif isinstance(self, ApparentColors):
                combination_instances.append(self.__class__(colors=cdata, errors=cerror, extvec=extvec,
                                                            coordinates=self.coordinates, names=cnames))

        # Return list of combinations.
        return combination_instances

    # ----------------------------------------------------------------------------- #
    #                       Coordinate methods and attributes                       #
    # ----------------------------------------------------------------------------- #

    # -----------------------------------------------------------------------------
    # noinspection PyUnresolvedReferences
    def _build_wcs_grid(self, proj_code="TAN", pixsize=10 / 60, return_skycoord=False, **kwargs):
        """
        Generates a WCS grid.

        Parameters
        ----------
        proj_code : str, optional
            Projection code. Default is 'TAN'.
        pixsize : int, float, optional
            Pixel size of grid.
        return_skycoord : bool, optional
            Whether to return the grid coordinates as a SkyCoord object. Default is False
        kwargs
            Any additional header arguments for the projection (e.g. PV2_1, ect.)

        Returns
        -------
        tuple
            Tuple holding an astropy fits header and the world coordinate grid

        """

        return data2grid(lon=self.coordinates.spherical.lon.degree, lat=self.coordinates.spherical.lat.degree,
                         frame=self._frame_name, proj_code=proj_code, pixsize=pixsize,
                         return_skycoord=return_skycoord, **kwargs)

    # -----------------------------------------------------------------------------
    # noinspection PyUnresolvedReferences
    @property
    def _lon_deg(self):
        """ Longitudes in degrees """
        return self.coordinates.spherical.lon.degree

    # -----------------------------------------------------------------------------
    # noinspection PyUnresolvedReferences
    @property
    def _lat_deg(self):
        """ Latitudes in degrees """
        return self.coordinates.spherical.lat.degree

    # -----------------------------------------------------------------------------
    # noinspection PyUnresolvedReferences
    @property
    def _lon_rad(self):
        """ Longitudes in radian """
        return self.coordinates.spherical.lon.radian

    # -----------------------------------------------------------------------------
    # noinspection PyUnresolvedReferences
    @property
    def _lat_rad(self):
        """ Latitudes in radian """
        return self.coordinates.spherical.lat.radian

    # -----------------------------------------------------------------------------
    @property
    def _frame_name(self):
        """ Latitudes in radian """
        return self.coordinates.frame.name

    # ----------------------------------------------------------------------------- #
    #                                Plotting methods                               #
    # ----------------------------------------------------------------------------- #

    # -----------------------------------------------------------------------------
    # noinspection PyTypeChecker
    @property
    def _plotrange_features(self):
        """
        Convenience property to calculate a plot range for all provided features.

        Returns
        -------
        list
            List of plot ranges.

        """

        return [(np.floor(np.percentile(x[m], 0.01)), np.ceil(np.percentile(x[m], 99.99)))
                for x, m in zip(self.features, self._features_masks)]

    # -----------------------------------------------------------------------------
    @property
    def _plotrange_world(self):
        """
        Convenience property to calculate the plot range in world coordinates.

        Returns
        -------
        iterable
            List with (left, right) and (bottom, top) tuple entries

        """

        # Build a wcs gruid with the defaults
        header, _ = self._build_wcs_grid(proj_code="TAN", pixsize=1 / 60)

        # Get footprint coordinates
        flon, flat = wcs.WCS(header=header).calc_footprint().T

        # Calculate centroid
        clon, clat = centroid_sphere(lon=flon, lat=flat, units="degree")

        # Maximize distances to the field edges in longitude
        left = flon[:2][np.argmax(distance_sky(lon1=flon[:2], lat1=0, lon2=clon, lat2=0, unit="degree"))]
        right = flon[2:][np.argmax(distance_sky(lon1=flon[2:], lat1=0, lon2=clon, lat2=0, unit="degree"))]

        # Maximize distances in latitude
        top, bottom = np.max(flat), np.min(flat)

        return [(left, right), (bottom, top)]

    # -----------------------------------------------------------------------------
    @staticmethod
    def _get_plot_axsize(size):
        """
        Simple helper method to return axis sizes for plots.

        Parameters
        ----------
        size : int, float, list
            Size of axis

        Returns
        -------
        list
            Converted axis size.

        """

        # Return default
        if size is None:
            return [4, 4]

        # Or make list
        elif (isinstance(size, int)) | (isinstance(size, float)):
            return [size, size]

        # Or just return list
        elif isinstance(size, list):
            return size

        else:
            raise ValueError("Axis size must be given either with a single number or a list with two entries")

    # -----------------------------------------------------------------------------
    def _gridspec_world(self, pixsize, ax_size, proj_code):
        """
        Creates all necessary instances for plotting with a grid in world coordinates.

        Parameters
        ----------
        pixsize : int, float
            Pixel size of image data.
        ax_size : int, float
            Width of a single axis

        Returns
        -------
        tuple
            Tuple with the figure, the axes, the world coordinate grid, and the fits header.

        """

        # Import matplotlib
        from matplotlib import pyplot as plt
        from matplotlib.gridspec import GridSpec

        # Get a WCS grid
        header, grid_world = self._build_wcs_grid(proj_code=proj_code, pixsize=pixsize)

        # Get aspect ratio of grid
        ar = header["NAXIS2"] / header["NAXIS1"]

        # Choose layout depending on aspect ratio of image
        if ar < 1:
            ncols, nrows, o = 1, self.n_features, "v"
        else:
            ncols, nrows, o = self.n_features, 1, "h"

        # Create plot grid and figure
        fig = plt.figure(figsize=[ax_size * ncols, ax_size * nrows * ar])
        grid_plot = GridSpec(ncols=ncols, nrows=nrows, bottom=0.05, top=0.95, left=0.05, right=0.95,
                             hspace=0.25, wspace=0.25)

        # Add axes
        axes = [plt.subplot(grid_plot[idx], projection=wcs.WCS(header=header)) for idx in range(self.n_features)]

        # Generate labels
        llon, llat = "GLON" if "gal" in self._frame_name else "RA", "GLAT" \
            if "gal" in self._frame_name else "DEC"

        # Add feature labels
        [axes[idx].annotate(self.features_names[idx], xy=[0.5, 1.01], xycoords="axes fraction",
                            ha="center", va="bottom") for idx in range(self.n_features)]

        # Add axis labels
        if o == "v":

            # For a vertical arrangement we set the x label for only the bottom-most plot
            axes[-1].set_xlabel(llon)

            # and y labels for everything
            [ax.set_ylabel(llat) for ax in axes]

        elif o == "h":

            # For a horizontal arrangement we set the y label for the left most-plot
            axes[0].set_ylabel(llat)

            # and the x label for everything
            [ax.set_xlabel(llon) for ax in axes]

        return fig, axes, grid_world, header

    # -----------------------------------------------------------------------------
    def plot_combinations_scatter(self, path=None, ax_size=None, skip=1, **kwargs):
        """
        2D Scatter plot of combinations.

        Parameters
        ----------
        path : str, optional
            File path if it should be saved. e.g. "/path/to/image.png". Default is None.
        ax_size : int, float, optional
            Size of individual axis. Default is [3, 3]
        skip : int, optional
            Skip every n-th source for faster plotting. Default is 1.
        kwargs
            Any additional scatter plot arguments.

        """

        # Get figure and axes
        fig, axes = caxes(ndim=self.n_features, ax_size=self._get_plot_axsize(size=ax_size), labels=self.features_names)

        # Get 2D combination indices
        for idx, ax in zip(combinations(range(self.n_features), 2), axes):

            ax.scatter(self.features[idx[1]][::skip], self.features[idx[0]][::skip], lw=0, s=5, alpha=0.1, **kwargs)

            # We need a square grid!
            l, h = np.min([x[0] for x in self._plotrange_features]), np.max([x[1] for x in self._plotrange_features])

            # Ranges
            ax.set_xlim(l, h)
            ax.set_ylim(l, h)

        # Modify tick labels
        caxes_delete_ticklabels(axes=axes, xfirst=False, xlast=True, yfirst=False, ylast=True)

        # Save or show figure
        finalize_plot(path=path)

    # -----------------------------------------------------------------------------
    def plot_combinations_kde(self, path=None, ax_size=None, grid_bw=0.1, kernel="epanechnikov", cmap="gist_heat_r"):
        """
        KDE for all 2D combinations of features

        Parameters
        ----------
        path : str, optional
            File path if it should be saved. e.g. "/path/to/image.png". Default is None.
        ax_size : int, float, optional
            Size of individual axis. Default is [3, 3]
        grid_bw : int, float, optional
            Grid size. Default is 0.1.
        kernel : str, optional
            Name of kernel for KDE. e.g. 'epanechnikov' or 'gaussian'. Default is 'epanechnikov'.
        cmap : str, optional
            Colormap to be used in plot. Default is 'gist_heat_r'.

        """

        # Get default axis size
        ax_size = self._get_plot_axsize(size=ax_size)

        # Get figure and axes
        fig, axes = caxes(ndim=self.n_features, ax_size=self._get_plot_axsize(size=ax_size),
                          labels=self.features_names)

        # Get 2D combination indices
        for idx, ax in zip(combinations(range(self.n_features), 2), axes):

            # Get clean data from the current combination
            mask = np.prod(np.vstack([self._features_masks[idx[0]], self._features_masks[idx[1]]]), axis=0, dtype=bool)

            # We need a square grid!
            l, h = np.min([x[0] for x in self._plotrange_features]), np.max([x[1] for x in self._plotrange_features])

            x, y = np.meshgrid(np.arange(start=l, stop=h, step=grid_bw), np.arange(start=l, stop=h, step=grid_bw))

            # Get kernel density
            data = np.vstack([self.features[idx[0]][mask], self.features[idx[1]][mask]]).T
            xgrid = np.vstack([x.ravel(), y.ravel()]).T
            dens = mp_kde(grid=xgrid, data=data, bandwidth=grid_bw * 2, kernel=kernel, absolute=True,
                          sampling=2).reshape(x.shape)

            # Plot result
            ax.imshow(dens.T, origin="lower", interpolation="nearest", extent=[l, h, l, h], cmap=cmap)

        # Modify tick labels
        caxes_delete_ticklabels(axes=axes, xfirst=False, xlast=True, yfirst=False, ylast=True)

        # Save or show figure
        finalize_plot(path=path)

    # -----------------------------------------------------------------------------
    def plot_sources_scatter(self, path=None, ax_size=10, skip=1, **kwargs):
        """
        Plot source coordinates in a scatter plot.

        Parameters
        ----------
        path : str, optional
            Path to file, if the plot should be save.
        ax_size : int, float, optional
            Single axis width in plot. Default is 10.
        skip : int, optional
            Skip every n-th source for faster plotting. Default is 1.
        kwargs
            Any additional scatter keyword argument from matplotlib.

        """

        # Get figure and axes
        fig, axes, _, header = self._gridspec_world(pixsize=10 / 60, ax_size=ax_size, proj_code="TAN")

        # Get plot limits
        lim = wcs.WCS(header=header).wcs_world2pix(self._plotrange_world[0], self._plotrange_world[1], 0)

        # Loop over features and plot
        for idx in range(self.n_features):

            # Grab axes
            ax = axes[idx]

            ax.scatter(self._lon_deg[self._features_masks[idx]][::skip],
                       self._lat_deg[self._features_masks[idx]][::skip],
                       transform=ax.get_transform(self._frame_name), **kwargs)

            # Set axes limits
            ax.set_xlim(lim[0])
            ax.set_ylim(lim[1])

        # Finalize plot
        finalize_plot(path=path)

    # -----------------------------------------------------------------------------
    def plot_sources_kde(self, path=None, bandwidth=10 / 60, ax_size=10, kernel="epanechnikov", skip=1, **kwargs):
        """
        Plot source densities for features

        Parameters
        ----------
        path : str, optional
            File path if it should be saved. e.g. '/path/to/image.png'. Default is None.
        bandwidth : int, float, optional
            Kernel bandwidth in degrees. Default is 10 arcmin. 2x sampling is forced.
        ax_size : int, float, optional
            Single axis width in plot. Default is 10.
        kernel : str, optional
            Name of kernel for KDE. e.g. 'epanechnikov' or 'gaussian'. Default is 'epanechnikov'.
        skip : int, optional
            Skip every n-th source for faster plotting. Default is 1.
        kwargs
            Additional keyword arguments for imshow.

        """

        # Get figure, axes, and wcs grid
        fig, axes, grid_world, header = self._gridspec_world(pixsize=bandwidth / 2, ax_size=ax_size, proj_code="TAN")

        # To avoid editor warning
        scale = 1

        # Loop over features and plot
        for idx in range(self.n_features):

            # Get density
            xgrid = np.vstack([grid_world[0].ravel(), grid_world[1].ravel()]).T
            data = np.vstack([self._lon_deg[self._features_masks[idx]][::skip],
                              self._lat_deg[self._features_masks[idx]][::skip]]).T
            dens = mp_kde(grid=xgrid, data=data, bandwidth=bandwidth, kernel=kernel,
                          norm=None).reshape(grid_world[0].shape)

            # Norm and save scale (we want everything scaled to the same reference! In this case the first feature)
            if idx == 0:
                scale = np.max(dens)
            dens /= scale

            # Mask lowest level with NaNs
            dens[dens <= 1e-4] = np.nan

            # Show density
            axes[idx].imshow(dens, origin="lower", interpolation="nearest", cmap="viridis", vmin=0, vmax=1, **kwargs)

        # Save or show figure
        finalize_plot(path=path)

    # ----------------------------------------------------------------------------- #
    #                              Main PNICER routines                             #
    # ----------------------------------------------------------------------------- #

    # -----------------------------------------------------------------------------
    @staticmethod
    def _set_defaults_gmm(**kwargs):
        """
        Sets the defaults for Gaussian Mixture Model fits.

        Parameters
        ----------
        kwargs : dict
            kwargs passed from pnicer frontend.

        Returns
        -------
        dict
            Updated dictionary with GMM defaults.

        """

        # Remove n_components if given
        if "n_components" in kwargs.keys():
            kwargs.pop("n_components", None)

        # Set defaults
        if "covariance_type" not in kwargs:
            kwargs["covariance_type"] = "full"
        if "tol" not in kwargs:
            kwargs["tol"] = 1E-3
        if "max_iter" not in kwargs:
            kwargs["max_iter"] = 100
        if "warm_start" not in kwargs:
            kwargs["warm_start"] = False

        # Return argument dictionary
        return kwargs

    # -----------------------------------------------------------------------------
    def _pnicer_univariate(self, control, max_components, **kwargs):
        """
        Univariate PNICER implmentation.

        Parameters
        ----------
        control
            Control Field Feature instance
        max_components : int
            Maximum number of GMM compponents to use.
        kwargs
            Additional kwargs for GaussianMixture

        Returns
        -------
            Fitted model, variance, model index, and zero-point for all sources.

        """

        # Generate outout arrays
        idx_all = np.full(self.n_data, fill_value=-1, dtype=np.int64)
        var_all = np.full(self.n_data, fill_value=1E6, dtype=np.float32)
        zp_all = np.full(self.n_data, fill_value=0, dtype=np.float32)

        # Reuqire a minimum of 20 sources in control fields
        if control._n_data_strict_mask < 20:
            return [None], var_all, idx_all, zp_all

        # Define and fit Gaussian Mixture Model
        gmm = mp_gmm(data=[control.features[0][control._strict_mask].reshape(-1, 1)], max_components=max_components,
                     parallel=False, **self._set_defaults_gmm(**kwargs))[0]

        # Scaling factor to extinction
        scale = self.extvec._extinction_norm

        # Shift factor to center on expected value
        shift = -gmm_expected_value(gmm=gmm)

        # Scale model to extinction
        gmm = gmm_scale(gmm=gmm, shift=shift, scale=scale, reverse=True)

        # Determine population variance in units of extinction
        var = gmm_population_variance(gmm=gmm)

        # Fill output arrays
        idx_all[self._strict_mask] = 0
        var_all[self._strict_mask] = var
        zp_all[self._strict_mask] = (self.features[0][self._strict_mask] + shift) / scale

        # Return
        return [gmm], var_all, idx_all, zp_all

    # -----------------------------------------------------------------------------
    def _pnicer_multivariate(self, control, max_components, **kwargs):
        """
        Mulitvariate PNICER implementation.

        Parameters
        ----------
        control
            Control Field Feature instance
        max_components : int
            Maximum number of GMM compponents to use.
        kwargs
            Additional kwargs for GaussianMixture

        Returns
        -------
            Fitted models, variance, model index, and zero-point for all sources.

        """

        # Create and index, variance and zp arrays for all sources
        idx_all = np.full(self.n_data, fill_value=-1, dtype=np.int64)
        var_all = np.full(self.n_data, fill_value=1E6, dtype=np.float32)
        zp_all = np.full(self.n_data, fill_value=np.nan, dtype=np.float32)

        # Rotate the data spaces
        science_rot, control_rot = self._rotate(), control._rotate()

        # Return if after rotation no valid photometry is left
        if control_rot.n_data == 0:
            return [None], var_all, idx_all, zp_all

        # Get bandwidth from photometric errors
        bandwidth = np.round(np.mean(np.nanmean(self.features_err, axis=1)), 2)

        # Determine bin widths for grid according to bandwidth and sampling
        # TODO: Optimize sampling or make it user choice
        sampling = 2
        bin_grid = np.float(bandwidth / sampling)

        # Now we build a grid from the rotated data for all components but the first
        grid_data = Features._build_feature_grid(data=np.vstack(science_rot.features)[1:, :], precision=bin_grid)

        # Define NN algorithm
        try:
            nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(grid_data.T)

        # Return if bad grid
        except ValueError:
            return [None], var_all, idx_all, zp_all

        # Assign each control field source the nearest neighbor grid point
        dis, idx = nbrs.kneighbors(np.vstack(control_rot.features)[1:, :].T)
        dis, idx = dis[:, 0], idx[:, 0]

        # Filter too large distances (in case the NN is further than the grid bin width)
        # TODO: The smaller the sampling (here fixed at 2), the less data in a vector!
        idx[dis > bandwidth / 2] = grid_data.shape[-1] + 1

        # Build data vectors for GMM-fits
        vectors_data = [control_rot.features[0][idx == i].reshape(-1, 1) for i in range(grid_data.shape[-1])]

        # Each control field vector needs to contain at least 20 sources
        vectors_data = [None if len(v) < 20 else v for v in vectors_data]

        # If there are no valid vectors, return
        if len(vectors_data) == 0:
            return [None], var_all, idx_all, zp_all

        # Fit GMM for each vector
        vectors_gmm = mp_gmm(data=vectors_data, max_components=max_components, **self._set_defaults_gmm(**kwargs))

        # Determine scaling and shifting factor for models
        scale = self.extvec._extinction_norm
        vectors_shift = [-gmm_expected_value(gmm=gmm, method="weighted")
                         if isinstance(gmm, GaussianMixture) else None for gmm in vectors_gmm]

        # Scale GMMs to extinction
        vectors_gmm = [gmm_scale(gmm=gmm, scale=scale, shift=shift, reverse=True)
                       if isinstance(gmm, GaussianMixture) else None for gmm, shift in zip(vectors_gmm, vectors_shift)]

        # Determine variance for each vector
        vectors_var = [gmm_population_variance(gmm=gmm, method="weighted")
                       if gmm is not None else np.nan for gmm in vectors_gmm]

        # Get good models
        grid_good_idx = [i for i, j in enumerate(vectors_gmm) if isinstance(j, GaussianMixture)]
        vectors_gmm = [vectors_gmm[i] for i in grid_good_idx]
        vectors_shift = [vectors_shift[i] for i in grid_good_idx]

        # Return if no model converged
        if len(vectors_gmm) == 0:
            return [None], var_all, idx_all, zp_all

        # Find the nearest neighbor for each science target in the cleaned grid
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(grid_data[:, grid_good_idx].T)
        science_idx = nbrs.kneighbors(np.vstack(science_rot.features)[1:, :].T)[1][:, 0]

        # Determine variance for each source
        var = np.array([vectors_var[idx] for idx in science_idx])

        # Fill and index, variance and zp arrays for all sources
        idx_all[self._strict_mask] = science_idx
        var_all[self._strict_mask] = var
        zp_all[self._strict_mask] = (science_rot.features[0] + np.array(vectors_shift)[science_idx]) / scale

        # Return
        return vectors_gmm, var_all, idx_all, zp_all

    # -----------------------------------------------------------------------------
    def _pnicer_combinations(self, combinations_science, combinations_control, max_components, **kwargs):
        """
        PNICER base implementation for combinations. Calls the PNICER implementation for all combinations. The output
        extinction is then the one with the smallest variance from all combinations.

        Parameters
        ----------
        control
            Instance of control field data.
        combinations_science : iterable
            List of combinations for the science data.
        combinations_control : iterable
            List of combinations for the control field data.

        Returns
        -------
        ContinuousExtinction

        """

        # Loop over all combinations and run PNICER
        gmm_combinations, var_combinations, uidx_combinations, zp_combinations, models_norm = [], [], [], [], []
        for sc, cc in zip(combinations_science, combinations_control):

            # Choose uni/multivariate PNICER
            if sc.n_features == 1:
                g, v, i, zp = sc._pnicer_univariate(control=cc, max_components=max_components, **kwargs)
            else:
                g, v, i, zp = sc._pnicer_multivariate(control=cc, max_components=max_components, **kwargs)

            # Generate unique index for stacked GMM array
            uidx_combinations.append(i + len(flatten_lol(gmm_combinations)))

            # Set bad indices to negative again
            uidx_combinations[-1][i < 0] = -1

            # Save results
            gmm_combinations.append(g)
            var_combinations.append(v)
            zp_combinations.append(zp)

        # Save combinations
        self.__setattr__("__gmm_combinations", gmm_combinations)
        self.__setattr__("__zp_combinations", zp_combinations)

        # Stack unique GMMs and norms
        gmm_unique = np.hstack(gmm_combinations)

        # Determine all bad slices
        all_bad = np.sum(~np.isfinite(np.array(var_combinations)), axis=0) == len(uidx_combinations)

        # Choose minimum variance GMM across all combinations
        minidx = np.argmin(np.array(var_combinations), axis=0)

        # Put an existing model into bad slices
        minidx[all_bad] = np.median(minidx)

        # Select model index
        sources_index = np.array(uidx_combinations)[minidx, np.arange(self.n_data)]

        # Create clean index of models
        clean_index = list(set(sources_index))
        clean_index.sort()
        clean_index = list(filter(lambda a: a != -1, clean_index))

        # Create new index
        new_index = [i for i in range(len(clean_index))]

        # Get difference between indices
        diff_index = [c - n for c, n in zip(clean_index, new_index)]

        # Fetch and sort all clean unique models
        gmm_unique = list(gmm_unique[clean_index])

        # Rebase sources_index
        for c, d in zip(clean_index, diff_index):
            sources_index[sources_index == c] = sources_index[sources_index == c] - d

        # Mask all bad slices
        sources_index[all_bad] = -1

        # Set all negative indices to bad value
        sources_index[sources_index < 0] = sources_index.size + 1

        # Get zero point for each source
        sources_zp = np.array(zp_combinations)[minidx, np.arange(self.n_data)]
        # TODO: Check what is happening to all-bad slices

        # Chech if all bad slices are also bad in the index
        # TODO: Do I need this check?
        # a = np.where(np.nansum(np.array(var_combinations), axis=0) >
        #              1E6 * (len(combinations_science) - 1) + 1)[0].shape  # All-bad variances
        # b = np.where(sources_index > self.n_data)[0].shape  # All bad indices
        # if not a == b:
        #     raise ValueError("Bad data is being propagated")

        # Return
        from pnicer.extinction import ContinuousExtinction
        return ContinuousExtinction(features=self, models=gmm_unique, index=sources_index, zp=sources_zp)

    # -----------------------------------------------------------------------------
    def features_intrinsic(self, extinction):
        """
        Calculates the de-reddened features

        Parameters
        ----------
        extinction : np.array
            Extinction for each source in the same unit as the extinction vector for this instance.

        Returns
        -------
        iterable
            List of intrinsic features

        """

        return [f - extinction * v for f, v in zip(self.features, self.extvec.extvec)]


# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
class ExtinctionVector:

    # -----------------------------------------------------------------------------
    def __init__(self, extvec):
        """
        Class for extinction vectors.

        Parameters
        ----------
        extvec : iterable
            List of extinction values for each input feature.

        """

        # Set attributes
        self.extvec = extvec

    # -----------------------------------------------------------------------------
    def __len__(self):
        return len(self.extvec)

    # -----------------------------------------------------------------------------
    def __str__(self):
        return self.extvec.__str__()

    # -----------------------------------------------------------------------------
    def __iter__(self):
        for x in self.extvec:
            yield x

    # -----------------------------------------------------------------------------
    @property
    def n_dimensions(self):
        """
        Number of dimensions.

        Returns
        -------
        int

        """

        return len(self.extvec)

    # -----------------------------------------------------------------------------
    @staticmethod
    def _unit_vectors(n_dimensions):
        """
        Calculate unit vectors for a given number of dimensions.

        Parameters
        ----------
        n_dimensions : int
            Number of dimensions

        Returns
        -------
        iterable
            Unit vectors for each dimension in a list.

        """

        return [np.array([1.0 if i == l else 0.0 for i in range(n_dimensions)]) for l in range(n_dimensions)]

    # -----------------------------------------------------------------------------
    @staticmethod
    def _get_rotmatrix(vector):
        """
        Method to determine the rotation matrix so that the rotated first vector component is the only non-zero
        component.

        Parameters
        ----------
        vector
            Input extinction vector

        Returns
        -------
        np.ndarray
            Rotation matrix

        """

        # Number of dimensions
        n_dimensions = len(vector)
        if n_dimensions < 2:
            ValueError("Vector must have at least two dimensions")

        # Get unit vectors
        uv = ExtinctionVector._unit_vectors(n_dimensions=n_dimensions)

        # To not raise editor warning
        vector_rot = [0]

        # Now we loop over all but the first component
        rotmatrices = []
        for n in range(n_dimensions - 1):

            # Calculate rotation angle of current component
            if n == 0:
                rot_angle = np.arctan(vector[n + 1] / vector[0])
            else:
                rot_angle = np.arctan(vector_rot[n + 1] / vector_rot[0])

            # Following the german Wikipedia... :)
            v = np.outer(uv[0], uv[0]) + np.outer(uv[n + 1], uv[n + 1])
            w = np.outer(uv[0], uv[n + 1]) - np.outer(uv[n + 1], uv[0])
            rotmatrices.append((np.cos(rot_angle) - 1) * v + np.sin(rot_angle) * w + np.identity(n_dimensions))

            # Rotate reddening vector
            if n == 0:
                vector_rot = rotmatrices[-1].dot(vector)
            else:
                vector_rot = rotmatrices[-1].dot(vector_rot)

        # Now we have rotation matrices for each component and we must combine them
        rotmatrix = rotmatrices[-1]
        for n in reversed(range(0, len(rotmatrices) - 1)):
            rotmatrix = rotmatrix.dot(rotmatrices[n])

        return rotmatrix

    # -----------------------------------------------------------------------------
    @property
    def _rotmatrix(self):
        """
        Property to hold the rotation matrix for all extinction components of the current instance.

        Returns
        -------
        np.ndarray
            Rotation matrix for the extinction vector of the instance.

        """

        return ExtinctionVector._get_rotmatrix(self.extvec)

    # -----------------------------------------------------------------------------
    @property
    def _rotmatrix_inv(self):
        """
        Inverted rotation matrix.

        Returns
        -------
        np.ndarray

        """

        return self._rotmatrix.T

    # -----------------------------------------------------------------------------
    @property
    def _extvec_rot(self):
        """
        Calculates the rotated extinction vector.

        Returns
        -------
        np.ndarray

        """

        return self._rotmatrix.dot(self.extvec)

    # -----------------------------------------------------------------------------
    @property
    def _extinction_norm(self):
        """
        Normalization component for projected extinction vector.

        Returns
        -------
        float

        """

        # For one-dimensional data there is no rotation matrix
        if self.n_dimensions == 1:
            return self.extvec[0]
        else:
            return self._extvec_rot[0]
