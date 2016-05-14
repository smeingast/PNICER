# ----------------------------------------------------------------------
# Import stuff
import wcsaxes
import warnings
import numpy as np

from astropy import wcs
from itertools import combinations
# noinspection PyPackageRequirements
from sklearn.neighbors import NearestNeighbors

# from pnicer.user import Magnitudes, Colors
from pnicer.utils import weighted_avg, caxes, mp_kde, data2header, caxes_delete_ticklabels, round_partial


# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #
# noinspection PyProtectedMember
class DataBase:

    def __init__(self, mag, err, extvec, coordinates=None, names=None):
        """
        Basic Data class which provides the foundation for extinction measurements.

        Parameters
        ----------
        mag : iterable
            List of magnitude arrays. All arrays must have the same length!
        err : iterable
            List off magnitude error arrays.
        extvec : iterable
            List holding the extinction components for each magnitude.
        coordinates : astropy.coordinates.SkyCoord, optional
            Astropy SkyCoord instance.
        names : list
            List of magnitude (feature) names.

        """

        # Set features
        self.features = mag
        self.features_err = err
        self.features_names = names
        self.extvec = ExtinctionVector(extvec=extvec)

        # Set coordinate attributes
        self.coordinates = coordinates

        # Define combination properties determined while running PNICER
        # TODO: Where exactly is this used?
        self._ext_combinations = None
        self._var_combinations = None
        self._combination_names = None
        self._n_combinations = 0

        # Generate simple names for the magnitudes if not set
        if self.features_names is None:
            self.features_names = ["Mag" + str(idx + 1) for idx in range(self.n_features)]

        # Add data dictionary
        self.dict = {}
        for i in range(self.n_features):
            self.dict[self.features_names[i]] = self.features[i]
            self.dict[self.features_names[i] + "_err"] = self.features_err[i]

        # ----------------------------------------------------------------------
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
        if self.coordinates is not None:
            if len(self.coordinates) != len(self.features[0]):
                raise ValueError("Input coordinates do not match to data!")

    # ---------------------------------------------------------------------- #
    #                         Some useful properties                         #
    # ---------------------------------------------------------------------- #

    # ----------------------------------------------------------------------
    @property
    def n_features(self):
        """
        Number of features.

        Returns
        -------
        int

        """

        return len(self.features)

    # ----------------------------------------------------------------------
    @property
    def n_data(self):
        """
        Number of provided sources.

        Returns
        -------
        int

        """

        return self.features[0].size

    # ----------------------------------------------------------------------
    @property
    def _features_masks(self):
        """
        Provides a list with masks for each given feature.


        Returns
        -------
        iterable
            List of masks.

        """

        return [np.isfinite(m) & np.isfinite(e) for m, e in zip(self.features, self.features_err)]

    # ---------------------------------------------------------------------- #
    #                                  Masks                                 #
    # ---------------------------------------------------------------------- #

    # ----------------------------------------------------------------------
    @property
    def _combined_mask(self):
        """
        Combines all feature masks into a single mask.

        Returns
        -------
        np.ndarray
            Combined mask.

        """

        return np.prod(np.vstack(self._features_masks), axis=0, dtype=bool)

    # ----------------------------------------------------------------------
    def _custom_mask(self, idx=None, names=None):
        """
        Creates a custom mask for a given set of combined features

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

    # ---------------------------------------------------------------------- #
    #                          Coordinate properties                         #
    # ---------------------------------------------------------------------- #

    # ----------------------------------------------------------------------
    @property
    def _frame(self):
        """
        Coordinate frame type

        Returns
        -------
        str

        """

        if self.coordinates is None:
            return None
        else:

            # Get frame
            frame = self.coordinates.frame.name

            # Check coordinate system
            if frame not in ["icrs", "galactic"]:
                raise ValueError("Frame '{0:s}' not suppoerted".format(self._frame))

            # Otherwise return
            return self.coordinates.frame.name

    # ----------------------------------------------------------------------
    # noinspection PyUnresolvedReferences
    @property
    def _lon(self):
        """
        Longitude coordinate array.

        Returns
        -------
        np.ndarray

        """

        if self._frame is None:
            return None
        elif self._frame == "galactic":
            return self.coordinates.l.degree
        elif self._frame == "icrs":
            return self.coordinates.ra.degree

    # ----------------------------------------------------------------------
    # noinspection PyUnresolvedReferences
    @property
    def _lat(self):
        """
        Latitude coordinate array.

        Returns
        -------
        np.ndarray

        """

        if self._frame is None:
            return None
        elif self._frame == "galactic":
            return self.coordinates.b.degree
        elif self._frame == "icrs":
            return self.coordinates.dec.degree

    # ---------------------------------------------------------------------- #
    #                             Helper methods                             #
    # ---------------------------------------------------------------------- #

    # ----------------------------------------------------------------------
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

    # ----------------------------------------------------------------------
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

    # ---------------------------------------------------------------------- #
    #                            Instance methods                            #
    # ---------------------------------------------------------------------- #

    # ----------------------------------------------------------------------
    def _rotate(self):
        """
        Method to rotate data space with the given extinction vector. Only finite data are transmitted intp the new
        data space.

        Returns
        -------
            New instance with rotated data.

        """

        data = np.vstack(self.features).T[self._combined_mask].T
        err = np.vstack(self.features_err).T[self._combined_mask].T

        # Rotate data
        rotdata = self.extvec._rotmatrix.dot(data)

        # Rotate extinction vector
        extvec = self.extvec._extinction_rot

        # In case no coordinates are supplied they need to be masked
        if self.coordinates is not None:
            coordinates = self.coordinates[self._combined_mask]
        else:
            coordinates = None

        # Return
        return self.__class__(mag=[rotdata[idx, :] for idx in range(self.n_features)],
                              err=[err[idx, :] for idx in range(self.n_features)],
                              extvec=extvec, coordinates=coordinates,
                              names=[x + "_rot" for x in self.features_names])

    # ----------------------------------------------------------------------
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

        combination_instances = []
        for c in all_c:
            cdata, cerror = [self.features[idx] for idx in c], [self.features_err[idx] for idx in c]
            cnames = [self.features_names[idx] for idx in c]
            extvec = [self.extvec.extvec[idx] for idx in c]
            combination_instances.append(self.__class__(mag=cdata, err=cerror, extvec=extvec,
                                                        coordinates=self.coordinates, names=cnames))

        # Return list of combinations.
        return combination_instances

    # ----------------------------------------------------------------------
    def _build_wcs_grid(self, proj_code="CAR", pixsize=5. / 60, **kwargs):
        """
        Method to build a WCS grid with a valid projection given a pixel scale.

        Parameters
        ----------
        proj_code : str, optional
            Any WCS projection code (e.g. CAR, TAN, etc.)
        pixsize : int, float, optional
            Pixel size of grid. Default is 10 arcminutes.
        kwargs
            Additioanl projection parameters if required (e.g. pv2_1=-30, pv2_2=0 for a given COE projection)

        Returns
        -------
        tuple
            Tuple containing the header and the world coordinate grids (lon and lat)

        """

        # Create header from data
        header = data2header(lon=self._lon, lat=self._lat, frame=self._frame, proj_code=proj_code, pixsize=pixsize,
                             **kwargs)

        # Get WCS
        mywcs = wcs.WCS(header=header)

        # Create image coordinate grid
        image_grid = np.meshgrid(np.arange(0, header["NAXIS1"], 1), np.arange(0, header["NAXIS2"], 1))

        # Convert to world coordinates and get WCS grid for this projection
        world_grid = mywcs.wcs_pix2world(image_grid[0], image_grid[1], 0)

        # Return header and grid
        return header, world_grid

    # ---------------------------------------------------------------------- #
    #                            Plotting methods                            #
    # ---------------------------------------------------------------------- #

    # ----------------------------------------------------------------------
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

    # ----------------------------------------------------------------------
    @property
    def _plotrange_world(self):
        """
        Convenience property to calcualte the plot range in world coordinates

        Returns
        -------
        list
            List with (left, right) and (bottom, top) tuple entries

        """

        # Build a wcs gruid with the defaults
        header, _ = self._build_wcs_grid()

        # Get footprint coordinates
        flon, flat = wcs.WCS(header=header).calc_footprint().T

        # Calculate centroid
        from pnicer.utils import centroid_sphere, distance_sky
        clon, clat = centroid_sphere(lon=flon, lat=flat, units="degree")

        # Maximize distances to the field edges in longitude
        left = flon[:2][np.argmax(distance_sky(lon1=flon[:2], lat1=0, lon2=clon, lat2=0, unit="degree"))]
        right = flon[2:][np.argmax(distance_sky(lon1=flon[2:], lat1=0, lon2=clon, lat2=0, unit="degree"))]

        # Maximize distances in latititude
        top, bottom = np.max(flat), np.min(flat)

        return [(left, right), (bottom, top)]

    # ----------------------------------------------------------------------
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

    # ----------------------------------------------------------------------
    @staticmethod
    def _finalize_plot(path=None):
        """
        Helper method to save or show plot.

        Parameters
        ----------
        path : str, optional
            If set, the path where the figure is saved

        """

        # Import matplotlib
        from matplotlib import pyplot as plt

        # Save or show figure
        if path is None:
            plt.show()
        else:
            plt.savefig(path, bbox_inches='tight')
        plt.close()

    # ----------------------------------------------------------------------
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
        axes = [plt.subplot(grid_plot[idx], projection=wcsaxes.WCS(header=header)) for idx in range(self.n_features)]

        # Generate labels
        llon, llat = "GLON" if "gal" in self._frame else "RA", "GLAT" if "gal" in self._frame else "DEC"

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

    # ----------------------------------------------------------------------
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
        self._finalize_plot(path=path)

    # ----------------------------------------------------------------------
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
            dens = mp_kde(grid=xgrid, data=data, bandwidth=grid_bw * 2, shape=x.shape, kernel=kernel, absolute=True,
                          sampling=2)

            # Plot result
            ax.imshow(dens.T, origin="lower", interpolation="nearest", extent=[l, h, l, h], cmap=cmap)

        # Modify tick labels
        caxes_delete_ticklabels(axes=axes, xfirst=False, xlast=True, yfirst=False, ylast=True)

        # Save or show figure
        self._finalize_plot(path=path)

    # ----------------------------------------------------------------------
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
        fig, axes, _, header = self._gridspec_world(pixsize=10 / 60, ax_size=ax_size, proj_code="CAR")

        # Get plot limits
        lim = wcs.WCS(header=header).wcs_world2pix(*self._plotrange_world, 0)

        # Loop over features and plot
        for idx in range(self.n_features):

            # Grab axes
            ax = axes[idx]

            ax.scatter(self._lon[self._features_masks[idx]][::skip], self._lat[self._features_masks[idx]][::skip],
                       transform=ax.get_transform(self._frame), **kwargs)

            # Set axes limits
            ax.set_xlim(lim[0])
            ax.set_ylim(lim[1])

        # Finalize plot
        self._finalize_plot(path=path)

    # ----------------------------------------------------------------------
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
        fig, axes, grid_world, header = self._gridspec_world(pixsize=bandwidth / 2, ax_size=ax_size, proj_code="CAR")

        # To avoid editor warning
        scale = 1

        # Loop over features and plot
        for idx in range(self.n_features):

            # Get density
            xgrid = np.vstack([grid_world[0].ravel(), grid_world[1].ravel()]).T
            data = np.vstack([self._lon[self._features_masks[idx]][::skip],
                              self._lat[self._features_masks[idx]][::skip]]).T
            dens = mp_kde(grid=xgrid, data=data, bandwidth=bandwidth, shape=grid_world[0].shape, kernel=kernel,
                          norm=False)

            # Norm and save scale (we want everything scaled to the same reference! In this case the first feature)
            if idx == 0:
                scale = np.max(dens)
            dens /= scale

            # Mask lowest level with NaNs
            dens[dens <= 1e-4] = np.nan

            # Show density
            axes[idx].imshow(dens, origin="lower", interpolation="nearest", cmap="viridis", vmin=0, vmax=1, **kwargs)

        # Save or show figure
        self._finalize_plot(path=path)

    # ---------------------------------------------------------------------- #
    #                          Main PNICER routines                          #
    # ---------------------------------------------------------------------- #

    # ----------------------------------------------------------------------
    def _pnicer_univariate(self, control):
        """
        Univariate implementation of PNICER.

        Parameters
        ----------
        control
            Control field instance.

        Returns
        -------
        tuple
            Tuple containing extinction, variance and de-reddened features.

        """

        # Some dummy checks
        self._check_class(ccls=control)
        if (self.n_features != 1) | (control.n_features != 1):
            raise ValueError("Only one feature allowed for this method")

        # Get mean and variance of control field
        cf_mean, cf_var = np.nanmean(control.features[0]), np.nanvar(control.features[0])

        # Calculate extinction for each source
        ext = (self.features[0] - cf_mean) / self.extvec.extvec[0]

        # Also the variance
        var = (self.features_err[0] ** 2 + cf_var) / self.extvec.extvec[0] ** 2

        # Intrinsic features
        color0 = np.full_like(ext, fill_value=float(cf_mean))
        color0[~np.isfinite(ext)] = np.nan

        # Return Extinction, variance and intrinsic features
        return ext, var, color0[np.newaxis, :]

    # ----------------------------------------------------------------------
    def _pnicer_multivariate(self, control, sampling, kernel):
        """
        Main PNICER routine to get extinction. This will return only the extinction values for data for which all
        features are available.

        Parameters
        ----------
        control
            Instance of control field data.
        sampling : int
            Sampling of grid relative to bandwidth of kernel.
        kernel : str
            Name of kernel to be used for density estimation. e.g. "epanechnikov" or "gaussian".

        Returns
        -------
        tuple(np.ndarray, np.ndarray, np.ndarray)
            Tuple containing extinction, variance and de-reddened features.

        """

        # Check instances
        if self.__class__ != control.__class__:
            raise ValueError("Input and control instance not compatible")

        # Avoid circular import
        from pnicer.user import Magnitudes

        # Let's rotate the data spaces
        science_rot, control_rot = self._rotate(), control._rotate()

        # Get bandwidth of kernel
        bandwidth = np.round(np.mean(np.nanmean(self.features_err, axis=1)), 2)

        # Determine bin widths for grid according to bandwidth and sampling
        bin_grid = bin_ext = np.float(bandwidth / sampling)

        # Now we build a grid from the rotated data for all components but the first
        grid_data = DataBase._build_feature_grid(data=np.vstack(science_rot.features)[1:, :], precision=bin_grid)

        # Create a grid to evaluate along the reddening vector
        grid_ext = np.arange(start=np.floor(min(control_rot.features[0])),
                             stop=np.ceil(max(control_rot.features[0])), step=bin_ext)

        # Now we combine those to get _all_ grid points
        xgrid = np.column_stack([np.tile(grid_ext, grid_data.shape[1]),
                                 np.repeat(grid_data, len(grid_ext), axis=1).T])

        # With our grid, we evaluate the density on it for the control field (!)
        dens = mp_kde(grid=xgrid, data=np.vstack(control_rot.features).T, bandwidth=bandwidth, kernel=kernel,
                      absolute=True, sampling=sampling)

        # Get all unique vectors
        dens_vectors = dens.reshape([grid_data.shape[1], len(grid_ext)])

        # Calculate weighted average and standard deviation for each vector
        grid_mean, grid_var = [], []
        for vec in dens_vectors:

            # In case there are too few stars
            if np.sum(vec) < 3:
                grid_mean.append(np.nan)
                grid_var.append(np.nan)
            else:
                # Get weighted average position along vector and the weighted variance
                a, b = weighted_avg(values=grid_ext, weights=vec)
                grid_mean.append(a)
                grid_var.append(b / self.extvec._extinction_norm)  # The normalisation converts this to extinction

        # Convert to arrays
        grid_var = np.array(grid_var)
        grid_mean = np.array(grid_mean)

        # Let's get the nearest neighbor grid point for each source
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(grid_data.T)
        _, indices = nbrs.kneighbors(np.vstack(science_rot.features)[1:, :].T)
        indices = indices[:, 0]

        # Inverse rotation of grid to get intrinsic features
        intrinsic = self.extvec._rotmatrix_inv.dot(np.vstack([grid_mean[indices], grid_data[:, indices]]))

        # In case of Magnitudes we need to calculate the intinsic value
        if isinstance(self, Magnitudes):
            color0 = np.array([intrinsic[i] - intrinsic[i + 1] for i in range(self.n_features - 1)])
        else:
            color0 = intrinsic

        # Now we have the instrisic colors for each vector and indices for all sources.
        # It's time to calculate the extinction. :)
        ext = (science_rot.features[0] - grid_mean[indices]) / self.extvec._extinction_norm
        var = grid_var[indices]

        # Lastly we put all the extinction measurements back into a full array
        out_ext = np.full(self.n_data, fill_value=np.nan, dtype=float)
        out_var = np.full(self.n_data, fill_value=np.nan, dtype=float)
        out_col = np.full([len(color0), self.n_data], fill_value=np.nan, dtype=float)

        # Output data for all sources
        out_ext[self._combined_mask], out_var[self._combined_mask] = ext, var
        out_col[:, self._combined_mask] = color0

        # Return Extinction, variance and intrinsic features
        return out_ext, out_var, out_col

    # ----------------------------------------------------------------------
    def _pnicer_combinations(self, control, comb, sampling, kernel):
        """
        PNICER base implementation for combinations. Basically calls the pnicer_single implementation for all
        combinations. The output extinction is then the one with the smallest error from all combinations.

        Parameters
        ----------
        control
            Instance of control field data.
        comb
            Zip object of combinations to use.
        sampling : int
            Sampling of grid relative to bandwidth of kernel.
        kernel : str
            Name of kernel to be used for density estimation. e.g. "epanechnikov" or "gaussian".

        Returns
        -------
        Extinction
            Extinction instance with the calcualted extinction and errors.

        """

        # Avoid circular import
        from pnicer.user import Magnitudes, Colors

        # Check instances
        self._check_class(ccls=control)

        # Dummy assertion for editor
        assert (isinstance(self, Magnitudes) | isinstance(self, Colors))

        # We loop over all combinations
        all_ext, all_var, all_n, all_color0, names = [], [], [], [], []

        # Create intrinsic color dictionary
        color0_dict = {k: [] for k in self.colors_names}
        color0_weig = {k: [] for k in self.colors_names}

        # Here we loop over color combinations since this is faster
        i = 0
        for sc, cc in comb:

            # Type assertion to not raise editor warning
            assert (isinstance(sc, Magnitudes) | isinstance(sc, Colors))

            # Depending on number of features, choose algorithm
            if sc.n_features == 1:
                ext, var, color0 = sc._pnicer_univariate(control=cc)
            else:
                ext, var, color0 = sc._pnicer_multivariate(control=cc, sampling=sampling, kernel=kernel)

            # Put the intrinsic color into the dictionary
            for c, cidx in zip(sc.colors_names, range(len(sc.colors_names))):
                try:
                    color0_dict[c].append(color0[cidx])
                    color0_weig[c].append(sc.n_features ** 2)
                except KeyError:
                    pass

            # Append data
            all_ext.append(ext)
            all_var.append(var)
            all_n.append(sc.n_features)
            names.append("(" + ",".join(sc.features_names) + ")")
            i += 1

        # Loop through color0_dict and calculate average
        for key in color0_dict.keys():
            # Get weighted average
            values = np.ma.masked_invalid(np.array(color0_dict[key]))
            color0_dict[key] = np.ma.average(values, axis=0, weights=color0_weig[key]).data
            # Optionally with a median
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # color0_dict[key] = np.nanmedian(np.array(color0_dict[key]), axis=0)

        # Get final list of intrinsic colors while forcing the original order
        self._color0 = []
        for key in self.colors_names:
            self._color0.append(color0_dict[key])

        # Convert to arrays and save combination data
        all_ext = np.array(all_ext)
        all_var = np.array(all_var)

        self._ext_combinations = all_ext.copy()
        self._var_combinations = all_var.copy()
        self._combination_names = names
        self._n_combinations = i

        # TODO: Weighted average or min?
        # # calculate weighted average
        # weight = np.array(all_n)[:, None] / all_var
        # ext = np.nansum(all_ext * weight, axis=0) / np.nansum(weight, axis=0)
        # var = np.nansum(all_var * weight**2, axis=0) / np.nansum(weight, axis=0)**2
        # return Extinction(db=db, extinction=ext, variance=var)

        # Chose extinction as minimum error across all combinations
        all_var[~np.isfinite(all_var)] = 100 * np.nanmax(all_var)
        ext = all_ext[np.argmin(all_var, axis=0), np.arange(self.n_data)]
        var = all_var[np.argmin(all_var, axis=0), np.arange(self.n_data)]

        # Make error cut
        ext[var > 10] = var[var > 10] = np.nan

        # Return Extinction instance
        from pnicer.extinction import Extinction
        return Extinction(db=self, extinction=ext, variance=var, color0=np.array(self._color0))


# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #
class ExtinctionVector:

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

    # ----------------------------------------------------------------------
    @property
    def n_dimensions(self):
        """
        Number of dimensions.

        Returns
        -------
        int

        """

        return len(self.extvec)

    # ----------------------------------------------------------------------
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

    # ----------------------------------------------------------------------
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

    # ----------------------------------------------------------------------
    __rotmatrix = None

    @property
    def _rotmatrix(self):
        """
        Property to hold the rotation matrix for all extinction components of the current instance.

        Returns
        -------
        np.ndarray
            Rotation matrix for the extinction vector of the instance.

        """

        # Check if already determined
        if self.__rotmatrix is not None:
            return self.__rotmatrix

        self.__rotmatrix = ExtinctionVector._get_rotmatrix(self.extvec)
        return self.__rotmatrix

    # ----------------------------------------------------------------------
    @property
    def _rotmatrix_inv(self):
        """
        Inverted rotation matrix.

        Returns
        -------
        np.ndarray

        """

        return self._rotmatrix.T

    # ----------------------------------------------------------------------
    @property
    def _extinction_rot(self):
        """
        Calculates the rotated extinction vector.

        Returns
        -------
        np.ndarray

        """

        return self._rotmatrix.dot(self.extvec)

    # ----------------------------------------------------------------------
    @property
    def _extinction_norm(self):
        """
        Normalization component for projected extinction vector.

        Returns
        -------
        float

        """

        return self._extinction_rot[0]
