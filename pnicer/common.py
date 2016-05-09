# ----------------------------------------------------------------------
# Import stuff
import wcsaxes
import warnings
import numpy as np

from astropy import wcs
from itertools import combinations
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
# noinspection PyPackageRequirements
from sklearn.neighbors import NearestNeighbors
from matplotlib.ticker import MaxNLocator, AutoMinorLocator

# from pnicer.user import Magnitudes, Colors
from pnicer.utils import weighted_avg, axes_combinations, mp_kde


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# noinspection PyProtectedMember
class DataBase:
    def __init__(self, mag, err, extvec, lon=None, lat=None, names=None):
        """
        Basic Data class which provides the foundation for extinction measurements.

        Parameters
        ----------
        mag : iterable
            List of magnitude arrays. All arrays must have the same length!
        err : iterable
            List off magnitude error arrays.
        extvec : iterable
            List holding the extinction components for each magnitude
        lon : iterable, optional
            Longitude of coordinates for each source; in decimal degrees!
        lat : iterable, optional
            Latitude of coordinates for each source; in decimal degrees!
        names : list
            List of magnitude (feature) names.

        """

        # Set attributes
        self.features = mag
        self.features_err = err
        self.lon = lon
        self.lat = lat
        self.features_names = names
        self.n_features = len(mag)
        self.extvec = ExtinctionVector(extvec=extvec)

        # Define combination properties determined while running PNICER
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
        if (lon is not None) | (lat is not None):
            if (len(lon) != len(lat)) | (len(lon) != len(self.features[0])):
                raise ValueError("Input coordinates do not match!")

    # ---------------------------------------------------------------------- #
    #                         Some useful properties                         #
    # ---------------------------------------------------------------------- #

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
    # noinspection PyTypeChecker
    @property
    def plotrange(self):
        """
        Convenience property to calculate a plot range for all provided features.

        Returns
        -------
        list
            List of plot ranges.

        """

        return [(np.floor(np.percentile(x[m], 0.01)), np.ceil(np.percentile(x[m], 99.99)))
                for x, m in zip(self.features, self.features_masks)]

    # ----------------------------------------------------------------------
    @property
    def features_masks(self):
        """
        Provides a list with masks for each given feature.


        Returns
        -------
        iterable
            List of masks.

        """

        return [np.isfinite(m) & np.isfinite(e) for m, e in zip(self.features, self.features_err)]

    # ----------------------------------------------------------------------
    @property
    def combined_mask(self):
        """
        Combines all feature masks into a single mask.

        Returns
        -------
        np.ndarray
            Combined mask.

        """

        return np.prod(np.vstack(self.features_masks), axis=0, dtype=bool)

    # ---------------------------------------------------------------------- #
    #                         Static helper methods                          #
    # ---------------------------------------------------------------------- #

    # ----------------------------------------------------------------------
    @staticmethod
    def _round_partial(data, precision):
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
    @staticmethod
    def _build_grid(data, precision):
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
        grid_data = DataBase._round_partial(data=data, precision=precision).T

        # Get unique positions for coordinates
        dummy = np.ascontiguousarray(grid_data).view(np.dtype((np.void, grid_data.dtype.itemsize * grid_data.shape[1])))
        _, idx = np.unique(dummy, return_index=True)

        return grid_data[np.sort(idx)].T

    # ---------------------------------------------------------------------- #
    #                        Useful Instance methods                         #
    # ---------------------------------------------------------------------- #

    # ----------------------------------------------------------------------
    def rotate(self):
        """
        Method to rotate data space with the given extinction vector. Only finite data are transmitted intp the new
        data space.

        Returns
        -------
            New instance with rotated data.

        """

        data = np.vstack(self.features).T[self.combined_mask].T
        err = np.vstack(self.features_err).T[self.combined_mask].T

        # Rotate data
        rotdata = self.extvec._rotmatrix.dot(data)

        # Rotate extinction vector
        extvec = self.extvec._extinction_rot

        # In case no coordinates are supplied
        if self.lon is not None:
            lon = self.lon[self.combined_mask]
        else:
            lon = None
        if self.lat is not None:
            lat = self.lat[self.combined_mask]
        else:
            lat = None

        # Return
        return self.__class__(mag=[rotdata[idx, :] for idx in range(self.n_features)],
                              err=[err[idx, :] for idx in range(self.n_features)],
                              extvec=extvec, lon=lon, lat=lat,
                              names=[x + "_rot" for x in self.features_names])

    # ----------------------------------------------------------------------
    def all_combinations(self, idxstart):
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
                                                        lon=self.lon, lat=self.lat, names=cnames))

        # Return list of combinations.
        return combination_instances

    # ----------------------------------------------------------------------
    def build_wcs_grid(self, frame, pixsize=10. / 60):
        """
        Method to build a WCS grid with a valid projection given a pixel scale.

        Parameters
        ----------
        frame : str
            Coordinate frame. 'equatorial' or 'galactic'.
        pixsize : int, float, optional
            Pixel size of grid. Default is 10 arcminutes.

        Returns
        -------
        tuple
            Tuple containing the header, the longitude and the latitude grid.

        """

        # TODO: Check if this method can be made better (e.g. include projcode)

        if frame == "equatorial":
            ctype = ["RA---COE", "DEC--COE"]
        elif frame == "galactic":
            ctype = ["GLON-COE", "GLAT-COE"]
        else:
            raise KeyError("frame must be'galactic' or 'equatorial'")

        # Calculate range of grid to 0.1 degree precision
        lon_range = [np.floor(np.min(self.lon) * 10) / 10, np.ceil(np.max(self.lon) * 10) / 10]
        lat_range = [np.floor(np.min(self.lat) * 10) / 10, np.ceil(np.max(self.lat) * 10) / 10]

        naxis1 = np.ceil((lon_range[1] - lon_range[0]) / pixsize)
        naxis2 = np.ceil((lat_range[1] - lat_range[0]) / pixsize)

        # Initialize WCS
        mywcs = wcs.WCS(naxis=2)

        # Set projection parameters
        mywcs.wcs.crpix = [naxis1 / 2., naxis2 / 2.]
        mywcs.wcs.cdelt = np.array([-pixsize, pixsize])
        mywcs.wcs.crval = [(lon_range[0] + lon_range[1]) / 2., (lat_range[0] + lat_range[1]) / 2.]
        mywcs.wcs.ctype = ctype

        mywcs.wcs.set_pv([(2, 1, np.around(np.median(self.lat), 1))])

        # Make header
        myheader = mywcs.to_header()

        # Create image coordinates
        x_pix = np.arange(0, naxis1, 1)
        y_pix = np.arange(0, naxis2, 1)
        # ...and grid
        xy_coo = np.meshgrid(x_pix, y_pix)

        # Convert to world coordinates and get WCS grid for this projection
        world = mywcs.wcs_pix2world(xy_coo[0], xy_coo[1], 0)

        return myheader, world[0], world[1]

    # ---------------------------------------------------------------------- #
    #                          Main PNICER routines                          #
    # ---------------------------------------------------------------------- #

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
        assert self.__class__ == control.__class__, "Input and control instance not compatible"

        # Let's rotate the data spaces
        science_rot, control_rot = self.rotate(), control.rotate()

        # Get bandwidth of kernel
        bandwidth = np.round(np.mean(np.nanmean(self.features_err, axis=1)), 2)

        # Determine bin widths for grid according to bandwidth and sampling
        bin_grid = bin_ext = np.float(bandwidth / sampling)

        # Now we build a grid from the rotated data for all components but the first
        grid_data = DataBase._build_grid(data=np.vstack(science_rot.features)[1:, :], precision=bin_grid)

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

        from pnicer.user import Magnitudes

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
        out_ext[self.combined_mask], out_var[self.combined_mask] = ext, var
        out_col[:, self.combined_mask] = color0

        return out_ext, out_var, out_col

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

        if self.__class__ != control.__class__:
            raise ValueError("Instance and control class do not match")
        if (self.n_features != 1) | (control.n_features != 1):
            raise ValueError("Only one feature allowed for this method")

        # Get mean and std of control field
        # TODO: Average or weighted average
        # cf_mean, cf_var = weighted_avg(values=control.features[0], weights=control.features_err[0])
        cf_mean, cf_var = np.nanmean(control.features[0]), np.nanvar(control.features[0])

        # Calculate extinctions
        ext = (self.features[0] - cf_mean) / self.extvec.extvec[0]
        var = (self.features_err[0] ** 2 + cf_var) / self.extvec.extvec[0] ** 2
        color0 = np.full_like(ext, fill_value=float(cf_mean))
        color0[~np.isfinite(ext)] = np.nan

        return ext, var, color0[np.newaxis, :]

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

        # Instance assertions
        assert self.__class__ == control.__class__, "instance and control class do not match"
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
    #                            Plotting methods                            #
    # ---------------------------------------------------------------------- #

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

        # Set default axis size
        if ax_size is None:
            ax_size = [3, 3]

        # Get axes for figure
        fig, axes = axes_combinations(ndim=self.n_features, ax_size=ax_size)

        # Get 2D combination indices
        for idx, ax in zip(combinations(range(self.n_features), 2), axes):

            ax.scatter(self.features[idx[1]][::skip], self.features[idx[0]][::skip], lw=0, s=5, alpha=0.1, **kwargs)

            # We need a square grid!
            l, h = np.min([x[0] for x in self.plotrange]), np.max([x[1] for x in self.plotrange])

            # Ranges
            ax.set_xlim(l, h)
            ax.set_ylim(l, h)

            # Axis labels
            if ax.get_position().x0 < 0.11:
                ax.set_ylabel(self.features_names[idx[0]])
            if ax.get_position().y0 < 0.11:
                ax.set_xlabel(self.features_names[idx[1]])

        # Save or show figure
        if path is None:
            plt.show()
        else:
            plt.savefig(path, bbox_inches='tight')
        plt.close()

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

        # Set default axis size
        if ax_size is None:
            ax_size = [3, 3]

        # Create figure
        fig, axes = axes_combinations(self.n_features, ax_size=ax_size)

        # Get 2D combination indices
        for idx, ax in zip(combinations(range(self.n_features), 2), axes):

            # Get clean data from the current combination
            mask = np.prod(np.vstack([self.features_masks[idx[0]], self.features_masks[idx[1]]]), axis=0, dtype=bool)

            # We need a square grid!
            l, h = np.min([x[0] for x in self.plotrange]), np.max([x[1] for x in self.plotrange])

            x, y = np.meshgrid(np.arange(start=l, stop=h, step=grid_bw), np.arange(start=l, stop=h, step=grid_bw))

            # Get density
            data = np.vstack([self.features[idx[0]][mask], self.features[idx[1]][mask]]).T
            xgrid = np.vstack([x.ravel(), y.ravel()]).T
            dens = mp_kde(grid=xgrid, data=data, bandwidth=grid_bw * 2, shape=x.shape, kernel=kernel, absolute=True,
                          sampling=2)

            # Show result
            ax.imshow(np.sqrt(dens.T), origin="lower", interpolation="nearest", extent=[l, h, l, h], cmap=cmap)

            # Modify labels
            if idx[0] == 0:
                xticks = ax.xaxis.get_major_ticks()
                xticks[-1].set_visible(False)
            if idx[1] == np.max(idx):
                yticks = ax.yaxis.get_major_ticks()
                yticks[-1].set_visible(False)

            # Axis labels
            if ax.get_position().x0 < 0.11:
                ax.set_ylabel("$" + self.features_names[idx[0]] + "$")
            if ax.get_position().y0 < 0.11:
                ax.set_xlabel("$" + self.features_names[idx[1]] + "$")

        # Save or show figure
        if path is None:
            plt.show()
        else:
            plt.savefig(path, bbox_inches="tight")
        plt.close()

    # ----------------------------------------------------------------------
    def plot_spatial_kde(self, frame, pixsize=10 / 60, path=None, kernel="epanechnikov", skip=1, cmap=None):
        """
        Plot source densities for features

        Parameters
        ----------
        frame : str
            Coordinate frame. 'equatorial' or 'galactic'.
        pixsize : int, float, optional
            Pixel size of grid.
        path : str, optional
            File path if it should be saved. e.g. '/path/to/image.png'. Default is None.
        kernel : str, optional
            Name of kernel for KDE. e.g. 'epanechnikov' or 'gaussian'. Default is 'epanechnikov'.
        skip : int, optional
            Skip every n-th source for faster plotting. Default is 1.
        cmap : str, optional
            Colormap to be used in plot. Default is None.

        """

        # Set cmap
        if cmap is None:
            try:
                cmap = plt.get_cmap("viridis")
            except ValueError:
                cmap = plt.get_cmap("gist_heat_r")

        # Get a WCS grid
        header, lon_grid, lat_grid = self.build_wcs_grid(frame=frame, pixsize=pixsize)

        # Get aspect ratio
        ar = lon_grid.shape[0] / lon_grid.shape[1]

        # Determine number of panels
        n_panels = [np.floor(np.sqrt(self.n_features)).astype(int), np.ceil(np.sqrt(self.n_features)).astype(int)]
        if n_panels[0] * n_panels[1] < self.n_features:
            n_panels[n_panels.index(min(n_panels))] += 1

        # Create grid
        plt.figure(figsize=[10 * n_panels[0], 10 * n_panels[1] * ar])
        grid = GridSpec(ncols=n_panels[0], nrows=n_panels[1], bottom=0.05, top=0.95, left=0.05, right=0.95,
                        hspace=0.2, wspace=0.2)

        # To avoid editor warning
        scale = 1

        # Loop over features and plot
        for idx in range(self.n_features):

            # Add axes
            ax = plt.subplot(grid[idx], projection=wcsaxes.WCS(header=header))

            # Get density
            xgrid = np.vstack([lon_grid.ravel(), lat_grid.ravel()]).T
            data = np.vstack([self.lon[self.features_masks[idx]][::skip], self.lat[self.features_masks[idx]][::skip]]).T
            dens = mp_kde(grid=xgrid, data=data, bandwidth=pixsize * 2, shape=lon_grid.shape, kernel=kernel)

            # Norm and save scale
            if idx == 0:
                scale = np.max(dens)

            dens /= scale

            # Plot density
            dens[dens < 0.1] = np.nan
            ax.imshow(dens, origin="lower", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)

        # Save or show figure
        if path is None:
            plt.show()
        else:
            plt.savefig(path, bbox_inches="tight")
        plt.close()

    # ----------------------------------------------------------------------
    def plot_kde_extinction_combinations(self, path=None, sampling=16):
        """
        Plot histogram of extinctions for all combinations. Requires PNICER to be run beforehand.

        Parameters
        ----------
        path : str, optional
            File path if it should be saved. e.g. '/path/to/image.png'. Default is None.
        sampling : int, optional
            Sampling factor of grid (the larger, the more samples). Default is 16.

        """

        # TODO: Improve or remove!

        # If PNICER has not yet been run, raise error
        if self._ext_combinations is None:
            raise RuntimeError("You have to run PNICER first!")

        # Determine number of panels
        n_panels = [np.floor(np.sqrt(self._n_combinations)).astype(int),
                    np.ceil(np.sqrt(self._n_combinations)).astype(int)]
        if n_panels[0] * n_panels[1] < self._n_combinations:
            n_panels[n_panels.index(min(n_panels))] += 1

        # Determine plot range
        # noinspection PyTypeChecker
        ax1_range = [DataBase._round_partial(np.nanmean(self._ext_combinations) -
                                             3 * np.nanstd(self._ext_combinations), 0.1),
                     DataBase._round_partial(np.nanmean(self._ext_combinations) +
                                             3.5 * np.nanstd(self._ext_combinations), 0.1)]
        # noinspection PyTypeChecker
        ax2_range = [0., DataBase._round_partial(np.nanmean(self._var_combinations) +
                                                 3.5 * np.nanstd(self._var_combinations), 0.1)]

        plt.figure(figsize=[5 * n_panels[1], 5 * n_panels[0] * 0.5])
        grid = GridSpec(ncols=n_panels[1], nrows=n_panels[0], bottom=0.1, top=0.9, left=0.1, right=0.9,
                        hspace=0, wspace=0)

        for idx in range(self._n_combinations):

            # Get densities for extinction
            ext = self._ext_combinations[idx, :]
            # noinspection PyTypeChecker
            ext = ext[np.isfinite(ext)]

            # Estimate bandwidth with Silverman's rule
            bandwidth_ext = np.float(1.06 * np.std(ext) * len(ext) ** (-1 / 5.))

            # Generate grid and evaluate densities
            grid_ext = np.arange(np.floor(ax1_range[0]), np.ceil(ax1_range[1]), bandwidth_ext / sampling)
            dens_ext = mp_kde(grid=grid_ext, data=ext, bandwidth=bandwidth_ext, shape=None,
                              kernel="gaussian", absolute=True, sampling=sampling)

            # Get densities for extinction error
            exterr = self._var_combinations[idx, :]
            # noinspection PyTypeChecker
            exterr = exterr[np.isfinite(exterr)]

            # Estimate bandwidth with Silverman's rule
            bandwidth_exterr = np.float(1.06 * np.std(exterr) * len(exterr) ** (-1 / 5.))

            # Generate grid and evaluate densities
            grid_exterr = np.arange(np.floor(ax2_range[0]), np.ceil(ax2_range[1]), bandwidth_exterr / sampling)
            dens_exterr = mp_kde(grid=grid_exterr, data=exterr[np.isfinite(exterr)], bandwidth=bandwidth_exterr,
                                 shape=None, kernel="gaussian", absolute=True, sampling=sampling)

            # Add axes
            ax1 = plt.subplot(grid[idx])
            ax2 = ax1.twiny()

            # Plot
            ax1.plot(grid_ext, dens_ext, lw=3, alpha=0.7, color="#cb181d")
            ax2.plot(grid_exterr, dens_exterr, lw=3, alpha=0.7, color="#2171b5")

            for ax in [ax1, ax2]:
                ax.xaxis.set_major_locator(MaxNLocator(5))
                ax.yaxis.set_major_locator(MaxNLocator(5))
                ax.xaxis.set_minor_locator(AutoMinorLocator(5))
                ax.yaxis.set_minor_locator(AutoMinorLocator(5))

            # Set and delete labels
            if idx >= n_panels[0] * n_panels[1] - n_panels[1]:
                ax1.set_xlabel("Extinction (mag)")
            else:
                ax1.axes.xaxis.set_ticklabels([])
            if idx % n_panels[1] == 0:
                ax1.set_ylabel("N")
            else:
                ax1.axes.yaxis.set_ticklabels([])
            if idx < n_panels[1]:
                ax2.set_xlabel("Error (mag)")
            else:
                ax2.axes.xaxis.set_ticklabels([])

            # TODO: Write stand-alone method to remove ticks from axes
            # Delete first and last label
            xticks = ax1.xaxis.get_major_ticks()
            xticks[-1].set_visible(False)
            xticks = ax2.xaxis.get_major_ticks()
            xticks[-1].set_visible(False)
            yticks = ax1.yaxis.get_major_ticks()
            yticks[-1].set_visible(False)
            yticks = ax2.yaxis.get_major_ticks()
            yticks[-1].set_visible(False)

        # Save or show figure
        if path is None:
            plt.show()
        else:
            plt.savefig(path, bbox_inches="tight")
        plt.close()


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
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
        self.n_dimensions = len(extvec)

    # ---------------------------------------------------------------------- #
    #                             Static methods                             #
    # ---------------------------------------------------------------------- #

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
        Simple property to hold the rotation matrix for all extinction components of the current instance.

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
    # Inverted rotation matrix
    __rotmatrix_inv = None

    @property
    def _rotmatrix_inv(self):
        """
        Inverted rotation matrix.

        Returns
        -------
        np.ndarray

        """

        # Check if already determined
        if self.__rotmatrix_inv is not None:
            return self.__rotmatrix_inv

        self.__rotmatrix_inv = self._rotmatrix.T
        return self.__rotmatrix_inv

    # ----------------------------------------------------------------------
    __extinction_rot = None

    @property
    def _extinction_rot(self):
        """
        Calculates the rotated extinction vector for current isntance.

        Returns
        -------
        np.ndarray

        """

        # Check if already determined
        if self.__extinction_rot is not None:
            return self.__extinction_rot

        self.__extinction_rot = self._rotmatrix.dot(self.extvec)
        return self.__extinction_rot

    # ----------------------------------------------------------------------
    __extinction_norm = None

    @property
    def _extinction_norm(self):
        """
        Normalization component for projected extinction vector.

        Returns
        -------
        float

        """
        # Check if already determined
        if self.__extinction_norm is not None:
            return self.__extinction_norm

        self.__extinction_norm = self._extinction_rot[0]
        return self.__extinction_norm
