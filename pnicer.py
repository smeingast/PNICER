# ----------------------------------------------------------------------
# Import stuff
import wcsaxes
import warnings
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt

from astropy import wcs
from astropy.io import fits
from multiprocessing import Pool
from matplotlib.pyplot import GridSpec
from itertools import combinations, repeat
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
# noinspection PyPackageRequirements
from sklearn.neighbors import KernelDensity, NearestNeighbors

# TODO: Change all assertions into error statements


# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #
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


# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #
class Magnitudes(DataBase):
    def __init__(self, mag, err, extvec, lon=None, lat=None, names=None):
        """
        Main class for users. Includes PNICER and NICER.

        Parameters
        ----------
        mag : list
            List of magnitude arrays. All arrays must have the same length.
        err : list
            List off magnitude error arrays.
        extvec : list
            List holding the extinction components for each magnitude.
        lon : np.ndarray, optional
            Longitude coordinates for each source.
        lat : np.ndarray, optional
            Latitude coordinates for each source.
        names : list, optional
            List of magnitude (feature) names.

        """

        # Call parent
        super(Magnitudes, self).__init__(mag=mag, err=err, extvec=extvec, lon=lon, lat=lat, names=names)

        # Create color names
        self.colors_names = [self.features_names[k - 1] + "-" + self.features_names[k]
                             for k in range(1, self.n_features)]

    # ----------------------------------------------------------------------
    def mag2color(self):
        """
        Method to convert magnitude to color instances.

        Returns
        -------
        Colors
            Colors instance.

        """

        # Calculate colors
        colors = [self.features[k - 1] - self.features[k] for k in range(1, self.n_features)]

        # Calculate color errors
        colors_error = [np.sqrt(self.features_err[k - 1] ** 2 + self.features_err[k] ** 2)
                        for k in range(1, self.n_features)]

        # Color names
        color_extvec = [self.extvec.extvec[k - 1] - self.extvec.extvec[k] for k in range(1, self.n_features)]

        # Return Colors instance
        return Colors(mag=colors, err=colors_error, extvec=color_extvec,
                      lon=self.lon, lat=self.lat, names=self.colors_names)

    # ----------------------------------------------------------------------
    def color_combinations(self):
        """
        Calculates a list of Colors instances for all combinations.

        Returns
        -------
        iterable
            List of Colors instances.

        """

        # Get all colors and then all combinations of colors
        return self.mag2color().all_combinations(idxstart=1)

    # ----------------------------------------------------------------------
    def pnicer(self, control, sampling=2, kernel="epanechnikov", add_colors=False):
        """
        Main PNICER method for magnitudes. Includes options to use combinations for input features, or convert them
        to colors.

        Parameters
        ----------
        control
            Control field instance. Same class as self.
        sampling : int, optional
            Sampling of grid relative to bandwidth of kernel. Default is 2.
        kernel : str, optional
            Name of kernel for KDE. e.g. 'epanechnikov' or 'gaussian'. Default is 'epanechnikov'.
        add_colors : bool, optional
            Whether to also include the colors generated from the given magnitudes.

        Returns
        -------
        Extinction
            Extinction instance with the calcualted extinction and errors.

        """

        if add_colors:
            # To create a color, we need at least two features
            assert self.n_features >= 2, "To use colors, at least two features are required"
            comb = zip(self.all_combinations(idxstart=2) + self.color_combinations(),
                       control.all_combinations(idxstart=2) + control.color_combinations())
        else:
            comb = zip(self.all_combinations(idxstart=2), control.all_combinations(idxstart=2))

        return self._pnicer_combinations(control=control, comb=comb, sampling=sampling, kernel=kernel)

    # ----------------------------------------------------------------------
    def nicer(self, control, n_features=None):
        """
        NICER routine as descibed in Lombardi & Alves 2001. Generalized for arbitrary input magnitudes

        Parameters
        ----------
        control
            Control field instance. Same class as self.
        n_features : int, optional
            If set, return only extinction values for sources with data for 'n' features.

        Returns
        -------
        Extinction
            Extinction instance with the calcualted extinction and errors.

        """

        # Some assertions
        if self.__class__ != control.__class__:
            raise ValueError("control and instance class do not match")
        if self.n_features != control.n_features:
            raise ValueError("Number of features do not match")

        # Features to be required can only be as much as input features
        if n_features is not None:
            assert n_features <= self.n_features, "Can't require more features than there are available"
            assert n_features > 0, "Must require at least one feature"

        # Get reddening vector
        k = [x - y for x, y in zip(self.extvec.extvec[:-1], self.extvec.extvec[1:])]

        # Calculate covariance matrix of control field
        cov_cf = np.ma.cov([np.ma.masked_invalid(control.features[l]) - np.ma.masked_invalid(control.features[l + 1])
                            for l in range(self.n_features - 1)])

        # Get intrisic color of control field
        color_0 = [np.nanmean(control.features[l] - control.features[l + 1]) for l in range(control.n_features - 1)]

        # Replace NaN errors with a large number
        errors = []
        for e in self.features_err:
            a = e.copy()
            a[~np.isfinite(a)] = 1000
            errors.append(a)

        # Calculate covariance matrix of errors in the science field
        cov_er = np.zeros([self.n_data, self.n_features - 1, self.n_features - 1])
        for i in range(self.n_features - 1):
            # Diagonal
            cov_er[:, i, i] = errors[i] ** 2 + errors[i + 1] ** 2
            # Other entries
            if i > 0:
                cov_er[:, i, i - 1] = cov_er[:, i - 1, i] = -errors[i] ** 2

        # Total covariance matrix
        cov = cov_cf + cov_er

        # Invert
        cov_inv = np.linalg.inv(cov)

        # Get b from the paper (equ. 12)
        upper = np.dot(cov_inv, k)
        b = upper.T / np.dot(k, upper.T)

        # Get colors
        scolors = np.array([self.features[l] - self.features[l + 1] for l in range(self.n_features - 1)])

        # Get those with no good color value at all
        bad_color = np.all(np.isnan(scolors), axis=0)

        # Write finite value for all NaNs (this makes summing later easier)
        scolors[~np.isfinite(scolors)] = 0

        # Put back NaNs for those with only bad colors
        scolors[:, bad_color] = np.nan

        # Equation 13 in the NICER paper
        ext = b[0, :] * (scolors[0, :] - color_0[0])
        for i in range(1, self.n_features - 1):
            ext += b[i, :] * (scolors[i, :] - color_0[i])

        # Calculate variance (has to be done in loop due to RAM issues!)
        first = np.array([np.dot(cov.data[idx, :, :], b.data[:, idx]) for idx in range(self.n_data)])
        var = np.array([np.dot(b.data[:, idx], first[idx, :]) for idx in range(self.n_data)])
        # Now we have to mask the large variance data again
        # TODO: This is the same as 1/denominator from Equ. 12
        var[~np.isfinite(ext)] = np.nan

        # Generate intrinsic source color list
        color_0 = np.repeat(color_0, self.n_data).reshape([len(color_0), self.n_data])
        color_0[:, ~np.isfinite(ext)] = np.nan

        if n_features is not None:
            mask = np.where(np.sum(np.vstack(self.features_masks), axis=0, dtype=int) < n_features)[0]
            ext[mask] = var[mask] = color_0[:, mask] = np.nan

        # ...and return :) Here, a Colors instance is returned!
        return Extinction(db=self.mag2color(), extinction=ext.data, variance=var, color0=color_0)

    # ----------------------------------------------------------------------
    def get_beta_lines(self, base_keys, fit_key, control, kappa=2, sigma=3, err_iter=1000):
        # TODO: Improve or remove!

        # Some assertions (quite some actually, haha)
        assert (isinstance(base_keys, tuple)) & (len(base_keys) == 2), " base_keys must be tuple with two entries"
        assert isinstance(fit_key, str), "fit_key must be string"
        assert isinstance(control, Magnitudes), "control must be magnitude instance"
        assert (kappa >= 0) & isinstance(kappa, int), "kappa must be non-zero positive integer"
        assert sigma > 0, "sigma must positive"
        assert fit_key in self.features_names, "fit_key not found"
        assert (base_keys[0] in self.features_names) & (base_keys[1] in self.features_names), "base_keys not found"

        # Get indices of requested keys
        base_idx = (self.features_names.index(base_keys[0]), self.features_names.index(base_keys[1]))
        fit_idx = self.features_names.index(fit_key)

        # Create common filter for all current filters
        smask = np.prod(np.vstack([self.features_masks[i] for i in base_idx + (fit_idx,)]), axis=0, dtype=bool)
        cmask = np.prod(np.vstack([control.features_masks[i] for i in base_idx + (fit_idx,)]), axis=0, dtype=bool)

        # Shortcuts for control field terms which are not clipped
        xdata_control = control.features[base_idx[0]][cmask] - control.features[base_idx[1]][cmask]
        ydata_control = control.features[base_idx[1]][cmask] - control.features[fit_idx][cmask]

        var_err_control = np.mean(control.features_err[base_idx[0]][cmask]) ** 2 + \
            np.mean(control.features_err[base_idx[1]][cmask]) ** 2
        cov_err_control = -np.mean(control.features_err[base_idx[1]][cmask]) ** 2
        var_control = np.var(xdata_control)
        cov_control = get_covar(xdata_control, ydata_control)

        x1_sc, x2_sc = self.features[base_idx[0]][smask], self.features[base_idx[1]][smask]
        x1_sc_err, x2_sc_err = self.features_err[base_idx[0]][smask], self.features_err[base_idx[1]][smask]
        y1_sc, y2_sc = x2_sc, self.features[fit_idx][smask]

        # Dummy mask for first iteration
        smask = np.arange(len(x1_sc))

        # Shortcut for data
        beta, ic = 0., 0.
        for _ in range(kappa + 1):

            # Mask data
            x1_sc, x2_sc = x1_sc[smask], x2_sc[smask]
            x1_sc_err, x2_sc_err = x1_sc_err[smask], x2_sc_err[smask]
            y1_sc, y2_sc = y1_sc[smask], y2_sc[smask]

            xdata_science, ydata_science = x1_sc - x2_sc, y1_sc - y2_sc

            # Determine (Co-)variance terms of errors for science field
            var_err_science = np.mean(x1_sc_err) ** 2 + np.mean(x2_sc_err) ** 2
            cov_err_science = -np.mean(x2_sc_err) ** 2

            # Determine beta for the given data
            # LINES
            upper = get_covar(xdata_science, ydata_science) - cov_control - cov_err_science + cov_err_control
            lower = np.var(xdata_science) - var_err_science - var_control + var_err_control
            beta = upper / lower

            # OLS
            # beta = get_covar(xdata_science, ydata_science) / np.var(xdata_science)

            # BCES
            # upper = get_covar(xdata_science, ydata_science) - cov_err_science
            # lower = np.var(xdata_science) - var_err_science
            # beta = upper / lower

            # Get intercept of linear fit through median
            ic = np.median(ydata_science) - beta * np.median(xdata_science)

            # ODR
            # from scipy.odr import Model, RealData, ODR
            # fit_model = Model(linear_model)
            # fit_data = RealData(xdata_science, ydata_science)#, sx=std_x, sy=std_y)
            # bdummy = ODR(fit_data, fit_model, beta0=[1., 0.]).run()
            # beta, ic = bdummy.beta
            # beta_err, ic_err = bdummy.sd_beta

            # Get orthogonal distance to this line
            dis = np.abs(beta * xdata_science - ydata_science + ic) / np.sqrt(beta ** 2 + 1)

            # 3 sig filter
            smask = dis - np.median(dis) < sigma * np.std(dis)
            # smask = dis - np.median(dis) < 0.15

        # Do the same for random splits to get errors
        beta_i = []
        for _ in range(err_iter):

            # Define random index array
            ridx_sc = np.random.permutation(len(x1_sc))

            # Split array in two equally sized parts and get data
            x1_sc_1, x2_sc_1 = x1_sc[ridx_sc[0::2]], x2_sc[ridx_sc[0::2]]
            x1_sc_2, x2_sc_2 = x1_sc[ridx_sc[1::2]], x2_sc[ridx_sc[1::2]]
            x1_sc_err_1, x2_sc_err_1 = x1_sc_err[ridx_sc[0::2]], x2_sc_err[ridx_sc[0::2]]
            x1_sc_err_2, x2_sc_err_2 = x1_sc_err[ridx_sc[1::2]], x2_sc_err[ridx_sc[1::2]]
            y1_sc_1, y2_sc_1 = y1_sc[ridx_sc[0::2]], y2_sc[ridx_sc[0::2]]
            y1_sc_2, y2_sc_2 = y1_sc[ridx_sc[1::2]], y2_sc[ridx_sc[1::2]]

            # Determine variance terms
            var_sc_1, var_sc_2 = np.nanvar(x1_sc_1 - x2_sc_1), np.nanvar(x1_sc_2 - x2_sc_2)
            var_sc_err_1 = np.nanmean(x1_sc_err_1) ** 2 + np.nanmean(x2_sc_err_1) ** 2
            var_sc_err_2 = np.nanmean(x1_sc_err_2) ** 2 + np.nanmean(x2_sc_err_2) ** 2

            # Determine covariance terms
            # TODO: This probably only works when no NaNs are present...
            cov_sc_1 = get_covar(xi=x1_sc_1 - x2_sc_1, yi=y1_sc_1 - y2_sc_1)
            cov_sc_2 = get_covar(xi=x1_sc_2 - x2_sc_2, yi=y1_sc_2 - y2_sc_2)
            cov_sc_err_1, cov_sc_err_2 = -np.nanmean(x1_sc_err_1) ** 2, -np.nanmean(x1_sc_err_2) ** 2

            upper1 = cov_sc_1 - cov_sc_err_1 - cov_control + cov_err_control
            lower1 = var_sc_1 - var_sc_err_1 - var_control + var_err_control
            beta1 = upper1 / lower1

            upper2 = cov_sc_2 - cov_sc_err_2 - cov_control + cov_err_control
            lower2 = var_sc_2 - var_sc_err_2 - var_control + var_err_control
            beta2 = upper2 / lower2

            # Append beta values
            beta_i.append(np.std([beta1, beta2]))

        # Get error estimate
        beta_err = 1.25 * np.sum(beta_i) / (np.sqrt(2) * 1000)

        # Return fit and data values
        return beta, beta_err, ic, x1_sc, x2_sc, y2_sc

    # ----------------------------------------------------------------------
    # noinspection PyPackageRequirements
    def get_beta_binning(self, base_keys, fit_key, extinction, step=0.1):
        # TODO: Improve or remove!

        # Try to import scipy, otherwise stop
        try:
            from scipy.odr import Model, RealData, ODR
        except ImportError:
            print("Scipy not installed")
            # TODO: If I don't put this here, I get a weird warning
            from scipy.odr import Model, RealData, ODR

        # The extinction data must have the same size as the photometry
        assert len(extinction) == self.n_data

        # First let's get the data into colors
        xdata, ydata = self.dict[base_keys[0]] - self.dict[base_keys[1]], self.dict[base_keys[1]] - self.dict[fit_key]

        # Get average colors in extinction bins
        avg_x, avg_y, std_x, std_y, avg_e, avg_n = [], [], [], [], [], []
        for e in np.arange(np.floor(np.nanmin(extinction)), np.ceil(np.nanmax(extinction)), step=step):

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                avg_fil = (extinction >= e) & (extinction < e + step)

                # Get color for current filter extinction
                xdummy, ydummy = xdata[avg_fil], ydata[avg_fil]

                # Do 3-sigma clipping around median
                # avg_clip = (np.abs(xdummy - np.nanmedian(xdummy)) < 3 * np.nanstd(xdummy)) & \
                #            (np.abs(ydummy - np.nanmedian(ydummy)) < 3 * np.nanstd(ydummy))
                avg_clip = np.full_like(xdummy, fill_value=True, dtype=bool)

                # We require at least 3 sources
                if np.sum(avg_clip) < 3:
                    continue

                # Append averages
                avg_x.append(np.nanmedian(xdummy[avg_clip]))
                avg_y.append(np.nanmedian(ydummy[avg_clip]))
                std_x.append(np.nanstd(xdummy[avg_clip]))
                std_y.append(np.nanstd(ydummy[avg_clip]))
                avg_e.append(e + step / 2)
                avg_n.append(np.sum(avg_clip))

        # Convert to arrays
        avg_x, avg_y, avg_e, avg_n = np.array(avg_x), np.array(avg_y), np.array(avg_e), np.array(avg_n)

        # Fit a line with ODR
        fit_model = Model(linear_model)
        fit_data = RealData(avg_x, avg_y, sx=std_x, sy=std_y)
        bdummy = ODR(fit_data, fit_model, beta0=[1., 0.]).run()
        beta, ic = bdummy.beta
        beta_err, ic_err = bdummy.sd_beta

        # Return slope, intercept and errors
        return beta, beta_err, ic, ic_err, avg_x, avg_y, avg_e, avg_n


# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #
class Colors(DataBase):

    def __init__(self, mag, err, extvec, lon=None, lat=None, names=None):
        """
        Basically the same as magnitudes without NICER. Naturally the PNICER implementation does not allow to convert
        to colors.

        Parameters
        ----------
        mag
        err
        extvec
        lon
        lat
        names

        Returns
        -------

        """
        # TODO: Add docstring

        # Call parent
        super(Colors, self).__init__(mag=mag, err=err, extvec=extvec, lon=lon, lat=lat, names=names)

        # Add attributes
        self.colors_names = self.features_names

    # ----------------------------------------------------------------------
    def pnicer(self, control, sampling=2, kernel="epanechnikov"):
        """
        PNICER call method for colors.

        Parameters
        ----------
        control
            Control field instance.
        sampling : int, optional
            Sampling of grid relative to bandwidth of kernel. Default is 2.
        kernel : str, optional
            Name of kernel for KDE. e.g. 'epanechnikov' or 'gaussian'. Default is 'epanechnikov'.

        """

        comb = zip(self.all_combinations(idxstart=1), control.all_combinations(idxstart=1))
        return self._pnicer_combinations(control=control, comb=comb, sampling=sampling, kernel=kernel)


# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #
class Extinction:

    def __init__(self, db, extinction, variance=None, color0=None):
        """
        Class for extinction measurements.

        Parameters
        ----------
        db : DataBase
            Base class from which the extinction was derived.
        extinction : np.ndarray
            Extinction data.
        variance : np.ndarray, optional
            Variance in extinction.
        color0 : np.ndarray, optional
            Intrisic color set for each source.

        """

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
        grid_header, grid_lon, grid_lat = self.db.build_wcs_grid(frame=frame, pixsize=pixsize)

        # Set some header keywords
        grid_header["BWIDTH"] = (bandwidth, "Bandwidth of kernel (degrees)")
        if use_fwhm:
            grid_header["FWHM"] = (bandwidth * 2 * np.sqrt(2 * np.log(2)), "FWHM of gaussian (degrees)")

        # Run extinction mapping for each pixel
        with Pool() as pool:
            # Submit tasks
            mp = pool.starmap(get_extinction_pixel,
                              zip(grid_lon.ravel(), grid_lat.ravel(),
                                  repeat(self.db.lon[self.clean_index]), repeat(self.db.lat[self.clean_index]),
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
        col1 = fits.Column(name="Lon", format='D', array=self.db.lon)
        col2 = fits.Column(name="Lat", format='D', array=self.db.lat)
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
        hdulist = fits.HDUList([fits.PrimaryHDU(),
                                fits.ImageHDU(data=self.map, header=self.fits_header),
                                fits.ImageHDU(data=self.var, header=self.fits_header),
                                fits.ImageHDU(data=self.num, header=self.fits_header),
                                fits.ImageHDU(data=self.rho, header=self.fits_header)])
        hdulist.writeto(path, clobber=True)


# ---------------------------------------------------------------------- #
#            Helper top level methods for parallel processing            #
# ---------------------------------------------------------------------- #

# ----------------------------------------------------------------------
def _mp_kde(kde, data, grid):
    """
    Parallelisation routine for kernel density estimation.

    Parameters
    ----------
    kde
        KernelDensity instance from scikit learn
    data
        Input data
    grid
        Grid on which to evaluate the density.

    Returns
    -------
    np.ndarray

    """

    return np.exp(kde.fit(data).score_samples(grid))


# ----------------------------------------------------------------------
def mp_kde(grid, data, bandwidth, shape=None, kernel="epanechnikov", norm=False, absolute=False, sampling=None):
    """
    Kernel density estimation with parallelisation.

    Parameters
    ----------
    grid
        Grid on which to evaluate the density.
    data
        Input data
    bandwidth : int, float
        Bandwidth of kernel (in data units).
    shape
        If set, shape out ouput.
    kernel : str, optional
        Name of kernel for KDE. e.g. 'epanechnikov' or 'gaussian'. Default is 'epanechnikov'.
    norm : bool, optional
        Whether to normalize the result (density estimate from 0 to 1). Default is False.
    absolute : bool, optional
        Whether to return absolute numbers.
    sampling : int, optional
        Sampling of grid. Necessary only when absolute numbers should be returned.

    Returns
    -------
    np.ndarray

    """

    # TODO: remove shape parameter

    # If we want absolute values, we must specify the sampling
    if absolute:
        assert sampling, "Sampling needs to be specified"

    # Dimensions of grid and data must match
    assert len(grid.shape) == len(data.shape), "Data and Grid dimensions must match"

    # If only one dimension, extend
    if len(grid.shape) == 1:
        grid = grid[:, np.newaxis]
        data = data[:, np.newaxis]

    # Split for parallel processing
    grid_split = np.array_split(grid, multiprocessing.cpu_count(), axis=0)

    # Define kernel according to Nyquist sampling
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)

    with Pool() as pool:
        mp = pool.starmap(_mp_kde, zip(repeat(kde), repeat(data), grid_split))

    # Create array
    mp = np.concatenate(mp)

    # If we want absolute numbers we have to evaluate the same thing for the grid
    if absolute:
        mp *= data.shape[0] / np.sum(mp) * sampling

    # Normalize if set
    if norm == "max":
        mp /= np.nanmax(mp)
    elif norm == "mean":
        mp /= np.nanmean(mp)
    elif norm == "sum":
        mp /= np.nansum(mp)

    # Unpack results and return
    if shape is None:
        return mp
    else:
        return mp.reshape(shape)


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
    dis = distance_on_unit_sphere(ra1=xdata, dec1=ydata, ra2=xgrid, dec2=ygrid, unit="degrees")

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
    weights_spatial_norm = weights_spatial / np.trapz(y=wfunc(dummy), x=dummy)

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


# ---------------------------------------------------------------------- #
#                       Helper methods for plotting                      #
# ---------------------------------------------------------------------- #

# ----------------------------------------------------------------------
def axes_combinations(ndim, ax_size=None):
    """
    Creates a grid of axes to plot all combinations of data.

    Parameters
    ----------
    ndim : int
        Number of dimensions.
    ax_size : list, optional
        Single axis size. Default is [3, 3].

    Returns
    -------
    list
        List of axes which can be used for plotting.

    """

    if ax_size is None:
        ax_size = [3, 3]

    # Get all combinations to plot
    c = combinations(range(ndim), 2)

    # Create basic plot layout
    fig, axes = plt.subplots(ncols=ndim - 1, nrows=ndim - 1, figsize=[(ndim - 1) * ax_size[0], (ndim - 1) * ax_size[1]])

    if ndim == 2:
        axes = np.array([[axes], ])

    # Adjust plots
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0, hspace=0)

    axes_out = []
    for idx in c:

        # Get index of subplot
        x_idx, y_idx = ndim - idx[0] - 2, ndim - 1 - idx[1]

        # Get axes
        axes_out.append(axes[x_idx, y_idx])

        # Hide tick labels
        if x_idx < ndim - 2:
            axes_out[-1].axes.xaxis.set_ticklabels([])
        if y_idx > 0:
            axes_out[-1].axes.yaxis.set_ticklabels([])

        # Delete the other axes
        if ((idx[0] > 0) | (idx[1] - 1 > 0)) & (idx[0] != idx[1] - 1):
            fig.delaxes(axes[idx[0], idx[1] - 1])

    return fig, axes_out


# ---------------------------------------------------------------------- #
#                         Various helper methods                         #
# ---------------------------------------------------------------------- #

# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
def get_covar(xi, yi):
    """
    Calculate sample covariance (can not contain NaNs!)

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

    return np.sum((xi - np.mean(xi)) * (yi - np.mean(yi))) / len(xi)


# ----------------------------------------------------------------------
def linear_model(vec, val):
    """
    Linear function model y = m*x + b.

    Parameters
    ----------
    vec : iterable
        Vector of the parameters.
    val : np.ndarray
        array of the current x values

    Returns
    -------

    """

    return vec[0] * val + vec[1]


# ----------------------------------------------------------------------
def distance_on_unit_sphere(ra1, dec1, ra2, dec2, unit="radians"):
    """
    Calculates the distance between two objects on a sphere along a great circle. Also works with arrays.

    Parameters
    ----------
    ra1 : int, float, np.ndarray
        Longitude of first object(s).
    dec1 : int, float, np.ndarray
        Latitude of first object(s).
    ra2 : int, float, np.ndarray
        Longitude of second object(s).
    dec2 : int, float, np.ndarray
        Latitude of second object(s).
    unit : str, optional
        Units of calculation. 'Degree' or 'Radians'

    Returns
    -------
        Distances.

    """

    # TODO: Update with more precise formula!

    if unit not in ["radians", "degrees"]:
        raise ValueError("'unit' must be either 'radians', or 'degrees'!")

    # Calculate distance on sphere
    if unit == "radians":
        dis = np.arccos(np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2))
    else:
        dis = np.degrees(np.arccos(np.sin(np.radians(dec1)) * np.sin(np.radians(dec2)) +
                                   np.cos(np.radians(dec1)) * np.cos(np.radians(dec2)) * np.cos(np.radians(ra1 - ra2))))

    # Return distance
    return dis
