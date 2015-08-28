from __future__ import absolute_import, division, print_function


# ----------------------------------------------------------------------
# Import stuff
import warnings
import wcsaxes
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt

from astropy import wcs
from astropy.io import fits
from multiprocessing import Pool
from matplotlib.pyplot import GridSpec
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from itertools import combinations, repeat
from sklearn.neighbors import KernelDensity, NearestNeighbors


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Define general data class
class DataBase:
    def __init__(self, mag, err, extvec, lon=None, lat=None, names=None):
        """
        Basic Data class which provides the foundation for extinction measurements
        :param mag: List of magnitude arrays. All arrays must have the same length!
        :param err: List off magnitude error arrays.
        :param extvec: List holding the extinction components for each magnitude
        :param lon: Longitude of coordinates for each source
        :param lat: Latitude of coordinates for each source
        :param names: List of magnitude (feature) names
        """

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

        # ----------------------------------------------------------------------
        # Do some input checks

        # Dimensions of extinction vector must be equal to dimensions of data
        if self.extvec.n_dimensions != self.n_features:
            raise ValueError("Dimensions of extinction vector must be equal to number of features")

        # Input data must be in lists
        if sum([type(x) in [list] for x in [self.features, self.features_err]]) != 2:
            raise TypeError("Input must be in lists")

        # There must be at least two features
        if self.n_features < 2:
            raise ValueError("There must be at least two features!")

        # All input lists must have equal length
        if len(set([len(l) for l in [self.features, self.features_err, self.features_names]])) != 1:
            raise ValueError("Input lists must have equal length")

        # Input data must also have the same size
        if len(set([x.size for x in self.features])) != 1:
            raise ValueError("Input arrays must have equal size")
        else:
            self.n_data = self.features[0].size

        # Coordinates must be supplied for all data if set
        if (lon is not None) | (lat is not None):
            if (len(lon) != len(lat)) | (len(lon) != len(self.features[0])):
                raise ValueError("Input coordinates do not match!")

        # ----------------------------------------------------------------------
        # Calculate some stuff

        # Generate feature masks and number of good data points per feature
        self.features_masks = [np.isfinite(m) & np.isfinite(e) for m, e in zip(self.features, self.features_err)]
        self.combined_mask = np.prod(np.vstack(self.features_masks), axis=0, dtype=bool)

        # ----------------------------------------------------------------------
        # Plot range
        # noinspection PyTypeChecker
        self.plotrange = [(np.floor(np.percentile(x[m], 0.01)), np.ceil(np.percentile(x[m], 99.99)))
                          for x, m in zip(self.features, self.features_masks)]

    # ----------------------------------------------------------------------
    # Static helper methods in namespace

    @staticmethod
    def round_partial(data, precision):
        """
        Simple static method to round data to arbitrary precision
        :param data: Data to be rounded
        :param precision: desired precision. e.g. 0.2
        :return: Rounded data
        """
        return np.around(data / precision) * precision

    @staticmethod
    def build_grid(data, precision):
        """
        Static method to build a grid of unique positons from given input data rounded to arbitrary precision
        :param data: Data from which to build grid (at least 2D)
        :param precision: Desired position, i.e. pixel scale
        :return: grid built from input data
        """

        grid_data = DataBase.round_partial(data=data, precision=precision).T

        # Get unique positions for coordinates
        dummy = np.ascontiguousarray(grid_data).view(np.dtype((np.void, grid_data.dtype.itemsize * grid_data.shape[1])))
        _, idx = np.unique(dummy, return_index=True)

        return grid_data[idx].T

    # ----------------------------------------------------------------------
    def rotate(self):
        """
        Method to rotate data space with the given extinction vector.
        Only finite data are transmitted intp the new data space.
        :return: Instance of input with rotated data
        """

        data = np.vstack(self.features).T[self.combined_mask].T
        err = np.vstack(self.features_err).T[self.combined_mask].T

        # Rotate data
        rotdata = self.extvec.rotmatrix.dot(data)

        # Rotate extinction vector
        extvec = self.extvec.extinction_rot

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
    def all_combinations(self):
        """
        Method to get all combinations of input features
        :return: All combinations from input features
        """

        all_c = [item for sublist in [combinations(range(self.n_features), p) for p in range(2, self.n_features + 1)]
                 for item in sublist]

        combination_instances = []
        for c in all_c:
            cdata, cerror = [self.features[idx] for idx in c], [self.features_err[idx] for idx in c]
            cnames = [self.features_names[idx] for idx in c]
            extvec = [self.extvec.extvec[idx] for idx in c]
            combination_instances.append(self.__class__(mag=cdata, err=cerror, extvec=extvec,
                                                        lon=self.lon, lat=self.lat, names=cnames))

        return combination_instances

    # ----------------------------------------------------------------------
    def _pnicer_single(self, control, sampling, kernel):
        """
        Main PNICER routine to get extinction. This will return only the extinction values for data for which all
        features are available
        :param control: instance of control field data
        :param sampling: Sampling of grid relative to bandwidth of kernel
        :param kernel: name of kernel to be used for density estimation. e.g. "epanechnikov" or "gaussian"
        :return: Extinction and variance for input data
        """

        # Check instances
        if self.__class__ != control.__class__:
            raise TypeError("Input and control instance not compatible")

        # Let's rotate the data spaces
        science_rot, control_rot = self.rotate(), control.rotate()

        # Get bandwidth of kernel
        bandwidth = np.around(np.mean(np.nanmean(self.features_err, axis=1)), 2)

        # Determine bin widths for grid according to bandwidth and sampling
        bin_grid = bin_ext = np.float(bandwidth / sampling)

        # Now we build a grid from the rotated data for all components but the first
        grid_data = DataBase.build_grid(data=np.vstack(science_rot.features)[1:, :], precision=bin_grid)

        # Create a grid to evaluate along the reddening vector
        grid_ext = np.arange(start=np.floor(min(science_rot.features[0])),
                             stop=np.ceil(max(science_rot.features[0])), step=bin_ext)

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
            # TODO: Check how many sources are non-0 :)
            if np.sum(vec) < 3:
                grid_mean.append(np.nan)
                grid_var.append(np.nan)
            else:
                # Get weighted average position along vector and the weighted variance
                a, b = weighted_avg(values=grid_ext, weights=vec)
                grid_mean.append(a)
                grid_var.append(b / self.extvec.extinction_norm)  # The normalisation converts this to extinction

        # Convert to arrays
        grid_var = np.array(grid_var)
        grid_mean = np.array(grid_mean)

        # Let's get the nearest neighbor grid point for each source
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(grid_data.T)
        _, indices = nbrs.kneighbors(np.vstack(science_rot.features)[1:, :].T)
        indices = indices[:, 0]

        # Now we have the instrisic colors for each vector and indices for all sources.
        # It's time to calculate the extinction. :)
        ext = (science_rot.features[0] - grid_mean[indices]) / self.extvec.extinction_norm
        var = grid_var[indices]

        # Lastly we put all the extinction measurements back into a full array
        out = np.full(self.n_data, fill_value=np.nan, dtype=float)
        outvar = np.full(self.n_data, fill_value=np.nan, dtype=float)

        # Output data for all sources
        out[self.combined_mask], outvar[self.combined_mask] = ext, var

        return out, outvar

    # ----------------------------------------------------------------------
    def _pnicer_combinations(self, control, comb, sampling, kernel):
        """
        PNICER base implementation for combinations. Basically calls the pnicer_single implementation for all
        combinations. The outpur extinction is then the one with the smallest error from all combinations
        :param control: instance of control field data
        :param comb: zip object of combinations to use
        :param sampling: Sampling of grid relative to bandwidth of kernel
        :param kernel: name of kernel to be used for density estimation. e.g. "epanechnikov" or "gaussian"
        :return: Extinction instance with the calcualted extinction and error
        """

        # Check instances
        if control.__class__ != self.__class__:
            raise TypeError("input and control instance do not match")

        # We loop over all combinations
        all_ext, all_var, names = [], [], []

        # Here we loop over color combinations since this is faster
        i = 0
        for sc, cc in comb:

            assert isinstance(sc, DataBase)
            # Run PNICER for current combination
            ext, var = sc._pnicer_single(control=cc, sampling=sampling, kernel=kernel)

            # Append data
            all_ext.append(ext)
            all_var.append(var)
            names.append("(" + ",".join(sc.features_names) + ")")
            i += 1

        # Convert to arrays and save combination data
        all_ext = np.array(all_ext)
        self._ext_combinations = all_ext.copy()
        all_var = np.array(all_var)
        self._var_combinations = all_var.copy()
        self._combination_names = names
        self._n_combinations = i

        # Chose extinction as minimum error across all combinations
        all_var[~np.isfinite(all_var)] = 100 * np.nanmax(all_var)
        ext = all_ext[np.argmin(all_var, axis=0), np.arange(self.n_data)]
        var = all_var[np.argmin(all_var, axis=0), np.arange(self.n_data)]
        # Make error cut
        ext[var > 10] = var[var > 10] = np.nan

        # Return
        return Extinction(db=self, extinction=ext, variance=var)

    # ----------------------------------------------------------------------
    def build_wcs_grid(self, frame, pixsize=10./60):
        """
        Method to build a WCS grid with a valid projection given a pixel scale
        :param frame: "equatorial" or "galactic"
        :param pixsize: pixel size of grid
        :return: header, longrid, latgrid
        """

        if frame == "equatorial":
            ctype = ["RA---COE", "DEC--COE"]
        elif frame == "galactic":
            ctype = ["GLON-COE", "GLAT-COE"]
        else:
            raise KeyError("frame must be'galactic' or 'equatorial'")

        # Calculate range of grid to 0.2 degree precision
        lon_range = [np.floor(np.min(self.lon) * 5) / 5, np.ceil(np.max(self.lon) * 5) / 5]
        lat_range = [np.floor(np.min(self.lat) * 5) / 5, np.ceil(np.max(self.lat) * 5) / 5]

        naxis1 = np.ceil((lon_range[1] - lon_range[0]) / pixsize).astype(np.int)
        naxis2 = np.ceil((lat_range[1] - lat_range[0]) / pixsize).astype(np.int)

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

    # ----------------------------------------------------------------------
    # Plotting methods

    def plot_combinations_scatter(self, path=None, ax_size=None, **kwargs):
        """
        2D Scatter plot of combinations
        :param path: file path if it should be saved. e.g. "/path/to/image.png"
        :param ax_size: Size of individual axis
        :param kwargs: Additional scatter plot arguments
        :return:
        """

        if ax_size is None:
            ax_size = [3, 3]

        # Get axes for figure
        fig, axes = axes_combinations(ndim=self.n_features, ax_size=ax_size)

        # Get 2D combination indices
        for idx, ax in zip(combinations(range(self.n_features), 2), axes):

            ax.scatter(self.features[idx[1]][::10], self.features[idx[0]][::10], lw=0, s=5, alpha=0.1, **kwargs)

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

    def plot_combinations_kde(self, path=None, ax_size=None, grid_bw=0.1, kernel="epanechnikov"):
        """
        KDE for all 2D combinations of features
        :param path: file path if it should be saved. e.g. "/path/to/image.png"
        :param ax_size: Size of individual axis
        :param grid_bw: grid bin width
        :param kernel: name of kernel for KDE. e.g. "epanechnikov" or "gaussian"
        :return:
        """

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
            dens = mp_kde(grid=xgrid, data=data, bandwidth=grid_bw*2, shape=x.shape, kernel=kernel)

            # Show result
            ax.imshow(np.sqrt(dens.T), origin="lower", interpolation="nearest", extent=[l, h, l, h], cmap="gist_heat_r")

            # Axis labels
            if ax.get_position().x0 < 0.11:
                ax.set_ylabel(self.features_names[idx[0]])
            if ax.get_position().y0 < 0.11:
                ax.set_xlabel(self.features_names[idx[1]])

        # Save or show figure
        if path is None:
            plt.show()
        else:
            plt.savefig(path, bbox_inches="tight")
        plt.close()

    def plot_spatial_kde(self, frame, pixsize=10/60, path=None, kernel="epanechnikov", skip=1):
        """
        Plot source densities for features
        :param frame: "equatorial" or "galactic"
        :param pixsize: pixel size of grid
        :param path: file path if it should be saved. e.g. "/path/to/image.png"
        :param kernel: name of kernel for KDE. e.g. "epanechnikov" or "gaussian"
        :param skip: Integer to skip every n-th source (for faster plotting)
        :return:
        """
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
            dens = mp_kde(grid=xgrid, data=data, bandwidth=pixsize*2, shape=lon_grid.shape, kernel=kernel)

            # Norm and save scale
            if idx == 0:
                scale = np.max(dens)

            dens /= scale

            # Plot density
            ax.imshow(dens, origin="lower", interpolation="nearest", cmap="gist_heat_r", vmin=0, vmax=1)

        # Save or show figure
        if path is None:
            plt.show()
        else:
            plt.savefig(path, bbox_inches="tight")
        plt.close()

    def plot_spatial_kde_gain(self, frame, pixsize=10/60, path=None, kernel="epanechnikov", skip=1):
        """
        Plot source densities for features
        :param frame: "equatorial" or "galactic"
        :param pixsize: pixel size of grid
        :param path: file path if it should be saved. e.g. "/path/to/image.png"
        :param kernel: name of kernel for KDE. e.g. "epanechnikov" or "gaussian"
        :param skip: Integer to skip every n-th source (for faster plotting)
        :return:
        """
        # Get a WCS grid
        header, lon_grid, lat_grid = self.build_wcs_grid(frame=frame, pixsize=pixsize)

        # Get aspect ratio
        ar = lon_grid.shape[0] / lon_grid.shape[1]

        # Determine number of panels
        n_panels = [np.floor(np.sqrt(self.n_features - 1)).astype(int),
                    np.ceil(np.sqrt(self.n_features - 1)).astype(int)]
        if n_panels[0] * n_panels[1] < self.n_features - 1:
            n_panels[n_panels.index(min(n_panels))] += 1

        # Create grid
        plt.figure(figsize=[10 * n_panels[0], 10 * n_panels[1] * ar])
        grid = GridSpec(ncols=n_panels[0], nrows=n_panels[1], bottom=0.05, top=0.95, left=0.05, right=0.95,
                        hspace=0.2, wspace=0.2)

        # To avoid editor warnings
        dens, dens_norm = 0, 0

        # Loop over features and plot
        for idx in range(self.n_features):

            # Save previous density
            if idx > 0:
                dens_norm = dens.copy()

            # Get density
            xgrid = np.vstack([lon_grid.ravel(), lat_grid.ravel()]).T
            data = np.vstack([self.lon[self.features_masks[idx]][::skip], self.lat[self.features_masks[idx]][::skip]]).T
            dens = mp_kde(grid=xgrid, data=data, bandwidth=pixsize*2, shape=lon_grid.shape, kernel=kernel)

            # Norm and save scale
            if idx > 0:
                # Add axes
                ax = plt.subplot(grid[idx - 1], projection=wcsaxes.WCS(header=header))

                # Plot density
                with warnings.catch_warnings():
                    # Ignore NaN and 0 division warnings
                    warnings.simplefilter("ignore")
                    ax.imshow(dens / dens_norm, origin="lower", interpolation="nearest",
                              cmap="coolwarm_r", vmin=0, vmax=2)

                # Grab axes
                lon = ax.coords[0]
                lat = ax.coords[1]

                # Set labels
                if idx % n_panels[1] == 1:
                    lat.set_axislabel("Latitude")
                if idx / n_panels[1] > 1:
                    lon.set_axislabel("Longitude")

        # Save or show figure
        if path is None:
            plt.show()
        else:
            plt.savefig(path, bbox_inches="tight")
        plt.close()

    def plot_kde_extinction_combinations(self, path=None, bandwidth=None, sampling=16):
        """
        Plot histogram of extinctions for all combinations. Requires PNICER to be run beforehand
        :param path: file path if it should be saved. e.g. "/path/to/image.png"
        :param bandwidth: Bandwidth of KDE. If not set, use Scott's rule divided by 2
        :param sampling: Sampling factor of grid (the larger, the more samples)
        """

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
        ax1_range = [DataBase.round_partial(np.nanmean(self._ext_combinations) -
                                            3 * np.nanstd(self._ext_combinations), 0.1),
                     DataBase.round_partial(np.nanmean(self._ext_combinations)
                                            + 3.5 * np.nanstd(self._ext_combinations), 0.1)]
        # noinspection PyTypeChecker
        ax2_range = [0., DataBase.round_partial(np.nanmean(self._var_combinations)
                                                + 3.5 * np.nanstd(self._var_combinations), 0.1)]

        plt.figure(figsize=[5 * n_panels[1], 5 * n_panels[0] * 0.5])
        grid = GridSpec(ncols=n_panels[1], nrows=n_panels[0], bottom=0.05, top=0.95, left=0.05, right=0.95,
                        hspace=0, wspace=0)

        for idx in range(self._n_combinations):

            # Get densities for extinction
            ext = self._ext_combinations[idx, :]
            # noinspection PyTypeChecker
            ext = ext[np.isfinite(ext)]
            if bandwidth is None:
                bandwidth = np.float(3.5 * np.std(ext) / np.power(np.sqrt(len(ext)), 1/3)) / 2
            grid_ext = np.arange(np.floor(ax1_range[0]), np.ceil(ax1_range[1]), bandwidth / sampling)
            dens_ext = mp_kde(grid=grid_ext, data=ext, bandwidth=bandwidth, shape=None,
                              kernel="epanechnikov", absolute=True, sampling=sampling)

            # Get densities for extinction error
            exterr = self._var_combinations[idx, :]
            # noinspection PyTypeChecker
            exterr = exterr[np.isfinite(exterr)]
            if bandwidth is None:
                bandwidth = np.float(3.5 * np.std(exterr) / np.power(np.sqrt(len(exterr)), 1/3)) / 2
            grid_exterr = np.arange(np.floor(ax2_range[0]), np.ceil(ax2_range[1]), bandwidth / sampling)
            dens_exterr = mp_kde(grid=grid_exterr, data=exterr[np.isfinite(exterr)], bandwidth=bandwidth, shape=None,
                                 kernel="epanechnikov", absolute=True, sampling=sampling)

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
# Magnitudes class
class Magnitudes(DataBase):

    def __init__(self, mag, err, extvec, lon=None, lat=None, names=None):
        """
        Main class for users. Includes PNICER and NICER
        :param mag: List of magnitude arrays. All arrays must have the same length!
        :param err: List off magnitude error arrays.
        :param extvec: List holding the extinction components for each magnitude
        :param lon: Longitude of coordinates for each source
        :param lat: Latitude of coordinates for each source
        :param names: List of magnitude (feature) names
        """

        # Call parent
        super(Magnitudes, self).__init__(mag=mag, err=err, extvec=extvec, lon=lon, lat=lat, names=names)

    # ----------------------------------------------------------------------
    def mag2color(self):
        """
        Method to convert to color instances
        :return: Colors instance of input data
        """

        # Calculate colors
        colors = [self.features[k - 1] - self.features[k] for k in range(1, self.n_features)]

        # Calculate color errors
        colors_error = [np.sqrt(self.features_err[k - 1] ** 2 + self.features_err[k] ** 2)
                        for k in range(1, self.n_features)]

        # Color names
        colors_names = [self.features_names[k - 1] + "-" + self.features_names[k] for k in range(1, self.n_features)]
        color_extvec = [self.extvec.extvec[k - 1] - self.extvec.extvec[k] for k in range(1, self.n_features)]

        return Colors(mag=colors, err=colors_error, extvec=color_extvec,
                      lon=self.lon, lat=self.lat, names=colors_names)

    # ----------------------------------------------------------------------
    # Method to get all color combinations
    def color_combinations(self):
        """
        Calculates a list of Colors instances for all combinations
        :return: List of Colors instances
        """

        # First get all colors...
        colors = self.mag2color()

        # ...and all combinations of colors
        colors_combinations = colors.all_combinations()

        """
        At a later stage I could include to also allow just two bands to produce and extinction with PNICER when used
        with colors. For now, color_combinations only produces combinations with at least two colors, i.e. at least
        three bands. In other words: PNICER for now requires always at least two input features: Either two magnitudes
        or two colors. In the casae of magnitudes, I spread the paramter space thin, in case of colors I need at least
        three bands.
        """
        # # Add color-magnitude combinations
        # for n in reversed(range(colors.n_features)):
        #     colors_combinations[:0] = \
        #         [Colors(mag=[self.features[n+1], colors.features[n]],
        #                 err=[self.features_err[n+1], colors.features_err[n]],
        #                 extvec=[self.extvec.extvec[n+1], colors.extvec.extvec[n]],
        #                 lon=self.lon, lat=self.lat, names=[self.features_names[n+1], colors.features_names[n]])]

        # for n in reversed(range(colors.n_features)):
        #     colors_combinations[:0] = \
        #         [Colors(mag=[self.features[n], self.features[n+1]],
        #                 err=[self.features_err[n], self.features_err[n+1]],
        #                 extvec=[self.extvec.extvec[n], self.extvec.extvec[n+1]],
        #                 lon=self.lon, lat=self.lat, names=[self.features_names[n], self.features_names[n+1]])]

        return colors_combinations

    # ----------------------------------------------------------------------
    def pnicer(self, control, sampling=2, kernel="epanechnikov", use_color=False):
        """
        PNICER call method for magnitudes. Includes options to use combinations for input features, or convert them
        to colors.
        :param control: instance of control field
        :param sampling: Sampling of grid relative to bandwidth of kernel
        :param kernel: name of kernel to be used for density estimation. e.g. "epanechnikov" or "gaussian"
        :param use_color: Whether or not to convert to colors
        :return: Extinction instance with the calculated extinction and error
        """
        if use_color:
            comb = zip(self.color_combinations(), control.color_combinations())
        else:
            comb = zip(self.all_combinations(), control.all_combinations())

        return self._pnicer_combinations(control=control, comb=comb, sampling=sampling, kernel=kernel)

    # ----------------------------------------------------------------------
    # NICER implementation
    def nicer(self, control, all_features=False):
        """
        NICER routine as descibed in Lombardi & Alves 2001. Generalized for arbitrary input magnitudes
        :param control: control field instance to calculate intrinsic colors
        :param all_features: If set, return only extinction values for sources with data for all features
        :return: Extinction instance
        """

        if isinstance(control, DataBase) is False:
            raise TypeError("control is not Data class instance")

        if self.n_features != control.n_features:
            raise ValueError("Number of features in the control field must match input")

        # Get reddening vector
        k = [x - y for x, y in zip(self.extvec.extvec[:-1], self.extvec.extvec[1:])]

        # Calculate covariance matrix of control field
        cov_cf = np.ma.cov([np.ma.masked_invalid(control.features[l]) - np.ma.masked_invalid(control.features[l+1])
                            for l in range(self.n_features - 1)])

        # Get intrisic color of control field
        color_0 = [np.nanmean(control.features[l] - control.features[l+1]) for l in range(control.n_features - 1)]

        # Calculate covariance matrix of errors in the science field
        cov_er = np.zeros([self.n_data, self.n_features - 1, self.n_features - 1])
        for i in range(self.n_features - 1):
            # Diagonal
            cov_er[:, i, i] = self.features_err[i]**2 + self.features_err[i+1] ** 2

            # Other entries
            if i > 0:
                cov_er[:, i, i-1] = cov_er[:, i-1, i] = -self.features_err[i] ** 2

        # Set NaNs to large covariance!
        cov_er[~np.isfinite(cov_er)] = 1E10

        # Total covariance matrix
        cov = cov_cf + cov_er

        # Invert
        cov_inv = np.linalg.inv(cov)

        # Get b from the paper (equ. 12)
        upper = np.dot(cov_inv, k)
        b = upper.T / np.dot(k, upper.T)

        # Get colors and set finite value for NaNs (will be downweighted by b!)
        scolors = np.array([self.features[l] - self.features[l+1] for l in range(self.n_features - 1)])

        # Get those with no good color value at all
        bad_color = np.all(np.isnan(scolors), axis=0)

        # Write finite value for all NaNs (this makes summing later easier)
        scolors[~np.isfinite(scolors)] = 0

        # Put back NaNs for those with only bad colors
        scolors[:, bad_color] = np.nan

        # Equation 13 in the NICER paper
        ext = b[0, :] * (scolors[0, :] - color_0[0])
        for i in range(1, self.n_features - 1):
            ext += b[i, :] * (scolors[i] - color_0[i])

        # Calculate variance (has to be done in loop due to RAM issues!)
        first = np.array([np.dot(cov.data[idx, :, :], b.data[:, idx]) for idx in range(self.n_data)])
        var = np.array([np.dot(b.data[:, idx], first[idx, :]) for idx in range(self.n_data)])
        # Now we have to mask the large variance data again
        var[~np.isfinite(ext)] = np.nan

        # Combined mask for all features
        if all_features:
            ext[~self.combined_mask] = var[~self.combined_mask] = np.nan

        # ...and return :)
        return Extinction(db=self, extinction=ext.data, variance=var)


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Colors class
class Colors(DataBase):

    def __init__(self, mag, err, extvec, lon=None, lat=None, names=None):
        """
        Basically the same as magnitudes. PNICER implementation does not allow to convert to colors.
        :param mag:
        :param err:
        :param extvec:
        :param lon:
        :param lat:
        :param names:
        :return:
        """
        super(Colors, self).__init__(mag=mag, err=err, extvec=extvec, lon=lon, lat=lat, names=names)

    # ----------------------------------------------------------------------
    def pnicer(self, control, sampling=2, kernel="epanechnikov"):
        """
        PNICER call method for colors.
        :param control: instance of control field
        :param sampling: Sampling of grid relative to bandwidth of kernel
        :param kernel: name of kernel to be used for density estimation. e.g. "epanechnikov" or "gaussian"
        :return: Extinction instance with the calculated extinction and error
        """

        comb = zip(self.all_combinations(), control.all_combinations())
        return self._pnicer_combinations(control=control, comb=comb, sampling=sampling, kernel=kernel)


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
class ExtinctionVector:

    def __init__(self, extvec):
        """
        Class for extinction vector components
        :param extvec: List of extinction values for each input feature
        :return:
        """

        self.extvec = extvec
        self.n_dimensions = len(extvec)

    # ----------------------------------------------------------------------
    # Some static helper methods within namespace

    @staticmethod
    def unit_vectors(n_dimensions):
        """
        Calculate unit vectors for a given number of dimensions
        :param n_dimensions: Number of dimensions
        :return: Unit vectors in a list
        """

        return [np.array([1.0 if i == l else 0.0 for i in range(n_dimensions)]) for l in range(n_dimensions)]

    @staticmethod
    def get_rotmatrix(vector):
        """
        Method to determine the rotation matrix so that the rotated first vector component is the only non-zero
        component. Critical for PNICER
        :param vector: Input extinction vector
        :return: rotation matrix
        """

        # Number of dimensions
        n_dimensions = len(vector)
        if n_dimensions < 2:
            ValueError("Vector must have at least two dimensions")

        # Get unit vectors
        uv = ExtinctionVector.unit_vectors(n_dimensions=n_dimensions)

        # To not raise editor warning
        vector_rot = [0]

        # Now we loop over all but the first component
        rotmatrices = []
        for n in range(n_dimensions - 1):
            # Calculate rotation angle of current component
            if n == 0:
                rot_angle = np.arctan(vector[n+1] / vector[0])
            else:
                rot_angle = np.arctan(vector_rot[n+1] / vector_rot[0])
            # Following the german Wikipedia... :)
            v = np.outer(uv[0], uv[0]) + np.outer(uv[n+1], uv[n+1])
            w = np.outer(uv[0], uv[n+1]) - np.outer(uv[n+1], uv[0])
            rotmatrices.append((np.cos(rot_angle)-1) * v + np.sin(rot_angle) * w + np.identity(n_dimensions))
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
    _rotmatrix = None

    @property
    def rotmatrix(self):
        """
        Simple property to hold the rotation matrix for all extinction components of this instance
        :return: rotation matrix
        """

        # Check if already determined
        if self._rotmatrix is not None:
            return self._rotmatrix

        self._rotmatrix = ExtinctionVector.get_rotmatrix(self.extvec)
        return self._rotmatrix

    # ----------------------------------------------------------------------
    _extinction_rot = None

    @property
    def extinction_rot(self):
        """
        :return: Rotated input extinction vector
        """

        # Check if already determined
        if self._extinction_rot is not None:
            return self._extinction_rot

        self._extinction_rot = self.rotmatrix.dot(self.extvec)
        return self._extinction_rot

    # ----------------------------------------------------------------------
    _extinction_norm = None

    @property
    def extinction_norm(self):
        """
        :return: Normalization component for projected extinction vector
        """
        # Check if already determined
        if self._extinction_norm is not None:
            return self._extinction_norm

        self._extinction_norm = self.extinction_rot[0]
        return self._extinction_norm


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
class Extinction:

    def __init__(self, db, extinction, variance=None):
        """
        Class for extinction measurements
        :param db: Base class from which the extinction was derived
        :param extinction: extinction measurements
        :param variance: extinction variance
        """

        # Check if db is really a DataBase instance
        assert isinstance(db, DataBase), "passed instance is not DataBase class"

        self.db = db
        self.extinction = extinction
        self.variance = variance
        if self.variance is None:
            self.variance = np.zeros_like(extinction)

        # extinction and variance must have same length
        if len(self.extinction) != len(self.variance):
            raise ValueError("Extinction and variance arrays must have equal length")

        # ----------------------------------------------------------------------
        # Calculate some simple things

    # ----------------------------------------------------------------------
    def build_map(self, bandwidth, method="median", sampling=2, nicest=False):
        """
        Method to build an extinction map
        :param bandwidth: Resolution of map
        :param method: Method to be used. e.g. "median", "gaussian", "epanechnikov", "uniform", "triangular"
        :param sampling: Sampling of data. i.e. how many pixels per bandwidth
        :param nicest: whether or not to adjust weights with NICEST correction factor
        :return: ExtinctionMap instance
        """

        # Sampling must be an integer
        assert isinstance(sampling, int), "sampling must be an integer"

        # First let's get a grid
        grid_header, grid_lon, grid_lat = self.db.build_wcs_grid(frame="galactic", pixsize=bandwidth / sampling)

        # Run extinction mapping for each pixel
        with Pool() as pool:
            # Submit tasks
            mp = pool.starmap(get_extinction_pixel,
                              zip(grid_lon.ravel(), grid_lat.ravel(), repeat(self.db.lon),
                                  repeat(self.db.lat), repeat(self.extinction), repeat(self.variance),
                                  repeat(bandwidth), repeat(method), repeat(nicest)))

        # Unpack results
        map_ext, map_var, map_num = list(zip(*mp))

        # reshape
        map_ext = np.array(map_ext).reshape(grid_lon.shape)
        map_var = np.array(map_var).reshape(grid_lon.shape)
        map_num = np.array(map_num).reshape(grid_lon.shape)

        # Return extinction map instance
        return ExtinctionMap(ext=map_ext, var=map_var, num=map_num, header=grid_header, metric=method)

    # ----------------------------------------------------------------------
    def save_fits(self, path):
        """
        Write the extinction data to a FITS table file
        :param path: file path; e.g. "/path/to/table.fits"
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


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
class ExtinctionMap:

    def __init__(self, ext, var, num, header, metric):
        """
        Extinction map class
        :param ext: 2D Extintion map
        :param var: 2D Extinction variance map
        :param num: 2D number map
        :param header: header of grid from which extinction map was built.
        :param metric: Metric used to create the map
        """

        self.map = ext
        self.var = var
        self.num = num
        self.metric = metric
        self.shape = self.map.shape
        self.fits_header = header

        # Input must be 2D
        if (len(self.map.shape) != 2) | (len(self.var.shape) != 2) | (len(self.num.shape) != 2):
            raise TypeError("Input must be 2D arrays")

    def plot_map(self, path=None, figsize=5):
        """
        Simple method to plot extinction map
        :param path: file path if it should be saved. e.g. "/path/to/image.png"
        :param figsize: figure size adjustment parameter
        :return:
        """

        fig = plt.figure(figsize=[figsize, 3 * 0.9 * figsize * (self.shape[0] / self.shape[1])])
        grid = GridSpec(ncols=2, nrows=3, bottom=0.05, top=0.95, left=0.05, right=0.95, hspace=0.08, wspace=0,
                        height_ratios=[1, 1, 1], width_ratios=[1, 0.05])

        for idx in range(0, 6, 2):

            # print(idx)
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
                               vmax=np.ceil(np.percentile(self.var[np.isfinite(self.var)], 99) * 10) / 10)
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

    def save_fits(self, path):
        """
        Method to save extinciton map as FITS file
        :param path: file path if it should be saved. e.g. "/path/to/image.fits"
        """

        # TODO: Add some header information
        # Create and save
        hdulist = fits.HDUList([fits.PrimaryHDU(),
                                fits.ImageHDU(data=self.map, header=self.fits_header),
                                fits.ImageHDU(data=self.var, header=self.fits_header),
                                fits.ImageHDU(data=self.num, header=self.fits_header)])
        hdulist.writeto(path, clobber=True)


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Helper top level methods for parallel processing

# ----------------------------------------------------------------------
# KDE functions for parallelisation
def _mp_kde(kde, data, grid):
    """
    :param kde: KernelDensity instance from scikit learn
    :param data input data
    :param grid Grid on which to evaluate the density
    :return density
    """
    return np.exp(kde.fit(data).score_samples(grid))


def mp_kde(grid, data, bandwidth, shape=None, kernel="epanechnikov", norm=False, absolute=False, sampling=None):
    """
    Parellisation for kernel density estimation
    :param grid Grid on which to evaluate the density
    :param data input data
    :param bandwidth: Bandwidth of kernel
    :param shape: If set, reshape output data
    :param kernel: e.g. "epanechnikov" or "gaussian"
    :param norm: Normalize output. "max", "mean", "sum"
    :param absolute: Whether to return absolute numbers
    :param sampling: Sampling factor of grid
    :return: Kernel densities
    """

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
# Extinction mapping functions
def get_extinction_pixel(xgrid, ygrid, xdata, ydata, ext, var, bandwidth, method, nicest=False):
    # TODO: Check what the slow part of this is and try to improve
    """
    Calculate extinction fro a given grid point
    :param xgrid: X grid point
    :param ygrid: Y grid point
    :param xdata: X data
    :param ydata: Y data
    :param ext: extinction data for each source
    :param var: extinction variance for each source
    :param bandwidth: bandwidth of kernel
    :param method: Method to be used. e.g. "median", "gaussian", "epanechnikov", "uniform", "triangular"
    :param nicest: Wether or not to use NICEST weight adjustment
    :return: extintion, variance, and number of sources for pixel
    """

    # In case the average or median is to be calculated, I set bandwidth == truncation scale
    if (method == "average") | (method == "median"):
        trunc = 1
    else:
        trunc = 3

    # Truncate input data to either pixel size (median or average) or to 3 times the bandwidth for the given kernel
    index = (xdata > xgrid - trunc * bandwidth / 2) & (xdata < xgrid + trunc * bandwidth / 2) & \
            (ydata > ygrid - trunc * bandwidth / 2) & (ydata < ygrid + trunc * bandwidth / 2)

    # If we have nothing here, immediately return
    if np.sum(index) == 0:
        return np.nan, np.nan, 0

    # pre-filter
    ext, var, xdata, ydata = ext[index], var[index], xdata[index], ydata[index]

    # Calculate the distance to the grid point in a spherical metric
    # TODO: Check if selection of sources for a pixel is OK
    dis = np.degrees(np.arccos(np.sin(np.radians(ydata)) * np.sin(np.radians(ygrid)) +
                               np.cos(np.radians(ydata)) * np.cos(np.radians(ygrid)) *
                               np.cos(np.radians(xdata - xgrid))))

    # For all kernel which use weights, we truncate a circular patch
    if method not in ["average", "median"]:
        index = dis < trunc * bandwidth / 2
        # Current data in bin within truncation radius
        ext, var, dis, xdata, ydata = ext[index], var[index], dis[index], xdata[index], ydata[index]

    # Calulate number of sources left over after truncation
    npixel = np.sum(index)

    # If there are no stars, or less than three extinction measurements skip
    if (npixel == 0) or (np.sum(np.isfinite(ext)) < 3):
        return np.nan, np.nan, npixel

    # Based on chosen method calculate extinction or weights
    if method == "average":
        pixel_ext = np.nanmean(ext)
        pixel_var = np.sqrt(np.nansum(var)) / npixel
        return pixel_ext, pixel_var, npixel
    elif method == "median":
        pixel_ext = np.nanmedian(ext)
        pixel_mad = np.median(np.abs(ext - pixel_ext))
        return pixel_ext, pixel_mad, npixel
    elif method == "uniform":
        weights = np.ones_like(ext)
    elif method == "triangular":
        weights = 1 - np.abs(dis / bandwidth)
    elif method == "gaussian":
        weights = np.exp(-0.5 * (dis / bandwidth)**2)
    elif method == "epanechnikov":
        weights = 1 - (dis / bandwidth)**2
    else:
        raise TypeError("method not implemented")

    # Set negative weights to 0
    weights[weights < 0] = 0

    # Modify weights for NICEST
    # TODO: Find out how to generalise this and add to class
    slope = 0.33
    k_lambda = 1
    if nicest:
        weights *= 10**(slope * k_lambda * ext)

    # Mask weights with no extinction
    weights[~np.isfinite(ext)] = np.nan

    # Get extinction based on weights
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pixel_ext = np.nansum(weights * ext) / np.nansum(weights)
        pixel_var = np.nansum(weights**2 * var) / np.nansum(weights)**2

    # Return
    if nicest:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Calculate correction factor
            cor = slope * k_lambda * np.log(10) * np.nansum(weights * var) / np.nansum(weights)
        return pixel_ext - cor, pixel_var, npixel
    else:
        return pixel_ext, pixel_var, npixel


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Helper methods for plotting
def axes_combinations(ndim, ax_size=None):
    """
    Creates a grid of axes to plot all combinations of data
    :param ndim: number of dimensions
    :param ax_size: basic size adjustment parameter
    :return: List of axes which can be used for plotting
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


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Various helper methods

# ----------------------------------------------------------------------
def weighted_avg(values, weights):
    """
    Calculates weighted mean and standard deviation
    :param values: data values
    :param weights: weights
    :return: weighted mean and variance
    """

    average = np.nansum(values * weights) / np.nansum(weights)
    # noinspection PyTypeChecker
    variance = np.nansum((values - average)**2 * weights) / np.nansum(weights)
    return average, variance
