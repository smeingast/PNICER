from __future__ import absolute_import, division, print_function


# ----------------------------------------------------------------------
# Import stuff
import numpy as np
import warnings
import multiprocessing
import matplotlib.pyplot as plt

from itertools import combinations, repeat
from astropy import wcs
from astropy.io import fits
from wcsaxes import WCS
from sklearn.neighbors import KernelDensity, NearestNeighbors
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from matplotlib.pyplot import GridSpec


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Define general data class
class DataBase:

    def __init__(self, mag, err, extvec, lon=None, lat=None, names=None):

        self.features = mag
        self.features_err = err
        self.lon = lon
        self.lat = lat
        self.features_names = names
        self.n_features = len(mag)
        self.extvec = ExtinctionVector(extvec=extvec)

        # Define combination properties determined while running PNICER
        self._ext_combinations = None
        self._exterr_combinations = None
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

        # ----------------------------------------------------------------------
        # Plot range
        # noinspection PyTypeChecker
        self.plotrange = [(np.floor(np.percentile(x[m], 0.01)), np.ceil(np.percentile(x[m], 99.99)))
                          for x, m in zip(self.features, self.features_masks)]

    # ----------------------------------------------------------------------
    # Static helper methods in namespace

    # Method to round data to arbitrary precision
    @staticmethod
    def round_partial(data, precision):
        return np.around(data / precision) * precision

    # Method to create grid from data
    @staticmethod
    def build_grid(data, precision):

        grid_data = DataBase.round_partial(data=data, precision=precision).T

        # Get unique positions for coordinates
        dummy = np.ascontiguousarray(grid_data).view(np.dtype((np.void, grid_data.dtype.itemsize * grid_data.shape[1])))
        _, idx = np.unique(dummy, return_index=True)

        return grid_data[idx].T

    # ----------------------------------------------------------------------
    # Method to rotate data space given a rotation matrix
    def rotate(self):

        mask = np.prod(np.vstack(self.features_masks), axis=0, dtype=bool)
        data = np.vstack(self.features).T[mask].T
        err = np.vstack(self.features_err).T[mask].T

        # Rotate data
        rotdata = self.extvec.rotmatrix.dot(data)

        # Rotate extinction vector
        extvec = self.extvec.extinction_rot

        # In case no coordinates are supplied
        if self.lon is not None:
            lon = self.lon[mask]
        else:
            lon = None
        if self.lat is not None:
            lat = self.lat[mask]
        else:
            lat = None

        # Return
        return self.__class__(mag=[rotdata[idx, :] for idx in range(self.n_features)],
                              err=[err[idx, :] for idx in range(self.n_features)],
                              extvec=extvec, lon=lon, lat=lat,
                              names=[x + "_rot" for x in self.features_names])

    # ----------------------------------------------------------------------
    # Method to get all combinations of input features
    def all_combinations(self):

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
    # Main PNICER routine to get extinction
    def _pnicer_single(self, control, bin_grid, bin_ext, kernel):

        # Check instances
        if self.__class__ != control.__class__:
            raise TypeError("Input and control instance not compatible")

        # Let's rotate the data spaces
        science_rot, control_rot = self.rotate(), control.rotate()

        # Now we build a grid from the rotated data for all components but the first
        grid_data = DataBase.build_grid(data=np.vstack(science_rot.features)[1:, :], precision=bin_grid)

        # Create a grid to evaluate along the reddening vector
        grid_ext = np.arange(start=np.floor(min(science_rot.features[0])),
                             stop=np.ceil(max(science_rot.features[0])), step=bin_ext)

        # Now we combine those to get _all_ grid points
        xgrid = np.column_stack([np.tile(grid_ext, grid_data.shape[1]),
                                 np.repeat(grid_data, len(grid_ext), axis=1).T])

        # Get bandwidth of kernel
        # TODO: Maybe implement different kernels based on errors...humm
        bandwidth = np.mean(np.nanmean(self.features_err, axis=1))

        # With our grid, we evaluate the density on it for the control field (!)
        dens = mp_kde(grid=xgrid, data=np.vstack(control_rot.features).T, bandwidth=bandwidth, kernel=kernel)

        # Get all unique vectors
        # TODO: Require at least, say, 3 sources in a vector, otherwise discard the PDF.
        dens_vectors = dens.reshape([grid_data.shape[1], len(grid_ext)])

        # Calculate weighted average and standard deviation for each vector
        grid_mean, grid_std = [], []
        for vec in dens_vectors:

            # In case there are only 0 probabilities
            if np.sum(vec) < 1E-6:
                grid_mean.append(np.nan)
                grid_std.append(np.nan)
            else:
                a, b = weighted_avg(values=grid_ext, weights=vec)
                grid_mean.append(a)
                grid_std.append(b / self.extvec.extinction_norm)  # The normalisation converts this to extinction

        # Convert to arrays
        grid_std = np.array(grid_std)
        grid_mean = np.array(grid_mean)

        # Let's get the nearest neighbor grid point for each source
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(grid_data.T)
        _, indices = nbrs.kneighbors(np.vstack(science_rot.features)[1:, :].T)
        indices = indices[:, 0]

        # Now we have the instrisic colors for each vector and indices for all sources.
        # It's time to calculate the extinction. :)
        ext = (science_rot.features[0] - grid_mean[indices]) / self.extvec.extinction_norm
        ext_err = grid_std[indices]

        # Lastly we put all the extinction measurements back into a full array
        out = np.full(self.n_data, fill_value=np.nan, dtype=float)
        outerr = np.full(self.n_data, fill_value=np.nan, dtype=float)

        # Combined mask for all features
        mask = np.prod(np.vstack(self.features_masks), axis=0, dtype=bool)

        out[mask], outerr[mask] = ext, ext_err

        return out, outerr

    # ----------------------------------------------------------------------
    # PNICER base implementation for combinations
    def _pnicer_combinations(self, control, comb, bin_grid, bin_ext, kernel):

        # Check instances
        if control.__class__ != self.__class__:
            raise TypeError("input and control instance do not match")

        # We loop over all combinations
        all_ext, all_exterr, names = [], [], []

        # Here we loop over color combinations since this is faster
        i = 0
        for sc, cc in comb:

            assert isinstance(sc, DataBase)
            # Run PNICER for current combination
            ext, ext_err = sc._pnicer_single(control=cc, bin_ext=bin_ext, bin_grid=bin_grid, kernel=kernel)

            # Append data
            all_ext.append(ext)
            all_exterr.append(ext_err)
            names.append("(" + ",".join(sc.features_names) + ")")
            i += 1

        # Convert to arrays and save
        all_ext = np.array(all_ext)
        self._ext_combinations = all_ext.copy()
        all_exterr = np.array(all_exterr)
        self._exterr_combinations = all_exterr.copy()
        self._combination_names = names
        self._n_combinations = i

        # Chose extinction with minimum error
        all_exterr[~np.isfinite(all_exterr)] = 100 * np.nanmax(all_exterr)
        ext = all_ext[np.argmin(all_exterr, axis=0), np.arange(self.n_data)]
        exterr = all_exterr[np.argmin(all_exterr, axis=0), np.arange(self.n_data)]

        return Extinction(db=self, extinction=ext, error=exterr)

    # ----------------------------------------------------------------------
    # Method to build a WCS grid with a valid projection
    def build_wcs_grid(self, frame, pixsize=10./60):

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
        # TODO: Set median of mean of something instead of -20
        mywcs.wcs.set_pv([(2, 1, -20.)])

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

    # 2D Scatter plot of combinations
    def plot_combinations_scatter(self, path=None, ax_size=None, **kwargs):

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

    # KDE for all 2D combinations of features
    def plot_combinations_kde(self, path=None, ax_size=None, grid_bw=0.1, kernel="epanechnikov"):

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

    # Plot source densities for features
    def plot_spatial_kde(self, frame, pixsize=10/60, path=None, kernel="epanechnikov", skip=1):

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
            ax = plt.subplot(grid[idx], projection=WCS(header=header))

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

    # Plot source densities for features
    def plot_spatial_kde_gain(self, frame, pixsize=10/60, path=None, kernel="epanechnikov", skip=1):

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
                ax = plt.subplot(grid[idx - 1], projection=WCS(header=header))

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

    def hist_extinction_combinations(self, path=None):

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
        ax2_range = [0, DataBase.round_partial(np.nanmean(self._exterr_combinations)
                                               + 3.5 * np.nanstd(self._exterr_combinations), 0.1)]

        plt.figure(figsize=[5 * n_panels[1], 5 * n_panels[0] * 0.5])
        grid = GridSpec(ncols=n_panels[1], nrows=n_panels[0], bottom=0.05, top=0.95, left=0.05, right=0.95,
                        hspace=0.05, wspace=0.05)

        for idx in range(self._n_combinations):

            ax1 = plt.subplot(grid[idx])
            ax1.hist(self._ext_combinations[idx, :], bins=25, range=ax1_range, histtype="stepfilled", lw=0,
                     alpha=0.5, color="#3288bd", log=True)
            ax1.annotate(self._combination_names[idx], xy=(0.95, 0.9), xycoords="axes fraction", ha="right")

            ax2 = ax1.twiny()
            ax2.hist(self._exterr_combinations[idx, :], bins=25, range=ax2_range, histtype="step", lw=2,
                     alpha=0.8, color="#66c2a5", log=True)

            # Set and delete labels
            if idx >= n_panels[0] * n_panels[1] - n_panels[1]:
                ax1.set_xlabel("Extinction")
            else:
                ax1.axes.xaxis.set_ticklabels([])
            if idx % n_panels[1] == 0:
                ax1.set_ylabel("N")
            else:
                ax1.axes.yaxis.set_ticklabels([])
            if idx < n_panels[1]:
                ax2.set_xlabel("Error")
            else:
                ax2.axes.xaxis.set_ticklabels([])

            # Delete first and last label
            xticks = ax1.xaxis.get_major_ticks()
            xticks[0].label1.set_visible(False)
            xticks[-1].label1.set_visible(False)
            xticks = ax2.xaxis.get_major_ticks()
            xticks[0].label2.set_visible(False)
            xticks[-1].label2.set_visible(False)
            """For some reason this does not work for the y axis..."""

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
        super(Magnitudes, self).__init__(mag=mag, err=err, extvec=extvec, lon=lon, lat=lat, names=names)

    # ----------------------------------------------------------------------
    # Method to convert to colors
    def mag2color(self):

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

        # First get all colors...
        colors = self.mag2color()

        # ...and all combinations of colors
        colors_combinations = colors.all_combinations()

        # TODO: Decide what to add here!
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
    def pnicer(self, control, bin_grid=0.1, bin_ext=0.05, kernel="epanechnikov", use_color=False):
        if use_color:
            comb = zip(self.color_combinations(), control.color_combinations())
        else:
            comb = zip(self.all_combinations(), control.all_combinations())

        return self._pnicer_combinations(control=control, comb=comb, bin_grid=bin_grid, bin_ext=bin_ext, kernel=kernel)

    # ----------------------------------------------------------------------
    # NICER implementation
    def nicer(self, control):

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
        ext_err = np.sqrt(np.array([np.dot(b.data[:, idx], first[idx, :]) for idx in range(self.n_data)]))

        # Combined mask for all features
        # mask = np.prod(np.vstack(self.features_masks), axis=0, dtype=bool)
        # ext[mask] = ext_err[mask] = np.nan

        # ...and return :)
        # TODO: Also return extinction class here!
        return ext.data, ext_err


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Colors class
class Colors(DataBase):

    def __init__(self, mag, err, extvec, lon=None, lat=None, names=None):
        super(Colors, self).__init__(mag=mag, err=err, extvec=extvec, lon=lon, lat=lat, names=names)

    # ----------------------------------------------------------------------
    def pnicer(self, control, bin_grid=0.1, bin_ext=0.05, kernel="epanechnikov"):
        comb = zip(self.all_combinations(), control.all_combinations())
        return self._pnicer_combinations(control=control, comb=comb, bin_grid=bin_grid, bin_ext=bin_ext, kernel=kernel)


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Class for extinction vector omponents
class ExtinctionVector:

    def __init__(self, extvec):
        self.extvec = extvec
        self.n_dimensions = len(extvec)

    # ----------------------------------------------------------------------
    # Some static helper methods within namespace

    # Calculate unit vectors for a given number of dimensions
    @staticmethod
    def unit_vectors(n_dimensions):
        return [np.array([1.0 if i == l else 0.0 for i in range(n_dimensions)]) for l in range(n_dimensions)]

    # Method to determine the rotation matrix so that the rotated first vector component is the only non-zero component
    @staticmethod
    def get_rotmatrix(vector):

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
    # Rotation matrix for all extinction components of this instance
    _rotmatrix = None

    @property
    def rotmatrix(self):

        # Check if already determined
        if self._rotmatrix is not None:
            return self._rotmatrix

        self._rotmatrix = ExtinctionVector.get_rotmatrix(self.extvec)
        return self._rotmatrix

    # ----------------------------------------------------------------------
    # Rotated input extinction vector
    _extinction_rot = None

    @property
    def extinction_rot(self):

        # Check if already determined
        if self._extinction_rot is not None:
            return self._extinction_rot

        self._extinction_rot = self.rotmatrix.dot(self.extvec)
        return self._extinction_rot

    # ----------------------------------------------------------------------
    # Normalization component for projected extinction vector
    _extinction_norm = None

    @property
    def extinction_norm(self):

        # Check if already determined
        if self._extinction_norm is not None:
            return self._extinction_norm

        self._extinction_norm = self.extinction_rot[0]
        return self._extinction_norm


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Class for extinction measurements
class Extinction:

    def __init__(self, db, extinction, error=None):

        # Check if db is really a DataBase instance
        assert isinstance(db, DataBase)

        self.db = db
        self.extinction = extinction
        self.error = error
        if self.error is None:
            self.error = np.zeros_like(extinction)

        # extinction and error must have same length
        if len(extinction) != len(error):
            raise ValueError("Extinction and error arrays must have equal length")

    # ----------------------------------------------------------------------
    # Method to build an extinction map
    def build_map(self, bandwidth, method="median", nicest=False):

        # First let's get a grid
        grid_header, grid_lon, grid_lat = self.db.build_wcs_grid(frame="galactic", pixsize=bandwidth / 2.)

        # Create process pool
        p = multiprocessing.Pool()
        # Submit tasks
        mp = p.map(get_extinction_pixel_star, zip(grid_lon.ravel(), grid_lat.ravel(), repeat(self.db.lon),
                                                  repeat(self.db.lat), repeat(self.extinction), repeat(self.error),
                                                  repeat(bandwidth), repeat(method), repeat(nicest)))
        # Close pool (!)
        p.close()

        # Unpack results
        extmap, extmap_err, nummap = list(zip(*mp))

        # reshape
        extmap = np.array(extmap).reshape(grid_lon.shape)
        extmap_err = np.array(extmap_err).reshape(grid_lon.shape)
        # TODO: Maybe number map should be KDE with bandwidth
        nummap = np.array(nummap).reshape(grid_lon.shape)

        # Return extinction map instance
        return ExtinctionMap(ext=extmap, ext_err=extmap_err, num=nummap, header=grid_header)


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Extinction map class
class ExtinctionMap:

    def __init__(self, ext, ext_err, num, header):
        self.map = ext
        self.err = ext_err
        self.num = num
        self.shape = self.map.shape
        self.fits_header = header

        # Input must be 2D
        if (len(self.map.shape) != 2) | (len(self.err.shape) != 2) | (len(self.num.shape) != 2):
            raise TypeError("Input must be 2D arrays")

    # Simple method to plot extinction map
    def plot_map(self, path=None, size=10):

        # TODO: Make this a bit more fancy
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[size, size * (self.shape[0] / self.shape[1])])

        im = ax.imshow(self.map, origin="lower", interpolation="nearest", vmin=0, vmax=2)
        cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        # Save or show figure
        if path is None:
            plt.show()
        else:
            plt.savefig(path, bbox_inches="tight")
        plt.close()

    # Save extinciton map as FITS file
    def save(self, path):
        # TODO: Also save number map
        # Create and save
        hdulist = fits.HDUList([fits.PrimaryHDU(),
                                fits.ImageHDU(data=self.map, header=self.fits_header),
                                fits.ImageHDU(data=self.err, header=self.fits_header)])
        hdulist.writeto(path, clobber=True)


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Helper top level methods for parallel processing

# ----------------------------------------------------------------------
# KDE functions for parallelisation
def _mp_kde(kde, data, grid):
    return np.exp(kde.fit(data).score_samples(grid)) * data.shape[0]


def _mp_kde_star(args):
    return _mp_kde(*args)


def mp_kde(grid, data, bandwidth, shape=None, kernel="epanechnikov"):

    # Split for parallel processing
    grid_split = np.array_split(grid, multiprocessing.cpu_count(), axis=0)

    # Define kernel according to Nyquist sampling
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)

    # Create process pool
    p = multiprocessing.Pool()

    # Submit tasks
    mp = p.map(_mp_kde_star, zip(repeat(kde), repeat(data), grid_split))

    # Close pool (!)
    p.close()

    # Unpack results and return
    if shape is None:
        return np.concatenate(mp)
    else:
        return np.concatenate(mp).reshape(shape)


# ----------------------------------------------------------------------
# Extinction mapping functions
def get_extinction_pixel(xgrid, ygrid, xdata, ydata, ext, ext_err, bandwidth, method, nicest=False):

    # In case the average or median is to be calculated, I set bandwidth == truncation scale
    if (method == "average") | (method == "median"):
        trunc = 1
    else:
        trunc = 3

    # Restrict the input array to some manageable size
    index = (xdata > xgrid - trunc * bandwidth) & (xdata < xgrid + trunc * bandwidth) & \
            (ydata > ygrid - trunc * bandwidth) & (ydata < ygrid + trunc * bandwidth)

    # If we have nothing here, immediately return
    if np.sum(index) == 0:
        return np.nan, np.nan, 0

    # pre-filter
    ext, xdata, ydata = ext[index], xdata[index], ydata[index]

    # Calculate the distance to the grid point in a spherical metric
    dis = np.degrees(np.arccos(np.sin(np.radians(ydata)) * np.sin(np.radians(ygrid)) +
                               np.cos(np.radians(ydata)) * np.cos(np.radians(ygrid)) *
                               np.cos(np.radians(xdata - xgrid))))

    # Get sources within the truncation radius
    index = dis < trunc * bandwidth

    # Current data in bin within truncation radius
    ext = ext[index]
    # TODO: Check what to do with negative extinction for weighting
    # This raises a lot of warnings!
    # ext[ext < 0] = 0
    ext_err = ext_err[index]
    dis = dis[index]

    # Calulate number of sources in bin
    # TODO: Should be the number of stars within pixel!
    npixel = np.sum(index)

    # If there are no stars, or less than two extinction measurements skip
    if (npixel == 0) or (np.sum(np.isfinite(ext)) < 2):
        return np.nan, np.nan, npixel

    # Based on chosen method calculate extinction or weights
    # TODO: Implement correct errors for mean and median
    if method == "average":
        return np.nanmean(ext), np.nan, npixel
    elif method == "median":
        return np.nanmedian(ext), np.nan, npixel
    elif method == "uniform":
        weights = np.ones_like(ext)
    elif method == "triangular":
        weights = 1 - np.abs((dis / bandwidth))
    elif method == "gauss":
        weights = 1 / (2 * np.pi) * np.exp(-0.5 * (dis / bandwidth)**2)
    elif method == "epanechnikov":
        weights = 1 - (dis / bandwidth)**2
    else:
        raise TypeError("method not implemented")

    # Set negative weights to 0
    weights[weights < 0] = 0

    # Modify weights for NICEST
    if nicest:
        weights *= 10**(0.33 * ext)

    # Mask weights with no extinction
    weights[~np.isfinite(ext)] = np.nan

    # Get extinction based on weights
    epixel = np.nansum(weights * np.array(ext)) / np.nansum(weights)
    epixel_err = np.sqrt(np.nansum((weights * np.array(ext_err)) ** 2) / np.nansum(weights) ** 2)

    # Return
    return epixel, epixel_err, npixel


# ----------------------------------------------------------------------
# For pool.starmap() method to accept multiple arguments
def get_extinction_pixel_star(args):
    return get_extinction_pixel(*args)


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Helper methods for plotting
def axes_combinations(ndim, ax_size=None):

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
# Weighted mean and standard deviation
def weighted_avg(values, weights):

    average = np.nansum(values * weights) / np.nansum(weights)
    # noinspection PyTypeChecker
    variance = np.nansum((values - average)**2 * weights) / np.nansum(weights)
    return average, np.sqrt(variance)
