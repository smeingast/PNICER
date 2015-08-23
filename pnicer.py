from __future__ import absolute_import, division, print_function


# ----------------------------------------------------------------------
# Import stuff
import numpy as np
import warnings
import multiprocessing
import matplotlib.pyplot as plt

from itertools import combinations, repeat
from time import time
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
    def _all_combinations(self):

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
    def pnicer(self, control, bin_grid=0.1, bin_ext=0.05):

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

        # With our grid, we evaluate the density on it for the control field (!)
        # TODO: Maybe implement different kernels based on errors...humm
        """ This is the slow part in PNICER...!! """
        # TODO: implement correct bandwidth
        kde = KernelDensity(kernel="epanechnikov", bandwidth=0.1)

        xgrid_split = np.array_split(xgrid, multiprocessing.cpu_count(), axis=0)

        # Create process pool
        p = multiprocessing.Pool()
        # Submit tasks
        mp = p.map(_mp_kde_star, zip(repeat(kde), repeat(np.vstack(control_rot.features).T),
                                     xgrid_split))
        # Close pool (!)
        p.close()
        # Unpack results
        dens = np.concatenate(mp)

        # Old serial approach
        # log_dens = kde.fit(control_rot.feature_goodmatrix.T).score_samples(xgrid)
        # dens = np.exp(log_dens)

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
    def plot_combinations_kde(self, path=None, ax_size=None, grid_bw=0.1):

        if ax_size is None:
            ax_size = [3, 3]

        # Create figure
        fig, axes = axes_combinations(self.n_features, ax_size=ax_size)

        # Get 2D combination indices
        for idx, ax in zip(combinations(range(self.n_features), 2), axes):

            # Get clean data from the current combination
            mask = np.prod(np.vstack([self.features_masks[idx[0]], self.features_masks[idx[1]]]), axis=0, dtype=bool)
            # data = np.vstack([, ])

            # We need a square grid!
            l, h = np.min([x[0] for x in self.plotrange]), np.max([x[1] for x in self.plotrange])

            xgrid, ygrid = np.meshgrid(np.arange(start=l, stop=h, step=grid_bw),
                                       np.arange(start=l, stop=h, step=grid_bw))

            dens = mp_kde(xgrid=xgrid, ygrid=ygrid,
                          xdata=self.features[idx[0]][mask], ydata=self.features[idx[1]][mask], bandwidth=grid_bw * 2)

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
    def plot_spatial_kde(self, frame, pixsize=10/60, path=None, kernel="epanechnikov"):

        # Get a WCS grid
        header, lon_grid, lat_grid = self.build_wcs_grid(frame=frame, pixsize=pixsize)

        # Get aspect ratio
        ar = lon_grid.shape[0] / lon_grid.shape[1]

        # Determine number of columns and rows
        ncols = np.floor(np.sqrt(self.n_features)).astype(int)
        nrows = np.ceil(np.sqrt(self.n_features)).astype(int)

        # Create grid
        plt.figure(figsize=[10 * ncols, 10 * nrows * ar])
        grid = GridSpec(ncols=ncols, nrows=nrows, bottom=0.05, top=0.95, left=0.05, right=0.95, hspace=0.2, wspace=0.2)

        # To avoid editor warning
        scale = 1

        # Loop over features and plot
        for idx in range(self.n_features):

            # Add axes
            ax = plt.subplot(grid[idx], projection=WCS(header=header))

            # Get density
            dens = mp_kde(xgrid=lon_grid, ygrid=lat_grid,
                          xdata=self.lon[self.features_masks[idx]], ydata=self.lat[self.features_masks[idx]],
                          bandwidth=pixsize * 2, kernel=kernel)

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

        # Determine number of columns and rows
        ncols = np.floor(np.sqrt(self.n_features - 1)).astype(int)
        nrows = np.ceil(np.sqrt(self.n_features - 1)).astype(int)

        # Create grid
        plt.figure(figsize=[10 * ncols, 10 * nrows * ar])
        grid = GridSpec(ncols=ncols, nrows=nrows, bottom=0.1, top=0.95, left=0.1, right=0.95, hspace=0.2, wspace=0.2)

        # To avoid editor warnings
        dens, dens_norm = 0, 0

        # Loop over features and plot
        for idx in range(self.n_features):

            # Save previous density
            if idx > 0:
                dens_norm = dens.copy()

            # Get density
            dens = mp_kde(xgrid=lon_grid, ygrid=lat_grid, xdata=self.lon[self.features_masks[idx]][::skip],
                          ydata=self.lat[self.features_masks[idx]][::skip], bandwidth=pixsize * 2, kernel=kernel)

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
                if idx % ncols == 1:
                    lat.set_axislabel("Latitude")
                if idx / ncols > 1:
                    lon.set_axislabel("Longitude")

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
        colors_combinations = colors._all_combinations()

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
    # NICER implementation
    def nicer(self, control):

        # TODO: Something is weird. I get 747291 good measurements which is the same as good H band measurements

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
        return ext.data, ext_err

    # ----------------------------------------------------------------------
    # PNICER implementation for all combinations
    def pnicer_combinations(self, control, bin_grid=0.1, bin_ext=0.05):

        # Check instances
        if control.__class__ != self.__class__:
            raise TypeError("input and control instance do not match")

        # We loop over all combinations
        all_ext, all_ext_err = [], []

        names = []

        # Here we loop over color combinations since this is faster
        for sc, cc in zip(self.color_combinations(), control.color_combinations()):

            ext, ext_err = sc.pnicer(control=cc, bin_ext=bin_ext, bin_grid=bin_grid)
            all_ext.append(ext)
            all_ext_err.append(ext_err)

            names.append("_".join(sc.features_names))
            print(sc.features_names)
            print(sc.extvec.extvec)

        all_ext = np.array(all_ext)
        all_ext_err = np.array(all_ext_err)

        # nf = all_ext.shape[0]
        # from matplotlib.pyplot import GridSpec
        # fig1 = plt.figure(figsize=[20, 15])
        # grid = GridSpec(ncols=5, nrows=3, bottom=0.05, top=0.95, left=0.05, right=0.95, hspace=0.1, wspace=0.1)
        #
        # for idx in range(nf):
        #
        #     ax = plt.subplot(grid[idx])
        #     ax.hist(all_ext[idx, :], bins=100, range=(-1.5, 2))
        #     ax.annotate(names[idx], xy=(0.05, 0.9), xycoords="axes fraction")
        #
        # fig2 = plt.figure(figsize=[20, 15])
        # grid = GridSpec(ncols=5, nrows=3, bottom=0.05, top=0.95, left=0.05, right=0.95, hspace=0.1, wspace=0.1)
        #
        # r = (-1.5, 2)
        # for idx in range(nf):
        #
        #     ax = plt.subplot(grid[idx])
        #     ax.hist(all_ext_err[idx, :], bins=20, range=(0, 1))
        #     ax.annotate(names[idx], xy=(0.05, 0.9), xycoords="axes fraction")
        #
        # plt.show()
        #
        # exit()

        # Calculate weighted average while ignoring the NaN warnings
        # TODO: Check error calculation before changing weight rule
        # TODO: Maybe the weights should be calculated by the standard deviation of the Aks in the control field!
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            weights = 1 / all_ext_err
            # ext = np.nansum(all_ext * weights, axis=0) / np.nansum(weights, axis=0)
            # ext = np.nanmean(all_ext, axis=0)

            all_ext_err[~np.isfinite(all_ext_err)] = 100
            idx = np.nanargmin(all_ext_err, axis=0)
            ext = []
            for m, e in zip(idx, all_ext.T):
                ext.append(e[m])
            ext = np.array(ext)

            # TODO: Check if this is correct
            ext_err = np.sqrt(np.nansum((weights * all_ext_err) ** 2, axis=0) / np.nansum(weights, axis=0) ** 2)

        return ext, ext_err


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Colors class
class Colors(DataBase):

    def __init__(self, mag, err, extvec, lon=None, lat=None, names=None):
        super(Colors, self).__init__(mag=mag, err=err, extvec=extvec, lon=lon, lat=lat, names=names)


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

    def __init__(self, lon, lat, extinction, error=None):

        self.lon = lon
        self.lat = lat
        self.extinction = extinction
        self.error = error

        if self.error is None:
            self.error = np.zeros_like(extinction)

        # extinction and error must have same length
        if len(extinction) != len(error):
            raise ValueError("Extinction and error arrays must have equal length")


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Helper top level methods for parallel processing

# KDE top level functions for parallelisation
def _mp_kde(kde, data, grid):
    return np.exp(kde.fit(data).score_samples(grid)) * data.shape[0]


def _mp_kde_star(args):
    return _mp_kde(*args)


def mp_kde(xgrid, ygrid, xdata, ydata, bandwidth, kernel="epanechnikov"):

    grid = np.vstack([xgrid.ravel(), ygrid.ravel()]).T

    # Split for parallel processing
    grid_split = np.array_split(grid, multiprocessing.cpu_count(), axis=0)

    # Define kernel according to Nyquist sampling
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)

    # Create process pool
    p = multiprocessing.Pool()

    # Prepare data grid
    data = np.vstack([xdata, ydata])

    # Submit tasks
    mp = p.map(_mp_kde_star, zip(repeat(kde), repeat(data.T), grid_split))

    # Close pool (!)
    p.close()

    # Unpack results and return
    return np.concatenate(mp).reshape(xgrid.shape)


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
    # average = np.average(values, weights=weights)
    # variance = np.average((values - average)**2, weights=weights)
    return average, np.sqrt(variance)
