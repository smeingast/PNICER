# ----------------------------------------------------------------------
# Import stuff
import wcsaxes
import warnings
import brewer2mpl
import numpy as np
import matplotlib.pyplot as plt

from itertools import repeat
from multiprocessing import Pool
from matplotlib.pyplot import GridSpec
from matplotlib.ticker import MultipleLocator
from MyFunctions import distance_on_unit_sphere
from helper import pnicer_ini, get_beta_ols, get_beta_bces, get_beta_lines


# ----------------------------------------------------------------------
# Load colorbrewer colormap
cmap1 = brewer2mpl.get_map("Spectral", "Diverging", number=11, reverse=True).get_mpl_colormap(N=10, gamma=1)
cmap2 = brewer2mpl.get_map("YlGnBu", "Sequential", number=9, reverse=True).get_mpl_colormap(N=10, gamma=1)


# ----------------------------------------------------------------------
# Intialize PNICER
science, control = pnicer_ini(skip_science=1, skip_control=1, n_features=5, color=False)
science_color, control_color = science.mag2color(), control.mag2color()


# ----------------------------------------------------------------------
# Get grid
header, all_glon, all_glat = science.build_wcs_grid(frame="galactic", pixsize=20 / 60)
grid_shape = all_glat.shape
glon_range, glat_range = all_glon.ravel(), all_glat.ravel()


# ----------------------------------------------------------------------
# Define parameter range
ajr = (2.4, 2.7)
ahr = (1.4, 1.7)
a1r = (0.5, 0.7)
a2r = (0.3, 0.6)

# Define baseline extinction
a_hk = 1.55


# ----------------------------------------------------------------------
# Initial guess for extinction
# ext = science_color.pnicer(control=control_color).extinction


# ----------------------------------------------------------------------
# Loop over bins and get extinction law
def get_slope(glon_pix, glat_pix, glon_all, glat_all, maxdis):

    # Calculate distance to all other sources from current grid point
    dis = np.degrees(distance_on_unit_sphere(ra1=np.radians(glon_all), dec1=np.radians(glat_all),
                                             ra2=np.radians(glon_pix), dec2=np.radians(glat_pix)))

    # Make pre-selection of data
    for data in [science_color, control_color]:
        # Define base photometric filter common to science and control field
        fil = (data.features[0] + 1 < (ajr[0] - ahr[0]) / (ahr[0] - 1) * (data.features[1] + 0.7)) & \
              (data.features[0] + 1 > (ajr[1] - ahr[1]) / (ahr[1] - 1) * (data.features[1] + 0.7)) & \
              (data.features[0] + 1 > (ajr[0] - ahr[0]) / (1 - a1r[0]) * (data.features[2] + 0.5)) & \
              (data.features[0] + 1 < (ajr[1] - ahr[1]) / (1 - a1r[1]) * (data.features[2] + 0.5)) & \
              (data.features[0] + 1 > (ajr[0] - ahr[0]) / (a1r[0] - a2r[0]) * (data.features[3] + 0.2)) & \
              (data.features[0] + 1 < (ajr[1] - ahr[1]) / (a1r[1] - a2r[1]) * (data.features[3] + 0.2)) & \
              (np.nanmax(np.array(data.features_err), axis=0) < 0.1)

        # Test no filtering
        # fil = data.combined_mask

        if data == science_color:
            sfil = fil.copy() & (dis < maxdis)
            # sfil = fil & (ext > 0.1) & (dis < maxdis)
        else:
            cfil = fil.copy()

    # Save number of sources within filter limits
    n = np.sum(sfil)

    # Skip if there are too few
    if np.sum(sfil) < 100:
        return np.nan, n

    plot_index = [0, 1, 2]
    data_index = [[1, 2, 2, 0], [1, 2, 2, 3], [1, 2, 2, 4]]
    for (idx1, idx2, idx3, idx4), pidx in zip(data_index, plot_index):

        # Get shortcut for data
        xdata = science.features[idx1][sfil] - science.features[idx2][sfil]
        ydata = science.features[idx3][sfil] - science.features[idx4][sfil]

        # # Get ordinary least squares slope
        # beta_ols = get_beta_ols(xj=xdata, yj=ydata)
        #
        # # Get BCES slope
        # beta_bces = get_beta_bces(x_sc=xdata, y_sc=ydata,
        #                           cov_err_sc=-np.mean(science.features_err[idx2][sfil])**2,
        #                           var_err_sc=np.mean(science.features_err[idx1][sfil])**2 +
        #                           np.mean(science.features_err[idx2][sfil])**2)

        # Get LINES slope
        beta_lines = get_beta_lines(x_sc=xdata, y_sc=ydata,
                                    x_cf=control.features[idx1][cfil] - control.features[idx2][cfil],
                                    y_cf=control.features[idx3][cfil] - control.features[idx4][cfil],
                                    cov_err_sc=-np.mean(science.features_err[idx2][sfil]) ** 2,
                                    cov_err_cf=-np.mean(control.features_err[idx2][cfil]) ** 2,
                                    var_err_sc=np.mean(science.features_err[idx1][sfil]) ** 2 +
                                    np.mean(science.features_err[idx2][sfil]) ** 2,
                                    var_err_cf=np.mean(control.features_err[idx1][cfil]) ** 2 +
                                    np.mean(control.features_err[idx2][cfil]) ** 2)

        # Get slopes
        # slope_ols = 1 - beta_ols * (a_hk - 1)
        # slope_bces = 1 - beta_bces * (a_hk - 1)
        slope_lines = 1 - beta_lines * (a_hk - 1)

        # Print results
        # print(science.features_names[idx4])
        # print("OLS:", 1 - beta_ols * (a_hk - 1))
        # print("BCES:", 1 - beta_bces * (a_hk - 1))
        # print("LINES:", 1 - beta_lines * (a_hk - 1))
        # print()

        # Just return for J band
        if science.features_names[idx4] == "J":
            return slope_lines, n


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with Pool() as pool:
        mp = pool.starmap(get_slope, zip(glon_range, glat_range, repeat(science.lon), repeat(science.lat), repeat(0.3)))

    # Unpack results
    slope, nsources = list(zip(*mp))

# Convert results to arrays
slope, nsources = np.array(slope), np.array(nsources, dtype=float)
nsources[nsources < 1] = np.nan

# plt.imshow(np.array(slope).reshape(grid_shape), cmap="brg", vmin=2.45, vmax=2.55, interpolation="nearest")
fig = plt.figure(figsize=[10, 6])
grid = GridSpec(ncols=2, nrows=2, bottom=0.05, top=0.95, left=0.05, right=0.95, hspace=0.1, wspace=0.1,
                width_ratios=[1, 0.02])
# Add axes
ax1 = plt.subplot(grid[0], projection=wcsaxes.WCS(header=header))
cax1 = plt.subplot(grid[1])
ax2 = plt.subplot(grid[2], projection=wcsaxes.WCS(header=header))
cax2 = plt.subplot(grid[3])

# Plot data
im1 = ax1.imshow(slope.reshape(grid_shape), cmap=cmap1, interpolation="nearest", origin="lower", vmin=2.45, vmax=2.55)
plt.colorbar(im1, cax=cax1, ticks=MultipleLocator(0.02), label="Slope")

# Plot number of sources
# ax2.scatter(glon_range, glat_range, c=slope, lw=0, marker="s", s=300, cmap=cmap, vmin=2.45, vmax=2.55)
im2 = ax2.imshow(nsources.reshape(grid_shape), cmap=cmap2, origin="lower", interpolation="nearest")
plt.colorbar(im2, cax=cax2, label="#")

plt.show()
