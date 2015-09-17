# ----------------------------------------------------------------------
# Import stuff
import wcsaxes
import warnings
import brewer2mpl
# import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from itertools import repeat
from multiprocessing import Pool
from matplotlib.pyplot import GridSpec
from matplotlib.ticker import MultipleLocator
from MyFunctions import distance_on_unit_sphere
from helper import *


# ----------------------------------------------------------------------
# Load colorbrewer colormap
cmap0 = brewer2mpl.get_map("Greys", "Sequential", number=9, reverse=False).get_mpl_colormap(N=500, gamma=1)
cmap1 = brewer2mpl.get_map("Spectral", "Diverging", number=11, reverse=True).get_mpl_colormap(N=10, gamma=1)
cmap2 = brewer2mpl.get_map("YlGnBu", "Sequential", number=9, reverse=True).get_mpl_colormap(N=10, gamma=1)


# ----------------------------------------------------------------------
# Intialize PNICER
science, control = pnicer_ini(skip_science=1, skip_control=1, n_features=5, color=False)
science_color, control_color = science.mag2color(), control.mag2color()


# ----------------------------------------------------------------------
# Get grid
pixsize = 2 / 60
header, all_glon, all_glat = science.build_wcs_grid(frame="galactic", pixsize=pixsize)
grid_shape = all_glat.shape
glon_range, glat_range = all_glon.ravel(), all_glat.ravel()


# ----------------------------------------------------------------------
# Initial guess for extinction
pnicer = science_color.pnicer(control=control_color)
ext = pnicer.extinction
emap = pnicer.build_map(bandwidth=pixsize * 2, metric="epanechnikov")


# ----------------------------------------------------------------------
# Define parameter range for filtering
ajr = (2.4, 2.7)
ahr = (1.4, 1.7)
a1r = (0.5, 0.7)
a2r = (0.3, 0.6)

# Define filter
for data in [science_color, control_color]:
    # Define base photometric filter common to science and control field
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fil = (data.features[0] + 1 < (ajr[0] - ahr[0]) / (ahr[0] - 1) * (data.features[1] + 0.7)) & \
              (data.features[0] + 1 > (ajr[1] - ahr[1]) / (ahr[1] - 1) * (data.features[1] + 0.7)) & \
              (data.features[0] + 1 > (ajr[0] - ahr[0]) / (1 - a1r[0]) * (data.features[2] + 0.5)) & \
              (data.features[0] + 1 < (ajr[1] - ahr[1]) / (1 - a1r[1]) * (data.features[2] + 0.5)) & \
              (data.features[0] + 1 > (ajr[0] - ahr[0]) / (a1r[0] - a2r[0]) * (data.features[3] + 0.2)) & \
              (data.features[0] + 1 < (ajr[1] - ahr[1]) / (a1r[1] - a2r[1]) * (data.features[3] + 0.2)) & \
              (np.nanmax(np.array(data.features_err), axis=0) < 0.1)

        if data == science_color:
            # sfil = fil.copy() & (dis < maxdis)
            fil_science = fil & (ext > 0.1)
        else:
            fil_control = fil.copy()


# ----------------------------------------------------------------------
# Loop over bins and get extinction law
def get_slope(glon_pix, glat_pix, glon_all, glat_all, maxdis):

    # Calculate distance to all other sources from current grid point
    dis = np.degrees(distance_on_unit_sphere(ra1=np.radians(glon_all), dec1=np.radians(glat_all),
                                             ra2=np.radians(glon_pix), dec2=np.radians(glat_pix)))

    # Additional distance filtering for control field
    sfil = fil_science.copy() & (dis < maxdis)
    cfil = fil_control.copy()

    # Save number of sources within filter limits
    n = np.sum(sfil)

    # Skip if there are too few
    if np.sum(sfil) < 100:
        return np.nan, np.nan

    # Get shortcut for data
    xdata_science = science.features[1][sfil] - science.features[2][sfil]
    ydata_science = science.features[0][sfil] - science.features[1][sfil]
    xdata_control = control.features[1][cfil] - control.features[2][cfil]
    ydata_control = control.features[0][cfil] - control.features[1][cfil]

    # Get ordinary least squares slope
    # beta_ols = get_beta_ols(xj=xdata_science, yj=ydata_science)

    # Get BCES slope
    # beta_bces = get_beta_bces(x_sc=xdata_science, y_sc=ydata_science,
    #                           cov_err_sc=-np.mean(science.features_err[1][sfil]) ** 2,
    #                           var_err_sc=np.mean(science.features_err[1][sfil]) ** 2 +
    #                           np.mean(science.features_err[2][sfil]) ** 2)

    # Get LINES slope
    beta_lines = get_beta_lines(x_sc=xdata_science, y_sc=ydata_science,
                                x_cf=xdata_control, y_cf=ydata_control,
                                cov_err_sc=-np.mean(science.features_err[1][sfil]) ** 2,
                                cov_err_cf=-np.mean(control.features_err[1][cfil]) ** 2,
                                var_err_sc=np.mean(science.features_err[1][sfil]) ** 2 +
                                np.mean(science.features_err[2][sfil]) ** 2,
                                var_err_cf=np.mean(control.features_err[1][cfil]) ** 2 +
                                np.mean(control.features_err[2][cfil]) ** 2)

    # Just return for J band
    return beta_lines, n



if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with Pool(4) as pool:
            mp = pool.starmap(get_slope, zip(glon_range, glat_range, repeat(science.lon),
                                             repeat(science.lat), repeat(0.2)))
else:
    mp = 0
    exit()

# Unpack results
slope, nsources = list(zip(*mp))

# Convert results to arrays
slope, nsources = np.array(slope), np.array(nsources, dtype=float)

# plt.imshow(np.array(slope).reshape(grid_shape), cmap="brg", vmin=2.45, vmax=2.55, interpolation="nearest")
fig = plt.figure(figsize=[11, 10])
grid = GridSpec(ncols=2, nrows=3, bottom=0.05, top=0.95, left=0.05, right=0.95, hspace=0.1, wspace=0.1,
                width_ratios=[1, 0.02])
# Add axes
ax0 = plt.subplot(grid[0], projection=wcsaxes.WCS(header=emap.fits_header))
cax0 = plt.subplot(grid[1])
ax1 = plt.subplot(grid[2], projection=wcsaxes.WCS(header=header))
cax1 = plt.subplot(grid[3])
ax2 = plt.subplot(grid[4], projection=wcsaxes.WCS(header=header))
cax2 = plt.subplot(grid[5])

# Plot extinction map
im0 = ax0.imshow(emap.map, interpolation="nearest", origin="lower", vmin=0, vmax=2, cmap=cmap0)
plt.colorbar(im0, cax=cax0, ticks=MultipleLocator(0.25), label="$A_K$")

# Plot slope
im1 = ax1.imshow(slope.reshape(grid_shape), cmap=cmap1, interpolation="nearest", origin="lower", vmin=1.68, vmax=1.78)
plt.colorbar(im1, cax=cax1, ticks=MultipleLocator(0.02), label="Slope")

# Plot number of sources
# ax2.scatter(glon_range, glat_range, c=slope, lw=0, marker="s", s=300, cmap=cmap, vmin=2.45, vmax=2.55)
im2 = ax2.imshow(nsources.reshape(grid_shape), cmap=cmap2, origin="lower", interpolation="nearest")
plt.colorbar(im2, cax=cax2, label="#")

# Draw IRAC1 coverage
irac1_coverage_path = "/Users/Antares/Dropbox/Data/Orion/Other/Spitzer_Orion_IRAC1_coverage_s.fits"
irac1_coverage = fits.open(irac1_coverage_path)[0].data
irac1_coverage_header = fits.open(irac1_coverage_path)[0].header

for ax in [ax0, ax1, ax2]:
    ax.set_xlim(0, grid_shape[1])
    ax.set_ylim(0, grid_shape[0])
    ax.contour(irac1_coverage, levels=[0, 1], transform=ax.get_transform(wcsaxes.WCS(irac1_coverage_header)),
               colors="black")

# Save figure
results_path = "/Users/Antares/Dropbox/Projects/PNICER/Paper/Results/"
plt.savefig(results_path + "extinction_law_spatial.pdf", bbox_inches="tight", dpi=300)
