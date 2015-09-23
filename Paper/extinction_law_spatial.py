# ----------------------------------------------------------------------
# Import stuff
import wcsaxes
import warnings
import brewer2mpl
import matplotlib.pyplot as plt

from itertools import repeat
from multiprocessing import Pool
from matplotlib.pyplot import GridSpec
from matplotlib.ticker import MultipleLocator
from MyFunctions import distance_on_unit_sphere
from helper import *


# ----------------------------------------------------------------------
# Define file paths
science_path = "/Users/Antares/Dropbox/Data/Orion/VISION/Catalog/VISION_+_Spitzer_s_noYSO.fits"
control_path = "/Users/Antares/Dropbox/Data/Orion/VISION/Catalog/VISION_CF+_Spitzer_s.fits"
results_path = "/Users/Antares/Dropbox/Projects/PNICER/Paper/Results/"


# ----------------------------------------------------------------------
# Load colorbrewer colormap
cmap0 = brewer2mpl.get_map("Greys", "Sequential", number=9, reverse=False).get_mpl_colormap(N=500, gamma=1)
cmap1 = brewer2mpl.get_map("Spectral", "Diverging", number=11, reverse=True).get_mpl_colormap(N=10, gamma=1)
cmap2 = brewer2mpl.get_map("YlGnBu", "Sequential", number=9, reverse=True).get_mpl_colormap(N=10, gamma=1)
viridis = get_viridis()

# ----------------------------------------------------------------------
# Intialize PNICER
science, control = pnicer_ini(skip_science=1, skip_control=1, n_features=3, color=False)
science_color, control_color = science.mag2color(), control.mag2color()

# Additionally load galaxy classifier
class_sex_science = fits.open(science_path)[1].data["class_sex"]
class_sex_control = fits.open(control_path)[1].data["class_sex"]
class_cog_science = fits.open(science_path)[1].data["class_cog"]
class_cog_control = fits.open(control_path)[1].data["class_cog"]


# ----------------------------------------------------------------------
# Make pre-selection of data
pnicer = science_color.pnicer(control=control_color)
ext = pnicer.extinction
# ext = np.full_like(science.features[0], fill_value=1.0)
for d in [science.dict, control.dict]:
    # Define filter
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fil = (d["J"] > 0) & (d["H"] > 0) & (d["Ks"] < 16) & (d["Ks_err"] < 0.1)

        # Test no filtering
        # fil = data.combined_mask

        if d == science.dict:
            sfil = fil & (ext > 0.3) & (class_sex_science > 0.8) & (class_cog_science == 1)
            # sfil = fil.copy()
        else:
            cfil = fil & (class_sex_control > 0.8) & (class_cog_control == 1)


# ----------------------------------------------------------------------
# Re-initialize with filtered data
# noinspection PyUnboundLocalVariable
science, control = pnicer_ini(skip_science=1, skip_control=1, n_features=3, color=False, sfil=sfil, cfil=cfil)
science_color, control_color = science.mag2color(), control.mag2color()

# Additionally load galaxy classifier
class_sex_science = fits.open(science_path)[1].data["class_sex"][sfil]
class_sex_control = fits.open(control_path)[1].data["class_sex"][cfil]
class_cog_science = fits.open(science_path)[1].data["class_cog"][sfil]
class_cog_control = fits.open(control_path)[1].data["class_cog"][cfil]


# ----------------------------------------------------------------------
# Get slopes of CCDs
base_idx = (1, 2)
fit_idx, betas, err = science.get_extinction_law(base_index=base_idx, method="OLS", control=control)


# ----------------------------------------------------------------------
# Do iterative clipping
fil_idx = []
for fidx, beta in zip(fit_idx, betas):

    mask = np.arange(science.n_data)
    idx = np.arange(science.n_data)

    # Get intercept
    for _ in range(1):

        # Shortcut for data
        xdata_science = science.features[base_idx[0]][mask] - science.features[base_idx[1]][mask]
        ydata_science = science.features[base_idx[1]][mask] - science.features[fidx][mask]


        # Get intercept of linear fir throguh median
        ic = np.median(ydata_science) - beta * np.median(xdata_science)

        # Get orthogonal distance
        odis = np.abs(beta * xdata_science - ydata_science + ic) / np.sqrt(beta ** 2 + 1)

        # 3 sig filter
        mask = odis < 3 * np.std(odis)
        idx = idx[mask]

    # Now append the bad ones
    fil_idx.extend(np.setdiff1d(np.arange(science.n_data), idx))

# Determine which sources to filter
uidx = np.unique(np.array(fil_idx))
# And finally get the good index
good_index = np.setdiff1d(np.arange(science.n_data), uidx)


# ----------------------------------------------------------------------
# Final initialization with cleaned data
science_data_good = [science.features[i][good_index] for i in range(science.n_features)]
science_err_good = [science.features_err[i][good_index] for i in range(science.n_features)]

science = Magnitudes(mag=science_data_good, err=science_err_good, extvec=science.extvec.extvec,
                     lon=science.lon[good_index], lat=science.lat[good_index])
sciencecolor = science.mag2color()


# ----------------------------------------------------------------------
# Get grid
pixsize = 10 / 60
header, all_glon, all_glat = science.build_wcs_grid(frame="galactic", pixsize=pixsize)
grid_shape = all_glat.shape
glon_range, glat_range = all_glon.ravel(), all_glat.ravel()


# ----------------------------------------------------------------------
# Build preliminary extinction map
emap = pnicer.build_map(bandwidth=pixsize * 2, metric="epanechnikov")


# ----------------------------------------------------------------------
# Define function tog et extinction law
def get_slope(glon_pix, glat_pix, glon_all, glat_all, maxdis):

    # Calculate distance to all other sources from current grid point
    dis = np.degrees(distance_on_unit_sphere(ra1=np.radians(glon_all), dec1=np.radians(glat_all),
                                             ra2=np.radians(glon_pix), dec2=np.radians(glat_pix)))

    # Additional distance filtering for science field
    dfil = dis < maxdis

    # Save number of sources within filter limits
    n = np.sum(dfil)

    # Skip if there are too few
    if np.sum(dfil) < 100:
        return np.nan, np.nan, np.nan, np.nan

    # Initialize with current distance filter
    csdata = [science.features[i][dfil] for i in range(science.n_features)]
    cserr = [science.features_err[i][dfil] for i in range(science.n_features)]
    sc = Magnitudes(mag=csdata, err=cserr, extvec=science.extvec.extvec)
    f, b, e = sc.get_extinction_law(base_index=(1, 2), method="LINES", control=control)

    # Just return for J band
    return f[0], b[0], e[0], n


# ----------------------------------------------------------------------
# Run parallel
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with Pool(6) as pool:
        mp = pool.starmap(get_slope, zip(glon_range, glat_range, repeat(science.lon),
                                         repeat(science.lat), repeat(20 / 60)))

# Unpack results
idx, slope, slope_err, nsources = list(zip(*mp))

# Convert results to arrays
slope, slope_err, nsources = np.array(slope), np.array(slope_err), np.array(nsources, dtype=float)


# ----------------------------------------------------------------------
# Plot results
# plt.imshow(np.array(slope).reshape(grid_shape), cmap="brg", vmin=2.45, vmax=2.55, interpolation="nearest")
fig = plt.figure(figsize=[11, 13])
grid = GridSpec(ncols=2, nrows=4, bottom=0.05, top=0.95, left=0.05, right=0.95, hspace=0.1, wspace=0.1,
                width_ratios=[1, 0.02])
# Add axes
ax0 = plt.subplot(grid[0], projection=wcsaxes.WCS(header=emap.fits_header))
cax0 = plt.subplot(grid[1])
ax1 = plt.subplot(grid[2], projection=wcsaxes.WCS(header=header))
cax1 = plt.subplot(grid[3])
ax2 = plt.subplot(grid[4], projection=wcsaxes.WCS(header=header))
cax2 = plt.subplot(grid[5])
ax3 = plt.subplot(grid[6], projection=wcsaxes.WCS(header=header))
cax3 = plt.subplot(grid[7])

# Plot extinction map
im0 = ax0.imshow(emap.map, interpolation="nearest", origin="lower", vmin=0, vmax=2, cmap=cmap0)
plt.colorbar(im0, cax=cax0, ticks=MultipleLocator(0.25), label="$A_K$")

# Plot slope
im1 = ax1.imshow(slope.reshape(grid_shape), cmap=cmap1, interpolation="nearest", origin="lower", vmin=-2.82, vmax=-2.62)
plt.colorbar(im1, cax=cax1, ticks=MultipleLocator(0.02), label=r"$\beta$")

# Plot slope error
im2 = ax2.imshow(slope_err.reshape(grid_shape), cmap=viridis, interpolation="nearest", origin="lower", vmin=0, vmax=0.1)
plt.colorbar(im2, cax=cax2, label=r"$\sigma_{\beta}$")

# Plot number of sources
# ax2.scatter(glon_range, glat_range, c=slope, lw=0, marker="s", s=300, cmap=cmap, vmin=2.45, vmax=2.55)
im3 = ax3.imshow(nsources.reshape(grid_shape), cmap=viridis, interpolation="nearest",  origin="lower")
plt.colorbar(im3, cax=cax3, label="#")

# Draw IRAC1 coverage
irac1_coverage_path = "/Users/Antares/Dropbox/Data/Orion/Other/Spitzer_Orion_IRAC1_coverage_s.fits"
irac1_coverage = fits.open(irac1_coverage_path)[0].data
irac1_coverage_header = fits.open(irac1_coverage_path)[0].header

# Adjust axes
for ax in [ax0, ax1, ax2]:
    ax.set_xlim(0, grid_shape[1])
    ax.set_ylim(0, grid_shape[0])
    ax.contour(irac1_coverage, levels=[0, 1], transform=ax.get_transform(wcsaxes.WCS(irac1_coverage_header)),
               colors="black")

# Save figure
plt.savefig(results_path + "extinction_law_spatial.pdf", bbox_inches="tight", dpi=300)
