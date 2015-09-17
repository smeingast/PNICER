# ----------------------------------------------------------------------
# Import stuff
import warnings
import brewer2mpl
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from pnicer import Magnitudes
from matplotlib.pyplot import GridSpec
from matplotlib.ticker import MultipleLocator


# ----------------------------------------------------------------------
# Define file paths
science_path = "/Users/Antares/Dropbox/Data/Orion/VISION/Catalog/VISION_+_Spitzer_s_noYSO.fits"
control_path = "/Users/Antares/Dropbox/Data/Orion/VISION/Catalog/VISION_CF+_Spitzer_s.fits"

# Load colormap
cmap = brewer2mpl.get_map("YlOrRd", "Sequential", number=9, reverse=False).get_mpl_colormap(N=10, gamma=0.7)


# ----------------------------------------------------------------------
# Helper function
# def get_distance(sl, inter, x0, y0):
#     return np.abs(sl * x0 - y0 + inter) / np.sqrt(sl**2 + 1)


# ----------------------------------------------------------------------
# Load data
skip = 1
science_dummy = fits.open(science_path)[1].data
control_dummy = fits.open(control_path)[1].data

# Coordinates
science_glon = science_dummy["GLON"][::skip]
science_glat = science_dummy["GLAT"][::skip]
control_glon = control_dummy["GLON"]
control_glat = control_dummy["GLAT"]

# Definitions
features_names = ["J", "H", "Ks", "IRAC1", "IRAC2"]
errors_names = ["J_err", "H_err", "Ks_err", "IRAC1_err", "IRAC2_err"]
features_extinction = [2.5, 1.55, 1.0, 0.636, 0.54]

# Photometry
science_data = [science_dummy[n][::skip] for n in features_names]
science_error = [science_dummy[n][::skip] for n in errors_names]
control_data = [control_dummy[n] for n in features_names]
control_error = [control_dummy[n] for n in errors_names]


# # ----------------------------------------------------------------------
# # Define source filters
# fil_jh_sc = np.isfinite(science_data[0]) & np.isfinite(science_data[1])
# fil_hks_sc = np.isfinite(science_data[1]) & np.isfinite(science_data[2])
# fil_ksi1_sc = np.isfinite(science_data[2]) & np.isfinite(science_data[3])
# fil_i1i2_sc = np.isfinite(science_data[3]) & np.isfinite(science_data[4])
# fil_all_sc = [fil_jh_sc, fil_hks_sc, fil_ksi1_sc, fil_i1i2_sc]
# fil_jh_cf = np.isfinite(control_data[0]) & np.isfinite(control_data[1])
# fil_hks_cf = np.isfinite(control_data[1]) & np.isfinite(control_data[2])
# fil_ksi1_cf = np.isfinite(control_data[2]) & np.isfinite(control_data[3])
# fil_i1i2_cf = np.isfinite(control_data[3]) & np.isfinite(control_data[4])
# fil_all_cf = [fil_jh_cf, fil_hks_cf, fil_ksi1_cf, fil_i1i2_cf]
#
# # ----------------------------------------------------------------------
# # Create figure
# fig = plt.figure(figsize=[10, 10])
# grid = GridSpec(ncols=3, nrows=3, bottom=0.05, top=0.90, left=0.05, right=0.95, hspace=0, wspace=0)
# cax = fig.add_axes([0.05, 0.91, 0.3, 0.02])
#
# plot_index = [8, 7, 6, 4, 3, 0]
# data_index = [[0, 1, 2], [0, 1, 2, 3], [0, 1, 3, 4], [1, 2, 3], [1, 2, 3, 4], [2, 3, 4]]
# for idx, pidx in zip(data_index, plot_index):
#
#     # Construct combined masks
#     smask = np.prod([np.isfinite(science_data[x]) for x in idx], axis=0, dtype=bool)
#     cmask = np.prod([np.isfinite(control_data[x]) for x in idx], axis=0, dtype=bool)
#
#     sglon, sglat = science_glon[smask], science_glat[smask]
#
#     # Get data
#     sdata = [science_data[x][smask] for x in idx]
#     serror = [science_error[x][smask] for x in idx]
#     cdata = [control_data[x][cmask] for x in idx]
#     cerror = [control_error[x][cmask] for x in idx]
#
#     # Get extinction and feature names
#     fext = [features_extinction[x] for x in idx]
#     fnames = [features_names[x] for x in idx]
#
#     # Initialize data
#     science = Magnitudes(mag=sdata, err=serror, extvec=fext, names=fnames, lon=sglon, lat=sglat)
#     control = Magnitudes(mag=cdata, err=cerror, extvec=fext, names=fnames)
#
#     # Get feature names
#     cnames = science.mag2color().features_names
#     name_y, name_x = cnames[0], cnames[-1]
#
#     # get slope in color space
#     dummy = science.mag2color().extvec.extvec
#     slope = dummy[0] / dummy[-1]
#
#     # print(name_x, name_y, pidx)
#     # print(dummy[-1], dummy[0])
#
#     # Calculate extinction
#     # pnicer = science.pnicer(control=control, add_colors=True)
#     pnicer = science.mag2color().pnicer(control=control.mag2color())
#     ext = pnicer.extinction
#     col0 = [pnicer.color0[0], pnicer.color0[-1]]
#     # nicer = science.nicer(control=control)
#     # ext = nicer.extinction
#     # col0 = [nicer.color0[0], nicer.color0[-1]]
#     # nicer.save_fits(path="/Users/Antares/Desktop/test.fits")
#
#     # Get average colors in extinction bins
#     step = 0.2
#     c1, c2, ak = [], [], []
#     colors = science.mag2color().features
#     for e in np.arange(-1, 15.01, step=step):
#
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             fil = (ext >= e) & (ext < e + step)
#
#             # We require at least 3 sources
#             if np.sum(fil) < 3:
#                 c1.append(np.nan)
#                 c2.append(np.nan)
#             else:
#                 # Get color averages
#                 c1.append(np.nanmedian(colors[-1][fil] - col0[-1][fil]))
#                 c2.append(np.nanmedian(colors[0][fil] - col0[0][fil]))
#             # Append extinction
#             ak.append(e)
#
#     # Add subplot
#     ax = plt.subplot(grid[pidx])
#
#     # Plot
#     im = ax.scatter(c1, c2, lw=1, s=40, alpha=1, c=ak, cmap=cmap, vmin=0, vmax=5)
#     x = np.arange(-1, 8, 0.5)
#     ax.plot(x, slope * x, color="black", lw=2, linestyle="dashed")
#
#     # Add colorbar
#     if pidx == 0:
#         cbar = plt.colorbar(im, cax=cax, ticks=MultipleLocator(1), label="$A_K$", orientation="horizontal")
#         cbar.ax.xaxis.set_ticks_position("top")
#         cbar.set_label("$A_K$ (mag)", labelpad=-50)
#         cbar.ax.minorticks_on()
#
#     # Limits
#     ax.set_xlim(-0.5, 2.5)
#     ax.set_ylim(-0.5, 4)
#
#     # Ticker
#     ax.xaxis.set_major_locator(MultipleLocator(1))
#     ax.xaxis.set_minor_locator(MultipleLocator(0.2))
#     ax.yaxis.set_major_locator(MultipleLocator(1))
#     ax.yaxis.set_minor_locator(MultipleLocator(0.2))
#
#     # Remove labels
#     if pidx < 6:
#         ax.axes.xaxis.set_ticklabels([])
#
#     if pidx in [4, 7, 8]:
#         ax.axes.yaxis.set_ticklabels([])
#
#     # set labels
#     if pidx == 8:
#         ax.set_xlabel("$E_{H-K_S}$")
#     if pidx == 7:
#         ax.set_xlabel("$E_{K_S - [3.6]}$")
#     if pidx == 6:
#         ax.set_xlabel("$E_{[3.6] - [4.5]}$")
#         ax.set_ylabel("$E_{J-H}$")
#     if pidx == 3:
#         ax.set_ylabel("$E_{H-K_S}$")
#     if pidx == 0:
#         ax.set_ylabel("$E_{K_S - [3.6]}$")
#
#     # plt.show()
#     print()
#     # exit()
#
# # Save
# plt.savefig("/Users/Antares/Dropbox/Projects/PNICER/Paper/Results/extinction_law.pdf", bbox_inches="tight")
# exit()

# ----------------------------------------------------------------------
# Filter only those with all 5 detections
# com = np.isfinite(science_data[0]) & np.isfinite(science_data[1]) & np.isfinite(science_data[2]) & \
#     np.isfinite(science_data[3]) & np.isfinite(science_data[4])
#
#  science_glon = science_glon[com]
# science_glat = science_glat[com]


# ----------------------------------------------------------------------
# Initialize data for all combinations
science = Magnitudes(mag=science_data, err=science_error, extvec=features_extinction, names=features_names)
control = Magnitudes(mag=control_data, err=control_error, extvec=features_extinction, names=features_names)
# Get extinction
# ext = science.pnicer(control=control, add_colors=True)
ext = science.mag2color().pnicer(control=control.mag2color())
# ext = science.nicer(control=control)


# Get average colors in extinction bins
step = 0.2
jh, hks, ksi1, i1i2, ak = [], [], [], [], []
for a in np.arange(-1, 15.01, step=step):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fil = (ext.extinction >= a) & (ext.extinction < a + step)

        # We require at least 3 sources
        if np.sum(fil) < 3:
            continue

        # Append extinction
        jh_avg = np.nanmean(science.features[0][fil] - science.features[1][fil] - ext.color0[0][fil])
        hk_avg = np.nanmean(science.features[1][fil] - science.features[2][fil] - ext.color0[1][fil])
        ksi1_avg = np.nanmean(science.features[2][fil] - science.features[3][fil] - ext.color0[2][fil])
        i1i2_avg = np.nanmean(science.features[3][fil] - science.features[4][fil] - ext.color0[3][fil])

        jh.append(jh_avg)
        hks.append(hk_avg)
        ksi1.append(ksi1_avg)
        i1i2.append(i1i2_avg)
        ak.append(a)

# Get slopes
slopes = science.mag2color().extvec.extvec


# ----------------------------------------------------------------------
# Create figure
fig = plt.figure(figsize=[10, 10])
grid = GridSpec(ncols=3, nrows=3, bottom=0.05, top=0.95, left=0.05, right=0.95, hspace=0, wspace=0)


# Add axes
ax_hks_jh = plt.subplot(grid[8])
ax_ksi1_jh = plt.subplot(grid[7])
ax_ksi1_hks = plt.subplot(grid[4])
ax_i1i2_jh = plt.subplot(grid[6])
ax_i1i2_hks = plt.subplot(grid[3])
ax_i1i2_ksi1 = plt.subplot(grid[0])

# Plot
s, alpha = 40, 1
ax_hks_jh.scatter(hks, jh, lw=1, s=s, alpha=alpha, c=ak, cmap=cmap, vmin=0, vmax=5)
ax_ksi1_jh.scatter(ksi1, jh, lw=1, s=s, alpha=alpha, c=ak, cmap=cmap, vmin=0, vmax=5)
ax_ksi1_hks.scatter(ksi1, hks, lw=1, s=s, alpha=alpha, c=ak, cmap=cmap, vmin=0, vmax=5)
ax_i1i2_jh.scatter(i1i2, jh, lw=1, s=s, alpha=alpha, c=ak, cmap=cmap, vmin=0, vmax=5)
ax_i1i2_hks.scatter(i1i2, hks, lw=1, s=s, alpha=alpha, c=ak, cmap=cmap, vmin=0, vmax=5)
ax_i1i2_ksi1.scatter(i1i2, ksi1, lw=1, s=s, alpha=alpha, c=ak, cmap=cmap, vmin=0, vmax=5)

# Plot slopes
x = np.arange(-10, 10, 0.5)
ax_hks_jh.plot(x, slopes[0]/slopes[1] * x, lw=2, ls="dashed", color="black")
ax_hks_jh.set_xlabel("$E_{H-K_S}$")
ax_ksi1_jh.plot(x, slopes[0]/slopes[2] * x, lw=2, ls="dashed", color="black")
ax_ksi1_jh.set_xlabel("$E_{K_S-[3.6]}$")
ax_ksi1_hks.plot(x, slopes[1]/slopes[2] * x, lw=2, ls="dashed", color="black")
ax_i1i2_jh.plot(x, slopes[0]/slopes[3] * x, lw=2, ls="dashed", color="black")
ax_i1i2_jh.set_xlabel("$E_{[4.5]-[3.6]}$")
ax_i1i2_jh.set_ylabel("$E_{J-H}$")
ax_i1i2_hks.plot(x, slopes[1]/slopes[3] * x, lw=2, ls="dashed", color="black")
ax_i1i2_hks.set_ylabel("$E_{H-K_S}$")
ax_i1i2_ksi1.plot(x, slopes[2]/slopes[3] * x, lw=2, ls="dashed", color="black")
ax_i1i2_ksi1.set_ylabel("$E_{K_S-[3.6]}$")

# Adjust axes properties
ax_all = [ax_i1i2_ksi1, ax_i1i2_hks, ax_ksi1_hks, ax_i1i2_jh, ax_ksi1_jh, ax_hks_jh]
for idx in range(len(ax_all)):
    # Limits
    ax_all[idx].set_xlim(-0.5, 4)
    ax_all[idx].set_ylim(-0.5, 6)

    # Ticker
    ax_all[idx].xaxis.set_major_locator(MultipleLocator(1))
    ax_all[idx].xaxis.set_minor_locator(MultipleLocator(0.2))
    ax_all[idx].yaxis.set_major_locator(MultipleLocator(1))
    ax_all[idx].yaxis.set_minor_locator(MultipleLocator(0.2))

    if idx < 3:
        ax_all[idx].axes.xaxis.set_ticklabels([])

    if idx in [2, 4, 5]:
        ax_all[idx].axes.yaxis.set_ticklabels([])


# Save
plt.savefig("/Users/Antares/Dropbox/Projects/PNICER/Paper/Results/extinction_law_binning.pdf", bbox_inches="tight")