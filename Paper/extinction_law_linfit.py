# ----------------------------------------------------------------------
# Import stuff
import warnings
import brewer2mpl
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from matplotlib.pyplot import GridSpec
from matplotlib.ticker import MultipleLocator
from MyFunctions import point_density
from helper import pnicer_ini
from pnicer import Magnitudes

# ----------------------------------------------------------------------
# Define file paths
science_path = "/Users/Antares/Dropbox/Data/Orion/VISION/Catalog/VISION_+_Spitzer_s_noYSO.fits"
control_path = "/Users/Antares/Dropbox/Data/Orion/VISION/Catalog/VISION_CF+_Spitzer_s.fits"
results_path = "/Users/Antares/Dropbox/Projects/PNICER/Paper/Results/"


# ----------------------------------------------------------------------
# Load colorbrewer colormap
cmap1 = brewer2mpl.get_map("YlGnBu", "Sequential", number=9, reverse=True).get_mpl_colormap(N=11, gamma=1)
cmap2 = brewer2mpl.get_map("YlOrRd", "Sequential", number=9, reverse=True).get_mpl_colormap(N=11, gamma=1)


# ----------------------------------------------------------------------
# Intialize PNICER
science_all, control_all = pnicer_ini(skip_science=1, skip_control=1, n_features=5, color=False)
science_color_all, control_color_all = science_all.mag2color(), control_all.mag2color()

# Additionally load galaxy classifier
class_sex_science = fits.open(science_path)[1].data["class_sex"]
class_sex_control = fits.open(control_path)[1].data["class_sex"]
class_cog_science = fits.open(science_path)[1].data["class_cog"]
class_cog_control = fits.open(control_path)[1].data["class_cog"]


# ----------------------------------------------------------------------
# Make pre-selection of data
ext = science_color_all.pnicer(control=control_color_all).extinction
# ext.save_fits(path="/Users/Antares/Desktop/test.fits")
# ext = science_all.nicer(control=control_all).extinction
# ext = np.full_like(science.features[0], fill_value=1.0)
for d in [science_all.dict, control_all.dict]:
    # Define filter
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fil = (d["J"] > 0) & (d["H"] > 0) & (d["Ks"] < 15) & (d["IRAC1"] > 0) & (d["IRAC2"] > 0) & \
              (d["IRAC1_err"] < 0.1) & (d["Ks_err"] < 0.1)

        # Test no filtering
        # fil = data.combined_mask

        if d == science_all.dict:
            sfil = fil & (ext > 0.3) & (class_sex_science > 0.8) & (class_cog_science == 1)
            # sfil = fil.copy()
        else:
            cfil = fil.copy() & (class_sex_control > 0.8) & (class_cog_control == 1)


# ----------------------------------------------------------------------
# Re-initialize with filtered data
# noinspection PyUnboundLocalVariable
science, control = pnicer_ini(skip_science=1, skip_control=1, n_features=5, color=False, sfil=sfil, cfil=cfil)
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
        dis = np.abs(beta * xdata_science - ydata_science + ic) / np.sqrt(beta ** 2 + 1)

        # 3 sig filter
        mask = dis < 3 * np.std(dis)
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

science_good = Magnitudes(mag=science_data_good, err=science_err_good, extvec=science.extvec.extvec)
science_good_color = science_good.mag2color()


# ----------------------------------------------------------------------
# Get slopes of CCDs from cleaned data
fit_idx, betas, betas_err = science_good.get_extinction_law(base_index=base_idx, method="LINES", control=control)


# # ----------------------------------------------------------------------
# # Plot pre-selection of data
# fig1 = plt.figure(figsize=[14, 6.22])
# grid1 = GridSpec(ncols=3, nrows=3, bottom=0.05, top=0.95, left=0.05, right=0.45, hspace=0, wspace=0)
# grid2 = GridSpec(ncols=3, nrows=3, bottom=0.05, top=0.95, left=0.50, right=0.90, hspace=0, wspace=0)
#
# plot_index = [8, 7, 6, 4, 3, 0]
# data_index = [[1, 0], [2, 0], [3, 0], [2, 1], [3, 1], [3, 2]]
# for (idx1, idx2), pidx in zip(data_index, plot_index):
#
#     # Add axes
#     ax1 = plt.subplot(grid1[pidx])
#     ax2 = plt.subplot(grid2[pidx])
#
#     # Get source densities in CCDs
#     # dens_science = point_density(science_color.features[idx1], science_color.features[idx2], xsize=0.05, ysize=0.05)
#     dens_science = point_density(science_good_color.features[idx1], science_good_color.features[idx2], xsize=0.05, ysize=0.05)
#     dens_control = point_density(control_color.features[idx1], control_color.features[idx2], xsize=0.05, ysize=0.05)
#
#     # Plot all data for science field
#     ax1.scatter(science_color_all.features[idx1], science_color_all.features[idx2],
#                 lw=0, s=5, color="grey", alpha=0.01)
#     # Plot filtered data for science field
#     # ax1.scatter(science_color.features[idx1], science_color.features[idx2],
#     #             lw=1, s=7, facecolor="none", edgecolor="black", alpha=1)
#     ax1.scatter(science_good_color.features[idx1], science_good_color.features[idx2],
#                 lw=0, s=3, alpha=0.5, c=dens_science, cmap=cmap2)
#
#     # Plot all data for control field
#     ax2.scatter(control_color_all.features[idx1], control_color_all.features[idx2], lw=0, s=5, color="grey", alpha=0.01)
#     # Plot filtered data for control field
#     ax2.scatter(control_color.features[idx1], control_color.features[idx2],
#                 lw=0, s=2, alpha=0.5, c=dens_control, cmap=cmap1)
#
#     # Adjust axes
#     for ax in [ax1, ax2]:
#
#         # limits
#         ax.set_xlim(-0.7, 2.8)
#         ax.set_ylim(-0.2, 3.3)
#
#         # Force aspect ratio
#         # ax.set_aspect(1)
#
#         # Ticker
#         ax.xaxis.set_major_locator(MultipleLocator(1))
#         ax.xaxis.set_minor_locator(MultipleLocator(0.2))
#         ax.yaxis.set_major_locator(MultipleLocator(1))
#         ax.yaxis.set_minor_locator(MultipleLocator(0.2))
#         # Remove labels
#         if pidx < 6:
#             ax.axes.xaxis.set_ticklabels([])
#         if pidx in [4, 7, 8]:
#             ax.axes.yaxis.set_ticklabels([])
#         # set labels
#         if pidx == 8:
#             ax.set_xlabel("$H-K_S$")
#         if pidx == 7:
#             ax.set_xlabel("$K_S - [3.6]$")
#         if pidx == 6:
#             ax.set_xlabel("$[3.6] - [4.5]$")
#             ax.set_ylabel("$J-H$")
#         if pidx == 3:
#             ax.set_ylabel("$H-K_S$")
#         if pidx == 0:
#             ax.set_ylabel("$K_S - [3.6]$")
#
# # Save figure
# plt.savefig(results_path + "extinction_law_linfit_sources.png", bbox_inches="tight", dpi=300)
# plt.close()


# ----------------------------------------------------------------------
# Plot CCDs for fitting and actually do the fit...
fig2 = plt.figure(figsize=[12, 4])
grid = GridSpec(ncols=len(fit_idx), nrows=1, bottom=0.05, top=0.95, left=0.05, right=0.95, hspace=0.1, wspace=0.1,
                width_ratios=[0.5, 1, 1])

extvec_err = [0.15, 0.08, 0, 0.06, 0.08]
for beta, berr, idx, pidx in zip(betas, betas_err, fit_idx, range(len(fit_idx))):

    extvec = science_good.extvec.extvec

    # Get shortcut for data
    xdata_science = science.features[base_idx[0]] - science.features[base_idx[1]]
    ydata_science = science.features[base_idx[1]] - science.features[idx]
    xdata_science_good = science_good.features[base_idx[0]] - science_good.features[base_idx[1]]
    ydata_science_good = science_good.features[base_idx[1]] - science_good.features[idx]

    # Get extinction for current band
    ext = science_good.extvec.extvec[base_idx[1]] - beta * (extvec[base_idx[0]] - extvec[base_idx[1]])

    # Calculate error
    exterr = np.sqrt(((extvec[base_idx[0]] - extvec[base_idx[1]]) * berr) ** 2 +
                     (beta * extvec_err[base_idx[0]]) ** 2 +
                     (beta * extvec_err[base_idx[1]]) ** 2)

    # Print
    print(science.features_names[idx] + ":", ext, exterr)
    print(beta, berr)

    # Add subplot
    ax = plt.subplot(grid[pidx])

    # Get point density
    dens = point_density(xdata=xdata_science_good, ydata=ydata_science_good, xsize=0.1, ysize=0.1)

    # Plot data
    # ax.scatter(xdata_science, ydata_science, lw=1, s=8, alpha=1, facecolors="none", edgecolor="black")
    ax.scatter(xdata_science, ydata_science, lw=1, s=8, alpha=0.25, facecolors="black", edgecolor="none")
    ax.scatter(xdata_science_good, ydata_science_good, lw=0, s=11, alpha=0.8, c=dens, cmap=cmap2)

    # Plot slope and make it pass through median
    ic = np.median(ydata_science) - beta * np.median(xdata_science)
    ax.plot(np.arange(-1, 5, 1), beta * np.arange(-1, 5, 1) + ic, color="black", lw=2, linestyle="dashed")

    # Limits
    if pidx > 0:
        ax.set_xlim(-0.1, 2.6)
        ax.set_ylim(-0.1, 2.6)
    else:
        ax.set_xlim(-0.1, 2.45)
        ax.set_ylim(-5, 0.1)

    # Labels
    if pidx == 0:
        ax.set_ylabel("$" + science.features_names[base_idx[1]] + "-" + science.features_names[idx] + "$")
    if pidx == 1:
        ax.set_ylabel("$" + science.features_names[base_idx[1]] + "- [3.6]$")
    if pidx == 2:
        ax.set_ylabel("$" + science.features_names[base_idx[1]] + "- [4.5]$")
    # ax.set_xlabel("$" + science.features_names[base_idx[0]] + "-" + science.features_names[base_idx[1]] + "$")
    ax.set_xlabel("$" + science.features_names[base_idx[0]] + "-" + science.features_names[base_idx[1]] + "$")

    # Ticker
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.2))

    # Force aspect ratio
    ax.set_aspect(1)


# ----------------------------------------------------------------------
# Save figure
plt.savefig(results_path + "extinction_law_linfit_fit.png", bbox_inches="tight", dpi=300)
plt.close()
