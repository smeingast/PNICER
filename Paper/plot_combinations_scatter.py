from __future__ import absolute_import, division, print_function


# ----------------------------------------------------------------------
# Import stuff
import brewer2mpl
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from astropy.io import fits
from pnicer import Magnitudes
from matplotlib.pyplot import GridSpec
from matplotlib.ticker import MultipleLocator
from itertools import combinations
from MyFunctions import point_density

# ----------------------------------------------------------------------
# Change defaults
matplotlib.rcParams.update({'font.size': 13})


# ----------------------------------------------------------------------
# Define file paths
science_path = "/Users/Antares/Dropbox/Data/Orion/VISION/Catalog/VISION_+_Spitzer_s_noYSO.fits"
control_path = "/Users/Antares/Dropbox/Data/Orion/VISION/Catalog/VISION_CF+_Spitzer_s.fits"
results_path = "/Users/Antares/Dropbox/Projects/PNICER/Paper/Results/"
extinction_herschel_path = "/Users/Antares/Dropbox/Data/Orion/Other/Orion_Planck_Herschel_fit_wcs_AK_OriA.fits"


# ----------------------------------------------------------------------
# Load colorbrewer colormap
cmap = brewer2mpl.get_map("Blues", "Sequential", number=9, reverse=False).get_mpl_colormap(N=100, gamma=0.7)


# ----------------------------------------------------------------------
# Load catalog data
skip = 1
science_dummy = fits.open(science_path)[1].data
control_dummy = fits.open(control_path)[1].data

science_glon = science_dummy["GLON"][::skip]
science_glat = science_dummy["GLAT"][::skip]

control_glon = control_dummy["GLON"][::skip]
control_glat = control_dummy["GLAT"][::skip]

features_names = ["J", "H", "Ks", "IRAC1", "IRAC2"]
errors_names = ["J_err", "H_err", "Ks_err", "IRAC1_err", "IRAC2_err"]
features_extinction = [2.5, 1.55, 1.0, 0.636, 0.54]

n_features = 5

# Photometry
science_data = [science_dummy[n][::skip] for n in features_names[:n_features]]
control_data = [control_dummy[n][::skip] for n in features_names[:n_features]]

# Measurement errors
science_error = [science_dummy[n][::skip] for n in errors_names[:n_features]]
control_error = [control_dummy[n][::skip] for n in errors_names[:n_features]]
features_extinction = features_extinction[:n_features]
features_names = features_names[:n_features]


# ----------------------------------------------------------------------
# Initialize data
science = Magnitudes(mag=science_data, err=science_error, extvec=features_extinction,
                     lon=science_glon, lat=science_glat, names=features_names).mag2color()
control = Magnitudes(mag=control_data, err=control_error, extvec=features_extinction,
                     lon=control_glon, lat=control_glat, names=features_names).mag2color()


# ----------------------------------------------------------------------
# Set defaults
prange = [-0.5, 2.5]
grid_bw = 0.05
kernel = "epanechnikov"
x, y = np.meshgrid(np.arange(start=prange[0], stop=prange[1], step=grid_bw),
                   np.arange(start=prange[0], stop=prange[1], step=grid_bw))
xgrid = np.vstack([x.ravel(), y.ravel()]).T


# ----------------------------------------------------------------------
# Calculate densities for all color-color spaces
dens_sc, dens_cf = [], []
for idx in combinations(range(4), 2):

    # Create combined mask of features
    mask_sc = np.prod(np.vstack([science.features_masks[idx[1]], science.features_masks[idx[0]]]), axis=0, dtype=bool)
    mask_cf = np.prod(np.vstack([control.features_masks[idx[1]], control.features_masks[idx[0]]]), axis=0, dtype=bool)

    dens_sc.append(point_density(xdata=science.features[idx[1]], ydata=science.features[idx[0]],
                                 xsize=grid_bw, ysize=grid_bw))
    dens_cf.append(point_density(xdata=control.features[idx[1]], ydata=control.features[idx[0]],
                                 xsize=grid_bw, ysize=grid_bw))

    # print(science.features_names[idx[1]], science.features_names[idx[0]])

    # # Get data into shape
    # data_sc = np.vstack([science.features[idx[1]][mask_sc], science.features[idx[0]][mask_sc]]).T
    # data_cf = np.vstack([control.features[idx[1]][mask_cf], control.features[idx[0]][mask_cf]]).T

    # # Get density
    # dens_sc.append(mp_kde(grid=xgrid, data=data_sc, bandwidth=grid_bw*4, shape=x.shape, kernel=kernel))
    # dens_cf.append(mp_kde(grid=xgrid, data=data_cf, bandwidth=grid_bw*4, shape=x.shape, kernel=kernel))


# ----------------------------------------------------------------------
# Create Plot grid
fig = plt.figure(figsize=[15, 6.665])
grid_science = GridSpec(ncols=3, nrows=3, bottom=0.05, top=0.95, left=0.05, right=0.45, hspace=0, wspace=0)
grid_control = GridSpec(ncols=3, nrows=3, bottom=0.05, top=0.95, left=0.50, right=0.90, hspace=0, wspace=0)


# ----------------------------------------------------------------------
# Plot
pidx = reversed([0, 3, 4, 6, 7, 8])
names = ["$J-H \/ \mathrm{(mag)}$", "$H-K_S \/ \mathrm{(mag)}$",
         "$K_S-IRAC_{3.4} \/ \mathrm{(mag)}$", "$IRAC_{3.4}-IRAC_{4.5} \/ \mathrm{(mag)}$"]

for i, p, c in zip(range(len(dens_sc)), pidx, combinations(range(4), 2)):

    # print(i, p, c)

    # Add axes
    ax_sc = plt.subplot(grid_science[p])
    ax_cf = plt.subplot(grid_control[p])

    # # Show
    # ax_sc.imshow(np.sqrt(dens_sc[i]), origin="lower", interpolation="nearest",
    #              extent=[prange[0], prange[1], prange[0], prange[1]], cmap=cmap)
    # ax_cf.imshow(np.sqrt(dens_cf[i]), origin="lower", interpolation="nearest",
    #              extent=[prange[0], prange[1], prange[0], prange[1]], cmap=cmap)

    # Scatter plot
    ax_sc.scatter(science.features[c[1]], science.features[c[0]], lw=0, s=1, marker=".",
                  c=np.sqrt(dens_sc[i] / np.nanmax(dens_sc[i])), cmap=cmap)
    ax_cf.scatter(control.features[c[1]], control.features[c[0]], lw=0, s=1, marker=".",
                  c=np.sqrt(dens_cf[i] / np.nanmax(dens_cf[i])), cmap=cmap)

    # Adjust common stuff
    for ax in [ax_sc, ax_cf]:

        # Add extinction arrow
        ax.arrow(1, 0.2, science.extvec.extvec[c[1]], science.extvec.extvec[c[0]], length_includes_head=True,
                 width=0.01, head_width=0.04, head_length=0.06, facecolor="black", alpha=0.7)

        # # Annotate extinction arrow
        # ax.annotate("$A_K \/ = \/ 1$", xy=[1 + 0.40 * science.extvec.extvec[c[0]],
        #                                    0.2 + 0.2 * science.extvec.extvec[c[1]]],
        #             xycoords="data", ha="center", va="center",
        #             rotation=np.degrees(np.arctan(science.extvec.extvec[c[0]] / science.extvec.extvec[c[1]])))

        # Force aspect ratio
        ax.set_aspect(1)

        # Limits
        ax.set_xlim(-0.6, 2.3)
        ax.set_ylim(-0.3, 2.6)

        # Ticks
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.2))

        # Labels
        if c[0] == 0:
            ax.set_xlabel(names[c[1]])
        else:
            ax.axes.xaxis.set_ticklabels([])
        if c[1] == 3:
            ax.set_ylabel(names[c[0]])
        else:
            ax.axes.yaxis.set_ticklabels([])


# Save
plt.savefig(results_path + "combinations_scatter.png", bbox_inches="tight", dpi=300)
