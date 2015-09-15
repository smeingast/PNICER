# ----------------------------------------------------------------------
# Import stuff
import warnings
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
results_path = "/Users/Antares/Dropbox/Projects/PNICER/Paper/Results/"


# ----------------------------------------------------------------------
# Load data
skip = 1
science_dummy = fits.open(science_path)[1].data
control_dummy = fits.open(control_path)[1].data

# Coordinates
science_glon = science_dummy["GLON"][::skip]
science_glat = science_dummy["GLAT"][::skip]
control_glon = control_dummy["GLON"][::skip]
control_glat = control_dummy["GLAT"][::skip]

# Definitions
features_names = ["J", "H", "Ks", "IRAC1", "IRAC2"]
errors_names = ["J_err", "H_err", "Ks_err", "IRAC1_err", "IRAC2_err"]
features_extinction = [2.5, 1.55, 1.0, 0.636, 0.54]

# Photometry
science_data = [science_dummy[n][::skip] for n in features_names]
science_error = [science_dummy[n][::skip] for n in errors_names]
control_data = [control_dummy[n][::skip] for n in features_names]
control_error = [control_dummy[n][::skip] for n in errors_names]


# ----------------------------------------------------------------------
# Initialize data with PNICER
science = Magnitudes(mag=science_data, err=science_error, extvec=features_extinction,
                     lon=science_glon, lat=science_glat, names=features_names)
control = Magnitudes(mag=control_data, err=control_error, extvec=features_extinction,
                     lon=control_glon, lat=control_glat, names=features_names)

science_color = science.mag2color()
control_color = control.mag2color()


# ----------------------------------------------------------------------
# Define parameter range
ajr = (2.4, 2.7)
ahr = (1.4, 1.7)
a1r = (0.5, 0.7)
a2r = (0.3, 0.6)


# ----------------------------------------------------------------------
# Make pre-selection of data
for data in [science_color, control_color]:
    # Define filter
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fil = (data.features[0] + 1 < (ajr[0] - ahr[0]) / (ahr[0] - 1) * (data.features[1] + 0.7)) & \
              (data.features[0] + 1 > (ajr[1] - ahr[1]) / (ahr[1] - 1) * (data.features[1] + 0.7)) & \
              (data.features[0] + 1 > (ajr[0] - ahr[0]) / (1 - a1r[0]) * (data.features[2] + 0.5)) & \
              (data.features[0] + 1 < (ajr[1] - ahr[1]) / (1 - a1r[1]) * (data.features[2] + 0.5)) & \
              (data.features[0] + 1 > (ajr[0] - ahr[0]) / (a1r[0] - a2r[0]) * (data.features[3] + 0.2)) & \
              (data.features[0] + 1 < (ajr[1] - ahr[1]) / (a1r[1] - a2r[1]) * (data.features[3] + 0.2)) & \
              (np.nanmax(np.array(data.features_err), axis=0) < 0.1)

        # fil = data.combined_mask

    if data == science_color:
        sfil = fil.copy()
    else:
        cfil = fil.copy()


# ----------------------------------------------------------------------
# Plot pre-selection of data
fig1 = plt.figure(figsize=[20, 10])
grid1 = GridSpec(ncols=3, nrows=3, bottom=0.05, top=0.95, left=0.05, right=0.45, hspace=0.01, wspace=0.01)
grid2 = GridSpec(ncols=3, nrows=3, bottom=0.05, top=0.95, left=0.50, right=0.90, hspace=0.01, wspace=0.01)

plot_index = [8, 7, 6, 4, 3, 0]
data_index = [[1, 0], [2, 0], [3, 0], [2, 1], [3, 1], [3, 2]]

for (idx1, idx2), pidx in zip(data_index, plot_index):

    ax1 = plt.subplot(grid1[pidx])
    ax2 = plt.subplot(grid2[pidx])

    # Plot all data for science field
    ax1.scatter(science_color.features[idx1], science_color.features[idx2], lw=0, s=5, color="red", alpha=0.01)
    # Plot filtered data for science field
    ax1.scatter(science_color.features[idx1][sfil], science_color.features[idx2][sfil],
                lw=0, s=5, color="blue", alpha=0.1)

    # Plot all data for control field
    ax2.scatter(control_color.features[idx1], control_color.features[idx2], lw=0, s=5, color="red", alpha=0.01)
    # Plot filtered data for control field
    ax2.scatter(control_color.features[idx1][cfil], control_color.features[idx2][cfil],
                lw=0, s=5, color="blue", alpha=0.1)

    # Adjust axes
    for ax in [ax1, ax2]:

        # limits
        ax.set_xlim(-0.5, 3)
        ax.set_ylim(-0.5, 3)

        # Ticker
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.2))
        # Remove labels
        if pidx < 6:
            ax.axes.xaxis.set_ticklabels([])
        if pidx in [4, 7, 8]:
            ax.axes.yaxis.set_ticklabels([])
        # set labels
        if pidx == 8:
            ax.set_xlabel("$H-K_S$")
        if pidx == 7:
            ax.set_xlabel("$K_S - [3.6]$")
        if pidx == 6:
            ax.set_xlabel("$[3.6] - [4.5]$")
            ax.set_ylabel("$J-H$")
        if pidx == 3:
            ax.set_ylabel("$H-K_S$")
        if pidx == 0:
            ax.set_ylabel("$K_S - [3.6]$")

# Save figure
plt.savefig(results_path + "extinction_law_sources.png", bbox_inches="tight")
plt.close()


# ----------------------------------------------------------------------
# Define helper functions
def get_covar(xi, yi):
    """
    Calculate sample covariance
    :param xi: x data
    :param yi: y data
    :return: sample covariance
    """
    return np.sum((xi - np.mean(xi)) * (yi - np.mean(yi))) / len(xi)


def get_beta_ols(xj, yj):
    """
    Get slope of ordinary least squares fit
    :param xj: x data
    :param yj: y data
    :return: slope of linear fit
    """
    return get_covar(xi=xj, yi=yj) / np.var(xj)


def get_beta_lines(x_sc, y_sc, x_cf, y_cf, cov_err_sc, cov_err_cf, var_err_sc, var_err_cf):
    """
    Get slope of distribution with LINES
    :param x_sc: x data science field
    :param y_sc: y data science field
    :param x_cf: x data control field
    :param y_cf: y data control field
    :param cov_err_sc: covariance of errors science field
    :param cov_err_cf: covariance of errors control field
    :param var_err_sc: variance in x science field
    :param var_err_cf: variance in y control field
    :return: slope of linear fit
    """
    upper = get_covar(x_sc, y_sc) - get_covar(x_cf, y_cf) - cov_err_sc + cov_err_cf
    lower = np.var(x_sc) - np.var(x_cf) - var_err_sc + var_err_cf
    return upper / lower


# ----------------------------------------------------------------------
# Plot CCDs for fitting
fig2 = plt.figure(figsize=[15, 5])
grid = GridSpec(ncols=3, nrows=1, bottom=0.05, top=0.95, left=0.05, right=0.95, hspace=0.1, wspace=0.1)

plot_index = [0, 1, 2]
data_index = [[1, 2, 2, 0], [1, 2, 2, 3], [1, 2, 2, 4]]

for (idx1, idx2, idx3, idx4), pidx in zip(data_index, plot_index):

    # Add subplot
    ax = plt.subplot(grid[pidx])

    # Plot
    ax.scatter(science.features[idx1][sfil] - science.features[idx2][sfil],
               science.features[idx3][sfil] - science.features[idx4][sfil], lw=0, s=5, alpha=.1)

    # Limits
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)

    # Get ordinary least squares slope
    beta_ols = get_beta_ols(xj=science.features[idx1][sfil] - science.features[idx2][sfil],
                            yj=science.features[idx3][sfil] - science.features[idx4][sfil])

    # Get LINES slope
    beta_lines = get_beta_lines(x_sc=science.features[idx1][sfil] - science.features[idx2][sfil],
                                y_sc=science.features[idx3][sfil] - science.features[idx4][sfil],
                                x_cf=control.features[idx1][cfil] - control.features[idx2][cfil],
                                y_cf=control.features[idx3][cfil] - control.features[idx4][cfil],
                                cov_err_sc=-np.mean(science.features_err[idx2][sfil])**2,
                                cov_err_cf=-np.mean(control.features_err[idx2][cfil])**2,
                                var_err_sc=np.mean(science.features_err[idx1][sfil])**2 +
                                np.mean(science.features_err[idx2][sfil])**2,
                                var_err_cf=np.mean(control.features_err[idx1][cfil])**2 +
                                np.mean(control.features_err[idx2][cfil])**2)

    print(science.features_names[idx4])
    print("OLS:", 1 - beta_ols * 0.55)
    print("LINES:", 1 - beta_lines * 0.55)
    print()

plt.savefig(results_path + "extinction_law_sources_fit.png", bbox_inches="tight")
plt.close()
