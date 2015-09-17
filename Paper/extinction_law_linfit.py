# ----------------------------------------------------------------------
# Import stuff
import warnings
import brewer2mpl
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.pyplot import GridSpec
from matplotlib.ticker import MultipleLocator
from MyFunctions import point_density
from helper import get_beta_ols, get_beta_bces, get_beta_lines, pnicer_ini


# ----------------------------------------------------------------------
# Define file paths
results_path = "/Users/Antares/Dropbox/Projects/PNICER/Paper/Results/"


# ----------------------------------------------------------------------
# Load colorbrewer colormap
cmap = brewer2mpl.get_map("YlGnBu", "Sequential", number=9, reverse=True).get_mpl_colormap(N=11, gamma=1)


# ----------------------------------------------------------------------
# Intialize PNICER
science, control = pnicer_ini(skip_science=1, skip_control=1, n_features=5, color=False)
science_color, control_color = science.mag2color(), control.mag2color()


# ----------------------------------------------------------------------
# Define parameter range
ajr = (2.4, 2.7)
ahr = (1.4, 1.7)
a1r = (0.5, 0.7)
a2r = (0.3, 0.6)

# Define baseline extinction
a_hk = 1.55


# ----------------------------------------------------------------------
# Make pre-selection of data
ext = science_color.pnicer(control=control_color).extinction
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

        # Test no filtering
        # fil = data.combined_mask

        if data == science_color:
            sfil = fil & (ext > 0.1)
            # sfil = fil.copy()
        else:
            cfil = fil.copy()


# ----------------------------------------------------------------------
# Plot pre-selection of data
fig1 = plt.figure(figsize=[17, 7.55])
grid1 = GridSpec(ncols=3, nrows=3, bottom=0.05, top=0.95, left=0.05, right=0.45, hspace=0, wspace=0)
grid2 = GridSpec(ncols=3, nrows=3, bottom=0.05, top=0.95, left=0.50, right=0.90, hspace=0, wspace=0)

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

        # Force aspect ratio
        # ax.set_aspect(1)

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
plt.savefig(results_path + "extinction_law_linfit.png", bbox_inches="tight", dpi=300)
plt.close()


# ----------------------------------------------------------------------
# Plot CCDs for fitting
fig2 = plt.figure(figsize=[15, 5])
grid = GridSpec(ncols=3, nrows=1, bottom=0.05, top=0.95, left=0.05, right=0.95, hspace=0.1, wspace=0.1)

plot_index = [0, 1, 2]
data_index = [[1, 2, 2, 0], [1, 2, 2, 3], [1, 2, 2, 4]]
for (idx1, idx2, idx3, idx4), pidx in zip(data_index, plot_index):

    # Get shortcut for data
    xdata = science.features[idx1][sfil] - science.features[idx2][sfil]
    ydata = science.features[idx3][sfil] - science.features[idx4][sfil]

    # Get ordinary least squares slope
    beta_ols = get_beta_ols(xj=xdata, yj=ydata)

    # Get BCES slope
    beta_bces = get_beta_bces(x_sc=xdata, y_sc=ydata,
                              cov_err_sc=-np.mean(science.features_err[idx2][sfil]) ** 2,
                              var_err_sc=np.mean(science.features_err[idx1][sfil]) ** 2 +
                              np.mean(science.features_err[idx2][sfil]) ** 2)

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
    slope_ols = 1 - beta_ols * (a_hk - 1)
    slope_bces = 1 - beta_bces * (a_hk - 1)
    slope_lines = 1 - beta_lines * (a_hk - 1)

    # Print results
    print(science.features_names[idx4])
    print("OLS:", 1 - beta_ols * (a_hk - 1))
    print("BCES:", 1 - beta_bces * (a_hk - 1))
    print("LINES:", 1 - beta_lines * (a_hk - 1))
    print()

    # Add subplot
    ax = plt.subplot(grid[pidx])

    # Get point density
    dens = point_density(xdata=xdata, ydata=ydata, xsize=0.1, ysize=0.1)

    # Plot data
    ax.scatter(xdata, ydata, lw=0, s=2, alpha=1, c=dens, cmap=cmap)

    # Plot slope
    ax.plot(np.arange(-1, 5, 1), beta_lines * np.arange(-1, 5, 1), color="black", lw=2, linestyle="dashed")

    # Limits
    ax.set_xlim(-0.5, 3.5)
    if pidx > 0:
        ax.set_ylim(-0.5, 3.5)
    else:
        ax.set_ylim(-3.5, 0.5)

    # Labels
    # if pidx > 1:
    #     ax.axes.yaxis.set_ticklabels([])
    if pidx == 0:
        ax.set_ylabel("$K_S - \lambda$")
    ax.set_xlabel("$H - K_S$")

    # Ticker
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.2))

    # Force aspect ratio
    ax.set_aspect(1)


# Save figure
plt.savefig(results_path + "extinction_law_linfit_fit.png", bbox_inches="tight", dpi=300)
plt.close()
