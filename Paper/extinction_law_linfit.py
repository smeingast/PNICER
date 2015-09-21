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
from helper import get_beta_ols, get_beta_bces, get_beta_lines, pnicer_ini


# ----------------------------------------------------------------------
# Define file paths
science_path = "/Users/Antares/Dropbox/Data/Orion/VISION/Catalog/VISION_+_Spitzer_s_noYSO.fits"
control_path = "/Users/Antares/Dropbox/Data/Orion/VISION/Catalog/VISION_CF+_Spitzer_s.fits"
results_path = "/Users/Antares/Dropbox/Projects/PNICER/Paper/Results/"


# ----------------------------------------------------------------------
# Load colorbrewer colormap
cmap = brewer2mpl.get_map("YlGnBu", "Sequential", number=9, reverse=True).get_mpl_colormap(N=11, gamma=1)


# ----------------------------------------------------------------------
# Intialize PNICER
science, control = pnicer_ini(skip_science=1, skip_control=1, n_features=5, color=False)
science_color, control_color = science.mag2color(), control.mag2color()

# Additionally load galaxy classifier
class_sex_science = fits.open(science_path)[1].data["class_sex"]
class_sex_control = fits.open(control_path)[1].data["class_sex"]
class_cog_science = fits.open(science_path)[1].data["class_cog"]
class_cog_control = fits.open(control_path)[1].data["class_cog"]


# ----------------------------------------------------------------------
# Make pre-selection of data
ext = science_color.pnicer(control=control_color).extinction
# ext = np.full_like(science.features[0], fill_value=1.0)
for d in [science.dict, control.dict]:
    # Define filter
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fil = (d["J"] > 0) & (d["H"] > 0) & (d["Ks"] < 15) & (d["IRAC1"] > 0) & (d["IRAC2"] > 0) & \
              (d["IRAC1_err"] < 0.1) & (d["Ks_err"] < 0.1)

        # Test no filtering
        # fil = data.combined_mask

        if d == science.dict:
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
class_sex_science = fits.open(science_path)[1].data["class_sex"]
class_sex_control = fits.open(control_path)[1].data["class_sex"]
class_cog_science = fits.open(science_path)[1].data["class_cog"]
class_cog_control = fits.open(control_path)[1].data["class_cog"]




a = science.get_extinction_law(base_index=(1, 2), method="LINES", control=control)
print(a)
exit()



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
plt.savefig(results_path + "extinction_law_linfit_sources.png", bbox_inches="tight", dpi=300)
plt.close()


# ----------------------------------------------------------------------
# Choose base photometric bands for fitting
base_key = ("H", "Ks")
base_ext = (1.55, 1.0)
fit_key = ("J", "IRAC1", "IRAC2")


# ----------------------------------------------------------------------
# Plot CCDs for fitting and actually do the fit...
fig2 = plt.figure(figsize=[15, 5])
grid = GridSpec(ncols=len(fit_key), nrows=1, bottom=0.05, top=0.95, left=0.05, right=0.95, hspace=0.1, wspace=0.1)

for key, pidx in zip(fit_key, range(len(fit_key))):

    # Get shortcut for data
    xdata_science = science.dict[base_key[0]][sfil] - science.dict[base_key[1]][sfil]
    ydata_science = science.dict[base_key[1]][sfil] - science.dict[key][sfil]

    xdata_control = control.dict[base_key[0]][cfil] - control.dict[base_key[1]][cfil]
    ydata_control = control.dict[base_key[1]][cfil] - control.dict[key][cfil]

    # Get ordinary least squares slope
    beta_ols = get_beta_ols(xj=xdata_science, yj=ydata_science)

    # calculate some things for fitting
    cov_err_sc = -np.mean(science.dict[base_key[1] + "_err"][sfil]) ** 2
    cov_err_cf = -np.mean(control.dict[base_key[1] + "_err"][cfil]) ** 2
    var_err_sc = np.mean(science.dict[base_key[0] + "_err"][sfil]) ** 2 + \
                 np.mean(science.dict[base_key[1] + "_err"][sfil]) ** 2
    var_err_cf = np.mean(control.dict[base_key[0] + "_err"][cfil]) ** 2 + \
                 np.mean(control.dict[base_key[1] + "_err"][cfil]) ** 2

    # Get BCES slope
    beta_bces = get_beta_bces(x_sc=xdata_science, y_sc=ydata_science,
                              cov_err_sc=cov_err_sc, var_err_sc=var_err_sc)

    # Get LINES slope
    beta_lines = get_beta_lines(x_sc=xdata_science, y_sc=ydata_science, x_cf=xdata_control, y_cf=ydata_control,
                                cov_err_sc=cov_err_sc, cov_err_cf=cov_err_cf,
                                var_err_sc=var_err_sc, var_err_cf=var_err_cf)

    # Get extinction for current band
    ext_ols = base_ext[1] - beta_ols * (base_ext[0] - base_ext[1])
    ext_bces = base_ext[1] - beta_bces * (base_ext[0] - base_ext[1])
    ext_lines = base_ext[1] - beta_lines * (base_ext[0] - base_ext[1])

    # Print results
    print(key)
    # print("OLS: Beta = ", beta_ols)
    # print("BCES: Beta = ", beta_bces)
    print("LINES: Beta = ", beta_lines)
    # print("OLS: Ext = ", ext_ols)
    # print("BCES: Ext = ", ext_bces)
    print("LINES: Ext = ", ext_lines)
    print()

    # Add subplot
    ax = plt.subplot(grid[pidx])

    # Get point density
    dens = point_density(xdata=xdata_science, ydata=ydata_science, xsize=0.1, ysize=0.1)

    # Plot data
    ax.scatter(xdata_science, ydata_science, lw=0, s=2, alpha=1, c=dens, cmap=cmap)

    # Plot slope and make it pass through median
    ic = np.median(ydata_science) - beta_lines * np.median(xdata_science)
    ax.plot(np.arange(-1, 5, 1), beta_lines * np.arange(-1, 5, 1) + ic, color="black", lw=2, linestyle="dashed")

    # Annotate band
    ax.annotate(key, xy=[0.5, 1.01], xycoords="axes fraction", ha="center", va="bottom")

    # Limits
    ax.set_xlim(-0.5, 4)
    if pidx > 0:
        ax.set_ylim(-0.5, 4)
    else:
        ax.set_ylim(-4, 0.5)

    # Labels
    # if pidx > 1:
    #     ax.axes.yaxis.set_ticklabels([])
    if pidx == 0:
        ax.set_ylabel("$" + base_key[1] + "-\lambda$")
    ax.set_xlabel("$" + base_key[0] + "-" + base_key[1] + "$")

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
