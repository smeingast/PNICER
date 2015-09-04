from __future__ import absolute_import, division, print_function
__author__ = 'Stefan Meingast'


# ----------------------------------------------------------------------
# Import stuff
import brewer2mpl
import numpy as np
import matplotlib.pyplot as plt

from pnicer import Magnitudes, mp_kde
from astropy.io import fits
from matplotlib.pyplot import GridSpec
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from MyFunctions import point_average


# ----------------------------------------------------------------------
# Define file paths
science_path = "/Users/Antares/Dropbox/Data/Orion/VISION/Catalog/VISION_+_Spitzer_s.fits"
control_path = "/Users/Antares/Dropbox/Data/Orion/VISION/Catalog/VISION_CF+_Spitzer_s.fits"
results_path = "/Users/Antares/Dropbox/Projects/Dereddening/Results/"

# Load colormap
cmap = brewer2mpl.get_map('RdYlBu', 'Diverging', number=11, reverse=True).get_mpl_colormap(N=15, gamma=1)


# ----------------------------------------------------------------------
# Load data
skip = 1
science_dummy = fits.open(science_path)[1].data
control_dummy = fits.open(control_path)[1].data

control_glon = control_dummy["GLON"][::skip]
control_glat = control_dummy["GLAT"][::skip]


# ----------------------------------------------------------------------
# Define features to be used
features_names = ["J", "H", "Ks", "IRAC1", "IRAC2"]
errors_names = ["J_err", "H_err", "Ks_err", "IRAC1_err", "IRAC2_err"]

# Define extinction
features_extinction = [2.5, 1.55, 1.0, 0.636, 0.54]


# ----------------------------------------------------------------------
# Load data into lists for PNICER
control_data = [control_dummy[n][::skip] for n in features_names]
control_error = [control_dummy[n][::skip] for n in errors_names]


# ----------------------------------------------------------------------
# Define KDE standard
erange = [-1, 1]
res, sampling = 0.05, 4
grid_kde = np.arange(start=1.1*erange[0], stop=1.1*erange[1],
                     step=res/sampling, dtype=np.double)

# ----------------------------------------------------------------------
# Define color color data
xdata = control_data[1] - control_data[2]
ydata = control_data[0] - control_data[1]
xsize = ysize = 0.05


# ----------------------------------------------------------------------
# Get 2D histogram of control field CCD
"""Same as in method plot"""
grid_bw = 0.04
l, h = -0.6, 2.4 + grid_bw / 2
x, y = np.meshgrid(np.arange(start=l, stop=h, step=grid_bw), np.arange(start=l, stop=h, step=grid_bw))
xgrid = np.vstack([x.ravel(), y.ravel()]).T
edges = (np.min(x), np.max(x), np.min(y), np.max(y))
fil = (np.isfinite(xdata)) & (np.isfinite(ydata))
data = np.vstack([xdata[fil], ydata[fil]]).T
hist = mp_kde(grid=xgrid, data=data, bandwidth=grid_bw * 2, shape=x.shape, kernel="epanechnikov")


# ----------------------------------------------------------------------
# Create plot grid
fig = plt.figure(figsize=[10, 20])
grid = GridSpec(ncols=2, nrows=5, bottom=0.05, top=0.9, left=0.05, right=0.95, hspace=0, wspace=0.1,
                height_ratios=[0.05, 1, 1, 1, 1])

# Add cax
cax = plt.subplot(grid[0])

# ----------------------------------------------------------------------
# Calculate extinction for given features

for idx, pidx in zip(range(2, 6), range(2, 9, 2)):

    # Initialize data
    control = Magnitudes(mag=control_data[0:idx], err=control_error[0:idx], extvec=features_extinction[0:idx],
                         lon=control_glon, lat=control_glat, names=features_names[0:idx])

    # Get NICER and PNICER extinction
    pnicer = control.pnicer(control=control, sampling=2, kernel="epanechnikov", use_color=True).extinction
    nicer = control.nicer(control=control).extinction

    # Get average extinction within box for each source
    avg_pnicer = point_average(xdata=xdata, ydata=ydata, zdata=pnicer, xsize=xsize, ysize=ysize)
    avg_nicer = point_average(xdata=xdata, ydata=ydata, zdata=nicer, xsize=xsize, ysize=ysize)

    # Add axes plot
    ax_diff = plt.subplot(grid[pidx])
    ax_hist = plt.subplot(grid[pidx+1])

    # Plot diff
    im = ax_diff.scatter(xdata, ydata, s=2, lw=0, c=avg_pnicer - avg_nicer, vmin=-0.25, vmax=0.25, cmap=cmap, alpha=1)

    # Add colorbar
    if pidx == 2:
        cbar = plt.colorbar(im, cax=cax, ticks=MultipleLocator(0.1), format="%.2f", orientation="horizontal")
        cbar.ax.xaxis.set_ticks_position("top")
        cbar.set_label("$\Delta A_K$ (PNICER - NICER)", labelpad=-50)

    # Filter NaNs
    pnicer = pnicer[np.isfinite(pnicer)]
    nicer = nicer[np.isfinite(nicer)]

    # Additional filter
    pnicer = pnicer[np.abs(pnicer) < 2]
    nicer = nicer[np.abs(nicer) < 2]

    # Do KDE
    pnicer_hist = mp_kde(grid=grid_kde, data=pnicer, bandwidth=res, sampling=sampling,
                         shape=None, kernel="epanechnikov", absolute=True)
    nicer_hist = mp_kde(grid=grid_kde, data=nicer, bandwidth=res, sampling=sampling,
                        shape=None, kernel="epanechnikov", absolute=True)

    # Plot histograms
    ax_hist.plot(grid_kde, nicer_hist, color="#d53e4f", lw=3, label="NICER")
    ax_hist.plot(grid_kde, pnicer_hist, color="#3288bd", lw=3, label="PNICER")

    # Plot CCD contour
    ax_diff.contour(hist / np.max(hist), extent=edges, levels=[0.005, 0.03, 0.25, 0.5],
                    colors="black", linewidths=1, alpha=0.3)

    if pidx == 2:
        ax_hist.legend(loc="upper center", bbox_to_anchor=(0.5, 1.105), ncol=2, frameon=False)

    # Annotate mean and std
    ax_hist.annotate("{:0.0e}".format(np.nanmean(pnicer)) + "$\pm$" + "{:0.2f}".format(np.nanstd(pnicer)),
                     xy=[0.95, 0.95], xycoords="axes fraction", ha="right", va="top", color="#3288bd")
    ax_hist.annotate("{:0.0e}".format(np.nanmean(nicer)) + "$\pm$" + "{:0.2f}".format(np.nanstd(nicer)),
                     xy=[0.95, 0.90], xycoords="axes fraction", ha="right", va="top", color="#d53e4f")

    # Set range
    ax_diff.set_xlim(-0.3, 1.7)
    ax_diff.set_ylim(-0.2, 1.8)
    ax_hist.set_xlim(erange)

    # Ticker
    ax_diff.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax_diff.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax_hist.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax_hist.yaxis.set_minor_locator(AutoMinorLocator(5))

    # Delete tick labels
    if idx < 5:
        ax_diff.axes.xaxis.set_ticklabels([])
        ax_hist.axes.xaxis.set_ticklabels([])

    if idx == 5:
        ax_diff.set_xlabel("$H-K_S \/ \mathrm{(mag)}$")
        ax_hist.set_xlabel("$A_K \/ \mathrm{(mag)}$")

    if idx > 2:
        ax_diff.yaxis.get_major_ticks()[-1].set_visible(False)
        ax_hist.yaxis.get_major_ticks()[-1].set_visible(False)

    ax_diff.set_ylabel("$J-H \/ \mathrm{(mag)}$")
    ax_hist.yaxis.tick_right()
    ax_hist.yaxis.set_ticks_position('both')
    ax_hist.yaxis.set_label_position("right")
    ax_hist.set_ylabel("N")


# Save
plt.savefig("/Users/Antares/Dropbox/Projects/PNICER/Paper/Results/nicer_vs_pnicer.png", bbox_inches="tight")
