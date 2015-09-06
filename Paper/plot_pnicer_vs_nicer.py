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
res, sampling = 0.05, 8
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
fig = plt.figure(figsize=[14, 16.7])
grid_diff = GridSpec(ncols=1, nrows=5, bottom=0.05, top=0.9, left=0.05, right=0.3, hspace=0, wspace=0,
                     height_ratios=[0.05, 1, 1, 1, 1])
grid_hist = GridSpec(ncols=2, nrows=5, bottom=0.05, top=0.9, left=0.35, right=0.85, hspace=0, wspace=0.02,
                     height_ratios=[0.05, 1, 1, 1, 1])

# Add cax
cax = plt.subplot(grid_diff[0])

# ----------------------------------------------------------------------
# Calculate extinction for given features
for fidx, hidx in zip(range(2, 6), range(2, 9, 2)):

    # Initialize data
    control = Magnitudes(mag=control_data[0:fidx], err=control_error[0:fidx], extvec=features_extinction[0:fidx],
                         lon=control_glon, lat=control_glat, names=features_names[0:fidx])

    # Get NICER and PNICER extinction
    pnicer = control.pnicer(control=control, sampling=2, kernel="epanechnikov", add_colors=True)
    # pnicer = control.mag2color().pnicer(control=control.mag2color(), sampling=2, kernel="epanechnikov").extinction
    # pnicer = control.pnicer(control=control, sampling=2, kernel="epanechnikov", add_colors=True).extinction
    nicer = control.nicer(control=control)

    pnicer_ext = pnicer.extinction
    pnicer_err = np.sqrt(pnicer.variance)
    nicer_ext = nicer.extinction
    nicer_err = np.sqrt(nicer.variance)

    # Get average extinction within box for each source
    avg_pnicer = point_average(xdata=xdata, ydata=ydata, zdata=pnicer_ext, xsize=xsize, ysize=ysize)
    avg_nicer = point_average(xdata=xdata, ydata=ydata, zdata=nicer_ext, xsize=xsize, ysize=ysize)

    # Add axes plot
    ax_diff = plt.subplot(grid_diff[fidx-1])
    ax_diff.set_aspect(1)
    ax_hist_ext = plt.subplot(grid_hist[hidx])
    ax_hist_err = plt.subplot(grid_hist[hidx+1])

    # Plot diff
    im = ax_diff.scatter(xdata, ydata, s=2, lw=0, c=avg_pnicer - avg_nicer, vmin=-0.25, vmax=0.25, cmap=cmap, alpha=1)

    # Add colorbar
    if fidx == 2:
        cbar = plt.colorbar(im, cax=cax, ticks=MultipleLocator(0.1), format="%.2f", orientation="horizontal")
        cbar.ax.xaxis.set_ticks_position("top")
        cbar.set_label("$A_{K, \/ \mathrm{PNICER}} \/ - \/ A_{K, \/ \mathrm{NICER}}$ (mag)", labelpad=-50)

    # Filter NaNs
    pnicer_ext = pnicer_ext[np.isfinite(pnicer_ext)]
    pnicer_err = pnicer_err[np.isfinite(pnicer_err)]
    nicer_ext = nicer_ext[np.isfinite(nicer_ext)]
    nicer_err = nicer_err[np.isfinite(nicer_err)]

    # Additional filter
    pnicer_ext = pnicer_ext[np.abs(pnicer_ext) < 2]
    nicer_ext = nicer_ext[np.abs(nicer_ext) < 2]

    # Do KDE
    pnicer_ext_hist = mp_kde(grid=grid_kde, data=pnicer_ext, bandwidth=res, sampling=sampling,
                             shape=None, kernel="epanechnikov", absolute=True)
    nicer_ext_hist = mp_kde(grid=grid_kde, data=nicer_ext, bandwidth=res, sampling=sampling,
                            shape=None, kernel="epanechnikov", absolute=True)
    pnicer_err_hist = mp_kde(grid=grid_kde, data=pnicer_err, bandwidth=res, sampling=sampling,
                             shape=None, kernel="epanechnikov", absolute=True)
    nicer_err_hist = mp_kde(grid=grid_kde, data=nicer_err, bandwidth=res, sampling=sampling,
                            shape=None, kernel="epanechnikov", absolute=True)

    # Plot histograms
    ax_hist_ext.plot(grid_kde, nicer_ext_hist, color="#d53e4f", lw=3, label="NICER")
    ax_hist_ext.plot(grid_kde, pnicer_ext_hist, color="#3288bd", lw=3, label="PNICER")
    ax_hist_err.plot(grid_kde, nicer_err_hist, color="#d53e4f", lw=3, label="NICER")
    ax_hist_err.plot(grid_kde, pnicer_err_hist, color="#3288bd", lw=3, label="PNICER")

    # Plot CCD contour
    ax_diff.contour(hist / np.max(hist), extent=edges, levels=[0.005, 0.03, 0.25, 0.5],
                    colors="black", linewidths=1, alpha=0.3)

    if fidx == 2:
        ax_hist_err.legend(loc="upper center", bbox_to_anchor=(0.5, 1.11), ncol=2, frameon=False)

    # Annotate mean and std
    ax_hist_ext.annotate("{:0.0e}".format(np.nanmean(nicer_ext)) + "$\pm$" + "{:0.2f}".format(np.nanstd(pnicer_ext)),
                         xy=[0.05, 0.95], xycoords="axes fraction", ha="left", va="top", color="#3288bd")
    ax_hist_ext.annotate("{:0.0e}".format(np.nanmean(nicer_ext)) + "$\pm$" + "{:0.2f}".format(np.nanstd(nicer_ext)),
                         xy=[0.05, 0.90], xycoords="axes fraction", ha="left", va="top", color="#d53e4f")

    # Set range
    ax_diff.set_xlim(-0.4, 1.6)
    ax_diff.set_ylim(-0.2, 1.8)
    ax_hist_ext.set_xlim(-0.8, 0.8)
    ax_hist_err.set_xlim(0.01, 0.6)

    # Ticker
    ax_diff.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax_diff.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax_hist_ext.xaxis.set_major_locator(MultipleLocator(0.3))
    ax_hist_ext.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax_hist_ext.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax_hist_err.xaxis.set_major_locator(MultipleLocator(0.1))
    ax_hist_err.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax_hist_err.yaxis.set_minor_locator(AutoMinorLocator(5))

    # Delete tick labels
    if fidx < 5:
        ax_diff.axes.xaxis.set_ticklabels([])
        ax_hist_ext.axes.xaxis.set_ticklabels([])
        ax_hist_err.axes.xaxis.set_ticklabels([])

    if fidx == 5:
        ax_diff.set_xlabel("$H-K_S \/ \mathrm{(mag)}$")
        ax_hist_ext.set_xlabel("$A_K \/ \mathrm{(mag)}$")
        ax_hist_err.set_xlabel("$\Delta A_K \/ \mathrm{(mag)}$")
        ax_hist_ext.xaxis.get_major_ticks()[-1].set_visible(False)

    if fidx > 2:
        ax_diff.yaxis.get_major_ticks()[-1].set_visible(False)
        ax_hist_ext.yaxis.get_major_ticks()[-1].set_visible(False)
        ax_hist_err.yaxis.get_major_ticks()[-1].set_visible(False)

    ax_diff.set_ylabel("$J-H \/ \mathrm{(mag)}$")
    ax_hist_err.yaxis.tick_right()
    ax_hist_err.yaxis.set_ticks_position('both')
    ax_hist_err.yaxis.set_label_position("right")
    ax_hist_err.set_ylabel("N")

# Save
plt.savefig("/Users/Antares/Dropbox/Projects/PNICER/Paper/Results/nicer_vs_pnicer.png", bbox_inches="tight")
