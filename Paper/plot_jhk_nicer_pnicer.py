from __future__ import absolute_import, division, print_function
__author__ = 'Stefan Meingast'

# ----------------------------------------------------------------------
# Import stuff
import brewer2mpl
import numpy as np
import matplotlib.pyplot as plt

from pnicer import Magnitudes
from scipy import ndimage
from astropy.io import fits
from matplotlib.pyplot import GridSpec
from matplotlib.ticker import MultipleLocator
from MyFunctions import point_average


# ----------------------------------------------------------------------
# Define file paths
science_path = "/Users/Antares/Dropbox/Data/Orion/VISION/Catalog/VISION_+_Spitzer_s.fits"
control_path = "/Users/Antares/Dropbox/Data/Orion/VISION/Catalog/VISION_CF+_Spitzer_s.fits"
results_path = "/Users/Antares/Dropbox/Projects/Dereddening/Results/"

# Load colormap
cmap = brewer2mpl.get_map('RdYlBu', 'Diverging', number=11, reverse=True).get_mpl_colormap(N=11, gamma=1)


# ----------------------------------------------------------------------
# Load data
science_dummy = fits.open(science_path)[1].data
control_dummy = fits.open(control_path)[1].data

science_glon = science_dummy["GLON"]
science_glat = science_dummy["GLAT"]

control_glon = control_dummy["GLON"]
control_glat = control_dummy["GLAT"]


# ----------------------------------------------------------------------
# Define features to be used
features_names = ["J", "H", "Ks", "IRAC1", "IRAC2"]
# features_names = ["J", "H", "Ks", "IRAC1"]
# features_names = ["J", "H", "Ks"]
errors_names = ["J_err", "H_err", "Ks_err", "IRAC1_err", "IRAC2_err"]
# errors_names = ["J_err", "H_err", "Ks_err", "IRAC1_err"]
# errors_names = ["J_err", "H_err", "Ks_err"]

# Define extinction
features_extinction = [2.5, 1.55, 1.0, 0.636, 0.54]
# features_extinction = [2.5, 1.55, 1.0, 0.636]
# features_extinction = [2.5, 1.55, 1.0]


# ----------------------------------------------------------------------
# Load data into lists for PNICER
science_data = [science_dummy[n] for n in features_names]
control_data = [control_dummy[n] for n in features_names]
science_error = [science_dummy[n] for n in errors_names]
control_error = [control_dummy[n] for n in errors_names]


# ----------------------------------------------------------------------
# Initialize data
science = Magnitudes(mag=science_data, err=science_error, extvec=features_extinction,
                     lon=science_glon, lat=science_glat, names=features_names)
control = Magnitudes(mag=control_data, err=control_error, extvec=features_extinction,
                     lon=control_glon, lat=control_glat, names=features_names)

# ----------------------------------------------------------------------
# Define X/Y data for plot
xdata = control.features[1] - control.features[2]
ydata = control.features[0] - control.features[1]

xrange = [-0.3, 1.5]
yrange = [-0.1, 1.7]


# ----------------------------------------------------------------------
# Get 2D histogram of control field CCD
fil = np.isfinite(xdata) & np.isfinite(ydata)
hist, xedges, yedges = np.histogram2d(x=xdata[fil], y=ydata[fil], bins=50, range=[xrange, yrange])
hist /= np.max(hist)
# Smooth
hist = ndimage.gaussian_filter(hist, sigma=0.5, order=0)

# ----------------------------------------------------------------------
# Run PNICER and NICER on control field
ext_pnicer = control.pnicer(control=control, bin_ext=0.02, bin_grid=0.02, use_color=False)
ext_nicer = control.nicer(control=control)


# ----------------------------------------------------------------------
# Get average extinction within box for each source
xdata = control.features[1] - control.features[2]
ydata = control.features[0] - control.features[1]
xsize = ysize = 0.05

avg_nicer = point_average(xdata=xdata, ydata=ydata, zdata=ext_nicer.extinction, xsize=xsize, ysize=ysize)
avg_pnicer = point_average(xdata=xdata, ydata=ydata, zdata=ext_pnicer.extinction, xsize=xsize, ysize=ysize)


# ----------------------------------------------------------------------
# Plot results
fig = plt.figure(figsize=[11, 5])
grid = GridSpec(ncols=3, nrows=1, bottom=0.09, top=0.99, left=0.05, right=0.95, hspace=0, wspace=0,
                width_ratios=[1, 1, 0.05])
ax1 = plt.subplot(grid[0])
ax2 = plt.subplot(grid[1])
ax3 = plt.subplot(grid[2])

# Define some scatter plot parameters
s = 5
vmin, vmax = -0.55, 0.55
alpha = 1

# Create scatter plot
im1 = ax1.scatter(control.features[1] - control.features[2], control.features[0] - control.features[1],
                  s=s, lw=0, c=avg_pnicer, vmin=vmin, vmax=vmax, cmap=cmap, alpha=alpha)
im2 = ax2.scatter(control.features[1] - control.features[2], control.features[0] - control.features[1],
                  s=s, lw=0, c=avg_nicer, vmin=vmin, vmax=vmax, cmap=cmap, alpha=alpha)

cbar = plt.colorbar(im1, cax=ax3, label="$A_K$", ticks=MultipleLocator(0.1))

# Modify axes
ax1.annotate("PNICER", xy=(0.05, 0.95), xycoords="axes fraction", ha="left", va="top")
ax2.annotate("NICER", xy=(0.05, 0.95), xycoords="axes fraction", ha="left", va="top")
ax1.set_ylabel("$J - H$ (mag)")
ax2.axes.yaxis.set_ticklabels([])
for ax in [ax1, ax2]:
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    ax.set_xlabel("$H - K_S$ (mag)")
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    # Plot contour of CF density
    # ax.scatter(xdata, ydata, s=1, alpha=0.2, lw=0, c="gray")
    ax.contour(hist.T, extent=(xrange[0], xrange[1], yrange[0], yrange[1]), levels=[0.02, 0.05, 0.1, 0.2, 0.5],
               colors="black", linewidths=1, alpha=0.3)

    # Extinction vector
    # ax.arrow(0.7, 0.3, features_extinction[1] - features_extinction[2], features_extinction[0] - features_extinction[1],
    #          head_width=0.02, head_length=0.05, fc="k", ec="k", length_includes_head=True)

# Save
plt.savefig("/Users/Antares/Dropbox/Projects/PNICER/Paper/Results/nicer_pnicer.png", bbox_inches="tight")