from __future__ import absolute_import, division, print_function
__author__ = "Stefan Meingast"


# ----------------------------------------------------------------------
# Import stuff
import time
import numpy as np
import brewer2mpl
import matplotlib.pyplot as plt

from astropy.io import fits
from pnicer import Magnitudes
from matplotlib.pyplot import GridSpec
# from matplotlib.ticker import MultipleLocator


# ----------------------------------------------------------------------
# Define file paths
science_path = "/Users/Antares/Dropbox/Data/Orion/VISION/Catalog/VISION_+_Spitzer_s_noYSO.fits"
control_path = "/Users/Antares/Dropbox/Data/Orion/VISION/Catalog/VISION_CF+_Spitzer_s.fits"
results_path = "/Users/Antares/Dropbox/Projects/PNICER/Paper/Results/"


# ----------------------------------------------------------------------
# Load colorbrewer colormap
cmap_blues = brewer2mpl.get_map("Blues", "Sequential", number=9, reverse=False).get_mpl_colormap(N=100, gamma=0.7)
cmap_reds = brewer2mpl.get_map("Reds", "Sequential", number=9, reverse=False).get_mpl_colormap(N=100, gamma=0.7)


# ----------------------------------------------------------------------
# Define features
features_names = ["J", "H", "Ks", "IRAC1", "IRAC2"]
errors_names = ["J_err", "H_err", "Ks_err", "IRAC1_err", "IRAC2_err"]
features_extinction = [2.5, 1.55, 1.0, 0.636, 0.54]

science_dummy = fits.open(science_path)[1].data
control_dummy = fits.open(control_path)[1].data

nmin, nmax, inc = 1, 5000, 50

# ----------------------------------------------------------------------
# Create figure
fig = plt.figure(figsize=[6, 5])
grid = GridSpec(ncols=1, nrows=2, bottom=0.05, top=0.95, left=0.05, right=0.95, hspace=0)
ax0, ax1 = plt.subplot(grid[1]), plt.subplot(grid[0])

linestyles = ["solid", "dashed", "dotted"]
for n_features, ls in zip(range(3, 6, 1), linestyles):

    # ----------------------------------------------------------------------
    # Loop over different skip indices while keeping the number of control field features constant
    t_nicer_science, t_pnicer_science, ns_science, nc_science = [], [], [], []
    for s in range(nmin, nmax, inc):

        c = 1
        # Photometry
        science_data = [science_dummy[n][::s] for n in features_names[:n_features]]
        control_data = [control_dummy[n][::c] for n in features_names[:n_features]]

        # Measurement errors
        science_error = [science_dummy[n][::s] for n in errors_names[:n_features]]
        control_error = [control_dummy[n][::c] for n in errors_names[:n_features]]
        features_ext = features_extinction[:n_features]
        features_n = features_names[:n_features]

        # Initialize data
        science = Magnitudes(mag=science_data, err=science_error, extvec=features_ext, names=features_n)
        control = Magnitudes(mag=control_data, err=control_error, extvec=features_ext, names=features_n)

        science_color = science.mag2color()
        control_color = control.mag2color()

        # Determine number counts
        ns_science.append(science.n_data)
        nc_science.append(control.n_data)

        # If number of science sources is below 200, skip
        if ns_science[-1] < 200:
            t_nicer_science.append(np.nan)
            t_pnicer_science.append(np.nan)
            continue

        # Time NICER
        tstart = time.time()
        science.nicer(control=control)
        t_nicer_science.append(time.time() - tstart)

        # Time PNICER
        tstart = time.time()
        science_color.pnicer(control=control_color)
        t_pnicer_science.append(time.time() - tstart)

    # Loop over different skip indices while keeping the number of control field features constant
    t_nicer_control, t_pnicer_control, ns_control, nc_control = [], [], [], []
    for c in range(nmin, nmax, inc):

        s = 10

        # Photometry
        science_data = [science_dummy[n][::s] for n in features_names[:n_features]]
        control_data = [control_dummy[n][::c] for n in features_names[:n_features]]

        # Measurement errors
        science_error = [science_dummy[n][::s] for n in errors_names[:n_features]]
        control_error = [control_dummy[n][::c] for n in errors_names[:n_features]]
        features_ext = features_extinction[:n_features]
        features_n = features_names[:n_features]

        # Initialize data
        science = Magnitudes(mag=science_data, err=science_error, extvec=features_ext, names=features_n)
        control = Magnitudes(mag=control_data, err=control_error, extvec=features_ext, names=features_n)

        science_color = science.mag2color()
        control_color = control.mag2color()

        # Determine number counts
        ns_control.append(science.n_data)
        nc_control.append(control.n_data)

        # If number of CF sources is below 200, skip
        if nc_control[-1] < 200:
            t_nicer_control.append(np.nan)
            t_pnicer_control.append(np.nan)
            continue

        # Time NICER
        tstart = time.time()
        science.nicer(control=control)
        t_nicer_control.append(time.time() - tstart)

        # Time PNICER
        tstart = time.time()
        science_color.pnicer(control=control_color)
        t_pnicer_control.append(time.time() - tstart)

    y0_range = [1E-2, 3E1]
    y1_range = [1E-1, 3E1]

    # This would plot a aline collection with variable colors
    # import matplotlib.colors as colors
    # from matplotlib.collections import LineCollection
    #
    # points = np.array([ns_science, t_nicer_science]).T.reshape(-1, 1, 2)
    # segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # lc = LineCollection(segments, cmap=cmap_reds,
    #                     norm=colors.LogNorm(y0_range[0], y0_range[1]), linewidths=2, array=np.array(t_nicer_science))
    # ax0.add_collection(lc)
    #
    # points = np.array([ns_science, t_pnicer_science]).T.reshape(-1, 1, 2)
    # segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # lc = LineCollection(segments, cmap=cmap_blues,
    #                     norm=colors.LogNorm(y0_range[0], y0_range[1]), linewidths=2, array=np.array(t_pnicer_science))
    # ax0.add_collection(lc)
    #
    # points = np.array([nc_control, t_nicer_control]).T.reshape(-1, 1, 2)
    # segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # lc = LineCollection(segments, cmap=cmap_reds,
    #                     norm=colors.LogNorm(y1_range[0], y1_range[1]), linewidths=2, array=np.array(t_nicer_control))
    # ax1.add_collection(lc)
    #
    # points = np.array([nc_control, t_pnicer_control]).T.reshape(-1, 1, 2)
    # segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # lc = LineCollection(segments, cmap=cmap_blues,
    #                     norm=colors.LogNorm(y1_range[0], y1_range[1]), linewidths=2, array=np.array(t_pnicer_control))
    # ax1.add_collection(lc)

    # Plot results
    alpha, lw = 0.7, 2
    ax0.plot(ns_science, t_nicer_science, lw=lw, color="#d53e4f", linestyle=ls, alpha=alpha)
    ax0.plot(ns_science, t_pnicer_science, lw=lw, color="#3288bd", linestyle=ls, alpha=alpha)
    ax1.plot(nc_control, t_nicer_control, lw=lw, color="#d53e4f", linestyle=ls, alpha=alpha, label="NICER")
    ax1.plot(nc_control, t_pnicer_control, lw=lw, color="#3288bd", linestyle=ls, alpha=alpha, label="PNICER")
    # ax0.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    # ax1.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    # ax1.xaxis.get_major_formatter().set_powerlimits((0, 1))

    # Set common properties
    for ax in [ax0, ax1]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(100, 1E6)

    ax0.set_ylim(y0_range)
    ax1.set_ylim(y1_range)

    # Place ticks and labels at top
    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_position("top")

    # Upon first iteration
    if n_features == 3:

        # Set legend
        ax1.legend(loc=2, frameon=False, ncol=2, fontsize=12, columnspacing=1)

        # Labels
        ax0.set_xlabel("Number of science field sources")
        ax1.set_xlabel("Number of control field sources")
        ax0.set_ylabel("time (s)")
        ax1.set_ylabel("time (s)")


# Save
plt.savefig(results_path + "performance.pdf", bbox_inches="tight")
