from __future__ import absolute_import, division, print_function
__author__ = "Stefan Meingast"


# ----------------------------------------------------------------------
# Import stuff
import brewer2mpl
import numpy as np
import matplotlib
import time
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


# ----------------------------------------------------------------------
# Load colorbrewer colormap
cmap = brewer2mpl.get_map("Blues", "Sequential", number=9, reverse=False).get_mpl_colormap(N=100, gamma=0.7)


# ----------------------------------------------------------------------
# Loop over different skip indices
for s in range(2, 21, 2):

    c = s

    science_dummy = fits.open(science_path)[1].data
    control_dummy = fits.open(control_path)[1].data

    science_glon = science_dummy["GLON"][::s]
    science_glat = science_dummy["GLAT"][::s]

    control_glon = control_dummy["GLON"][::c]
    control_glat = control_dummy["GLAT"][::c]

    features_names = ["J", "H", "Ks", "IRAC1", "IRAC2"]
    errors_names = ["J_err", "H_err", "Ks_err", "IRAC1_err", "IRAC2_err"]
    features_extinction = [2.5, 1.55, 1.0, 0.636, 0.54]

    n_features = 5

    # Photometry
    science_data = [science_dummy[n][::s] for n in features_names[:n_features]]
    control_data = [control_dummy[n][::c] for n in features_names[:n_features]]

    # Measurement errors
    science_error = [science_dummy[n][::s] for n in errors_names[:n_features]]
    control_error = [control_dummy[n][::c] for n in errors_names[:n_features]]
    features_extinction = features_extinction[:n_features]
    features_names = features_names[:n_features]

    # Initialize data
    science = Magnitudes(mag=science_data, err=science_error, extvec=features_extinction,
                         lon=science_glon, lat=science_glat, names=features_names)
    control = Magnitudes(mag=control_data, err=control_error, extvec=features_extinction,
                         lon=control_glon, lat=control_glat, names=features_names)

    science_color = science.mag2color()
    control_color = control.mag2color()

    # Determine number counts
    n_science = science.n_data
    n_control = control.n_data

    # Time NICER
    tstart = time.time()
    science.nicer(control=control)
    print(n_science, n_control, time.time() - tstart)

