from __future__ import absolute_import, division, print_function
__author__ = "Stefan Meingast"


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


# ----------------------------------------------------------------------
# Load data
skip = 1
science_dummy = fits.open(science_path)[1].data
control_dummy = fits.open(control_path)[1].data

# Coordinates
science_glon = science_dummy["GLON"][::skip]
science_glat = science_dummy["GLAT"][::skip]
control_glon = control_dummy["GLON"]
control_glat = control_dummy["GLAT"]

# Definitions
features_names = ["J", "H", "Ks", "IRAC1", "IRAC2"]
errors_names = ["J_err", "H_err", "Ks_err", "IRAC1_err", "IRAC2_err"]
features_extinction = [2.5, 1.55, 1.0, 0.636, 0.54]

# Photometry
science_data = [science_dummy[n][::skip] for n in features_names]
science_error = [science_dummy[n][::skip] for n in errors_names]
control_data = [control_dummy[n] for n in features_names]
control_error = [control_dummy[n] for n in errors_names]


# ----------------------------------------------------------------------
# Initialize data with best combination
science = Magnitudes(mag=science_data, err=science_error, extvec=features_extinction,
                     lon=science_glon, lat=science_glat, names=features_names)
control = Magnitudes(mag=control_data, err=control_error, extvec=features_extinction,
                     lon=control_glon, lat=control_glat, names=features_names)


# ----------------------------------------------------------------------
#
def get_covar(xi, yi):
    return np.nansum((xi - np.mean(xi) * (yi - np.mean(yi)))) / len(xi)
