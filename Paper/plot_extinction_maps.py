from __future__ import absolute_import, division, print_function
__author__ = "Stefan Meingast"


# ----------------------------------------------------------------------
# Import stuff
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from pnicer import Magnitudes


# ----------------------------------------------------------------------
# Define file paths
science_path = "/Users/Antares/Dropbox/Data/Orion/VISION/Catalog/VISION_+_Spitzer_s.fits"
control_path = "/Users/Antares/Dropbox/Data/Orion/VISION/Catalog/VISION_CF+_Spitzer_s.fits"


# ----------------------------------------------------------------------
# Load data
skip = 10
science_dummy = fits.open(science_path)[1].data
control_dummy = fits.open(control_path)[1].data

science_glon = science_dummy["GLON"][::skip]
science_glat = science_dummy["GLAT"][::skip]

control_glon = control_dummy["GLON"][::skip//2]
control_glat = control_dummy["GLAT"][::skip//2]


features_names = ["J", "H", "Ks", "IRAC1", "IRAC2"]
errors_names = ["J_err", "H_err", "Ks_err", "IRAC1_err", "IRAC2_err"]
features_extinction = [2.5, 1.55, 1.0, 0.636, 0.54]

n_features = 3

# Photometry
science_data = [science_dummy[n][::skip] for n in features_names[:n_features]]
control_data = [control_dummy[n][::skip//2] for n in features_names[:n_features]]

# Measurement errors
science_error = [science_dummy[n][::skip] for n in errors_names[:n_features]]
control_error = [control_dummy[n][::skip//2] for n in errors_names[:n_features]]
features_extinction = features_extinction[:n_features]
features_names = features_names[:n_features]


# ----------------------------------------------------------------------
# Initialize data
science = Magnitudes(mag=science_data, err=science_error, extvec=features_extinction,
                     lon=science_glon, lat=science_glat, names=features_names)
control = Magnitudes(mag=control_data, err=control_error, extvec=features_extinction,
                     lon=control_glon, lat=control_glat, names=features_names)

science_color = science.mag2color()
control_color = control.mag2color()


# ----------------------------------------------------------------------
# Get NICER and PNICER extinctions
ext_pnicer = science_color.pnicer(control=control_color)
ext_nicer = science.nicer(control=control)


# ----------------------------------------------------------------------
# Build extinction maps
bandwidth, metric, sampling, nicest = 5/60, "epanechnikov", 2, False
map_pnicer = ext_pnicer.build_map(bandwidth=bandwidth, metric=metric, sampling=sampling, nicest=nicest)
map_nicer = ext_nicer.build_map(bandwidth=bandwidth, metric=metric, sampling=sampling, nicest=nicest)



