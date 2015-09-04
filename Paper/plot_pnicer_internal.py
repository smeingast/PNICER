from __future__ import absolute_import, division, print_function
__author__ = 'Stefan Meingast'


# ----------------------------------------------------------------------
# Import stuff
import numpy as np
import brewer2mpl
import matplotlib.pyplot as plt

from astropy import wcs
from astropy.io import fits
from pnicer import Magnitudes


# ----------------------------------------------------------------------
# Define file paths
science_path = "/Users/Antares/Dropbox/Data/Orion/VISION/Catalog/VISION_+_Spitzer_s.fits"
control_path = "/Users/Antares/Dropbox/Data/Orion/VISION/Catalog/VISION_CF+_Spitzer_s.fits"
results_path = "/Users/Antares/Dropbox/Projects/PNICER/Paper/Results/"
extinction_herschel_path = "/Users/Antares/Dropbox/Data/Orion/Other/Orion_Planck_Herschel_fit_wcs_AK_OriA.fits"


# ----------------------------------------------------------------------
# Load colorbrewer colormap
cmap = brewer2mpl.get_map('RdBu', 'Diverging', number=5, reverse=False).get_mpl_colormap(N=20, gamma=1)


# ----------------------------------------------------------------------
# Read 2MASS extinction data
herschel_data = fits.open(extinction_herschel_path)[0].data
herschel_wcs = wcs.WCS(fits.open(extinction_herschel_path)[0].header)


# ----------------------------------------------------------------------
# Load catalog data
science_dummy = fits.open(science_path)[1].data
control_dummy = fits.open(control_path)[1].data

science_glon = science_dummy["GLON"]
science_glat = science_dummy["GLAT"]

control_glon = control_dummy["GLON"]
control_glat = control_dummy["GLAT"]


features_names = ["J", "H", "Ks", "IRAC1", "IRAC2"]
errors_names = ["J_err", "H_err", "Ks_err", "IRAC1_err", "IRAC2_err"]
features_extinction = [2.5, 1.55, 1.0, 0.636, 0.54]

n_features = 5

# Photometry
science_data = [science_dummy[n] for n in features_names[:n_features]]
control_data = [control_dummy[n] for n in features_names[:n_features]]

# Measurement errors
science_error = [science_dummy[n] for n in errors_names[:n_features]]
control_error = [control_dummy[n] for n in errors_names[:n_features]]
features_extinction = features_extinction[:n_features]
features_names = features_names[:n_features]


# ----------------------------------------------------------------------
# Initialize data
science = Magnitudes(mag=science_data, err=science_error, extvec=features_extinction,
                     lon=science_glon, lat=science_glat, names=features_names).mag2color()


# ----------------------------------------------------------------------
# Plot spatial source density gain
science.plot_spatial_kde_gain(frame="galactic", pixsize=2/60, path=results_path + "source_gain_kde.pdf",
                              kernel="epanechnikov", skip=4, cmap=cmap)

