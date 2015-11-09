# ----------------------------------------------------------------------
# Import stuff
from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from pnicer import Magnitudes


# ----------------------------------------------------------------------
# Define file paths
science_path = "/home/antares/Dropbox/Data/Orion/VISION/Catalog/VISION_+_Spitzer_s_noYSO.fits"
control_path = "/home/antares/Dropbox/Data/Orion/VISION/Catalog/VISION_CF+_Spitzer_s.fits"


# ----------------------------------------------------------------------
# Load data
sskip = 1
cskip = 1

science_dummy = fits.open(science_path)[1].data
control_dummy = fits.open(control_path)[1].data

science_glon = science_dummy["GLON"][::sskip]
science_glat = science_dummy["GLAT"][::sskip]

control_glon = control_dummy["GLON"][::cskip]
control_glat = control_dummy["GLAT"][::cskip]


features_names = ["J", "H", "Ks", "IRAC1", "IRAC2"]
errors_names = ["J_err", "H_err", "Ks_err", "IRAC1_err", "IRAC2_err"]
features_extinction = [2.5, 1.55, 1.0, 0.636, 0.54]

n_features = 5

# Photometry
science_data = [science_dummy[n][::sskip] for n in features_names[:n_features]]
science_error = [science_dummy[n][::sskip] for n in errors_names[:n_features]]


control_data = [control_dummy[n][::cskip] for n in features_names[:n_features]]
control_error = [control_dummy[n][::cskip] for n in errors_names[:n_features]]
features_extinction = features_extinction[:n_features]
features_names = features_names[:n_features]


# ----------------------------------------------------------------------
# Initialize data
science = Magnitudes(mag=science_data, err=science_error, extvec=features_extinction,
                     lon=science_glon, lat=science_glat, names=features_names)
control = Magnitudes(mag=control_data, err=control_error, extvec=features_extinction,
                     lon=control_glon, lat=control_glat, names=features_names)


# Test PNICER
test = science.pnicer(control=control, add_colors=True)
# test = science.mag2color().pnicer(control=control.mag2color())

# Test NICER
ext = science.nicer(control=control).extinction

exit()


# science.plot_combinations_kde()
# science.plot_spatial_kde(frame="galactic", pixsize=2/60, skip=5)
# science.plot_spatial_kde_gain(frame="galactic", pixsize=2/60, skip=5)

# # Get extinction measurements with NICER and/or PNICER
ext_pnicer = science.pnicer(control=control, sampling=2)
exit()
ext_nicer = science.nicer(control=control)
ext_pnicer.save_fits(path="/Users/Antares/Desktop/Orion_table_pnicer.fits")
ext_nicer.save_fits(path="/Users/Antares/Desktop/Orion_table_nicer.fits")

exit()
# ext_pnicer = control.pnicer(control=control, sampling=2, use_color=True)
# ext_nicer = control.nicer(control=control, all_features=False)

# ext_pnicer.save_fits(path="/Users/Antares/Desktop/CF_table_pnicer.fits")
# ext_nicer.save_fits(path="/Users/Antares/Desktop/CF_table_nicer.fits")

# control.plot_kde_extinction_combinations(path="/Users/Antares/Desktop/test.pdf")
