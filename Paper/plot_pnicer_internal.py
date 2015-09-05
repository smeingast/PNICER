from __future__ import absolute_import, division, print_function


# ----------------------------------------------------------------------
# Import stuff
import brewer2mpl

from astropy import wcs
from astropy.io import fits
from pnicer import Magnitudes


# ----------------------------------------------------------------------
# Define file paths
science_path = "/Users/Antares/Dropbox/Data/Orion/VISION/Catalog/VISION_+_Spitzer_s_noYSO.fits"
control_path = "/Users/Antares/Dropbox/Data/Orion/VISION/Catalog/VISION_CF+_Spitzer_s.fits"
results_path = "/Users/Antares/Dropbox/Projects/PNICER/Paper/Results/"
extinction_herschel_path = "/Users/Antares/Dropbox/Data/Orion/Other/Orion_Planck_Herschel_fit_wcs_AK_OriA.fits"


# ----------------------------------------------------------------------
# Load colorbrewer colormap
# cmap = brewer2mpl.get_map("RdBu", "Diverging", number=5, reverse=False).get_mpl_colormap(N=20, gamma=1)
cmap = brewer2mpl.get_map("Blues", "Sequential", number=9, reverse=False).get_mpl_colormap(N=100, gamma=0.8)


# ----------------------------------------------------------------------
# Read 2MASS extinction data
herschel_data = fits.open(extinction_herschel_path)[0].data
herschel_wcs = wcs.WCS(fits.open(extinction_herschel_path)[0].header)


# ----------------------------------------------------------------------
# Load catalog data
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

n_features = 5

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


# ----------------------------------------------------------------------
# Plot spatial source density gain
science.mag2color().plot_spatial_kde_gain(frame="galactic", pixsize=1/60, path=results_path + "source_gain_kde.pdf",
                                          kernel="gaussian", skip=1, cmap=cmap, contour=[herschel_data, herschel_wcs])


# ----------------------------------------------------------------------
# KDE plot of combinations
# science.mag2color().plot_combinations_kde(path=results_path + "combinations_kde.pdf", grid_bw=0.05, cmap=cmap)


# ----------------------------------------------------------------------
science_color = science.mag2color()
science_color.pnicer(control=control.mag2color(), sampling=2, kernel="epanechnikov")
science_color.plot_kde_extinction_combinations()
