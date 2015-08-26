from __future__ import absolute_import, division, print_function
__author__ = "Stefan Meingast"

# ----------------------------------------------------------------------
# Import stuff
import brewer2mpl
import numpy as np
import matplotlib.pyplot as plt

from pnicer import Magnitudes
from astropy.io import fits
from matplotlib.pyplot import GridSpec
from matplotlib.ticker import MultipleLocator
from pnicer import mp_kde


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
# features_names = ["J", "H", "Ks", "IRAC1", "IRAC2"]
# features_names = ["J", "H", "Ks", "IRAC1"]
features_names = ["J", "H", "Ks"]
# errors_names = ["J_err", "H_err", "Ks_err", "IRAC1_err", "IRAC2_err"]
# errors_names = ["J_err", "H_err", "Ks_err", "IRAC1_err"]
errors_names = ["J_err", "H_err", "Ks_err"]

# Define extinction
# features_extinction = [2.5, 1.55, 1.0, 0.636, 0.54]
# features_extinction = [2.5, 1.55, 1.0, 0.636]
features_extinction = [2.5, 1.55, 1.0]


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
# Run PNICER and NICER on control field
# ext_pnicer = control.pnicer(control=control, bin_ext=0.02, bin_grid=0.02, use_color=False).extinction
ext_nicer = control.nicer(control=control).extinction

# Filter NaNs
ext_nicer = ext_nicer[np.isfinite(ext_nicer)][:, np.newaxis]


# ----------------------------------------------------------------------
# Do KDE
res = 0.05
grid_kde = np.arange(start=-1, stop=1, step=res/2, dtype=np.double)[:, np.newaxis]

dens = mp_kde(grid=grid_kde, data=ext_nicer, bandwidth=res, shape=None, kernel="epanechnikov", absolute=True)
# norm = mp_kde(grid=grid_kde, data=grid_kde, bandwidth=res, shape=None, kernel="epanechnikov")
#
# dens /= norm


# ----------------------------------------------------------------------
# Plot
fig = plt.figure(figsize=[15, 10])
grid_plt= GridSpec(ncols=1, nrows=1, bottom=0.05, top=0.95, left=0.05, right=0.95, hspace=0.1, wspace=0.1)
ax = plt.subplot(grid_plt[0])

ax.plot(grid_kde, dens, lw=2, alpha=0.5)
edges = np.arange(-1, 1, res / 2)
ax.hist(ext_nicer, bins=edges, range=(-1, 1))

plt.show()