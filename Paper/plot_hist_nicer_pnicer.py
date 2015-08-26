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
features_names = ["J", "H", "Ks", "IRAC1", "IRAC2"]
errors_names = ["J_err", "H_err", "Ks_err", "IRAC1_err", "IRAC2_err"]

# Define extinction
features_extinction = [2.5, 1.55, 1.0, 0.636, 0.54]


# ----------------------------------------------------------------------
# Load data into lists for PNICER
science_data = [science_dummy[n] for n in features_names]
control_data = [control_dummy[n] for n in features_names]
science_error = [science_dummy[n] for n in errors_names]
control_error = [control_dummy[n] for n in errors_names]

dens_nicer, dens_pnicer = [], []
res = 0.02
sampling = 4
grid_kde = np.arange(start=-1, stop=1, step=res/sampling, dtype=np.double)[:, np.newaxis]
for i in range(3, 4):

    print(features_names[:i])

    # Initialize data
    science = Magnitudes(mag=science_data[:i], err=science_error[:i], extvec=features_extinction[:i],
                         lon=science_glon, lat=science_glat, names=features_names[:i])
    control = Magnitudes(mag=control_data[:i], err=control_error[:i], extvec=features_extinction[:i],
                         lon=control_glon, lat=control_glat, names=features_names[:i])

    # Run PNICER and NICER on control field
    ext_pnicer = control.pnicer(control=control, bin_ext=0.02, bin_grid=0.1, use_color=True).extinction
    ext_nicer = control.nicer(control=control).extinction

    # Filter NaNs
    ext_nicer = ext_nicer[np.isfinite(ext_nicer)][:, np.newaxis]
    ext_pnicer = ext_pnicer[np.isfinite(ext_pnicer)][:, np.newaxis]

    # Do KDE
    dens_nicer.append(mp_kde(grid=grid_kde, data=ext_nicer, bandwidth=res, sampling=sampling,
                             shape=None, kernel="epanechnikov", absolute=True))
    dens_pnicer.append(mp_kde(grid=grid_kde, data=ext_pnicer, bandwidth=res, sampling=sampling,
                              shape=None, kernel="epanechnikov", absolute=True))


# ----------------------------------------------------------------------
# Plot
fig = plt.figure(figsize=[12, 6])
grid_plt= GridSpec(ncols=1, nrows=1, bottom=0.05, top=0.95, left=0.05, right=0.95, hspace=0.1, wspace=0.1)
ax = plt.subplot(grid_plt[0])

ncolor = ["#cb181d", "#a50f15", "#67000d"]
pcolor = ["#6baed6", "#2171b5", "#08306b"]
lw, alpha = 2, 1
for n, p, nc, pc in zip(dens_nicer, dens_pnicer, ncolor, pcolor):
    ax.plot(grid_kde, n, lw=lw, alpha=alpha, color=nc)
    ax.plot(grid_kde, p, lw=lw, alpha=alpha, color=pc)

# ax.set_yscale("log")

# edges = np.arange(-1, 1, res)
# ax.hist(ext_nicer, bins=edges, range=(-1, 1), alpha=0.5, color="red")
# ax.hist(ext_pnicer, bins=edges, range=(-1, 1), alpha=0.5, color="blue")

# Annotate the standard deviations
# ax.annotate("NICER std = " + str(np.around(np.std(ext_nicer), 2)) +
#             "\nPNICER std = " + str(np.around(np.std(ext_pnicer), 2)),
#             xy=(0.05, 0.95), xycoords="axes fraction", ha="left", va="top")

plt.show()