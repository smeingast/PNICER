# ----------------------------------------------------------------------
# Import stuff
import wcsaxes
import brewer2mpl
# import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from pnicer import Magnitudes
from matplotlib.pyplot import GridSpec


# ----------------------------------------------------------------------
# Define file paths
science_path = "/Users/Antares/Dropbox/Data/Orion/VISION/Catalog/VISION_+_Spitzer_s.fits"
control_path = "/Users/Antares/Dropbox/Data/Orion/VISION/Catalog/VISION_CF+_Spitzer_s.fits"
results_path = "/Users/Antares/Dropbox/Projects/PNICER/Paper/Results/"

emap_2mass_path = "/Users/Antares/Dropbox/Data/Orion/Other/Orion_Wide_Emap_2MASS.fits"
emap_herschel_path = "/Users/Antares/Dropbox/Data/Orion/Other/Orion_Planck_Herschel_fit_wcs_AK_OriA.fits"


# ----------------------------------------------------------------------
# Load colorbrewer colormap
cmap = brewer2mpl.get_map("RdYlBu", "Diverging", number=9, reverse=False).get_mpl_colormap(N=10, gamma=0.7)


# ----------------------------------------------------------------------
# Load data

# Set feature parameters
features_names = ["J", "H", "Ks", "IRAC1", "IRAC2"]
errors_names = ["J_err", "H_err", "Ks_err", "IRAC1_err", "IRAC2_err"]
features_extinction = [2.5, 1.55, 1.0, 0.636, 0.54]

# Open files and load data
science_dummy = fits.open(science_path)[1].data
control_dummy = fits.open(control_path)[1].data

# Load coordinates
science_glon = science_dummy["GLON"]
science_glat = science_dummy["GLAT"]
control_glon = control_dummy["GLON"]
control_glat = control_dummy["GLAT"]

# Set map parameters
bandwidth, metric, sampling, nicest, fwhm = 10/60, "epanechnikov", 2, False, False

# ----------------------------------------------------------------------
# Loop over number of features
nicer, pnicer = [], []
for n_features in range(3, 6):

    # Load photometry
    sdata = [science_dummy[n] for n in features_names[:n_features]]
    cdata = [control_dummy[n] for n in features_names[:n_features]]

    # Load measurement errors
    serror = [science_dummy[n] for n in errors_names[:n_features]]
    cerror = [control_dummy[n] for n in errors_names[:n_features]]

    # Feature extinction and names
    fext = features_extinction[:n_features]
    fname = features_names[:n_features]

    # Initialize data
    science = Magnitudes(mag=sdata, err=serror, extvec=fext,  lon=science_glon, lat=science_glat, names=fname)
    control = Magnitudes(mag=cdata, err=cerror, extvec=fext, lon=control_glon, lat=control_glat, names=fname)

    print(science.features_names)

    # science_color = science.mag2color()
    # control_color = control.mag2color()

    # Get NICER and PNICER extinctions
    ext_pnicer = science.pnicer(control=control, add_colors=False)
    ext_nicer = science.nicer(control=control)

    # Build extinction maps
    nicer.append(ext_nicer.build_map(bandwidth=bandwidth, metric=metric,
                                     sampling=sampling, nicest=nicest, use_fwhm=fwhm))
    pnicer.append(ext_pnicer.build_map(bandwidth=bandwidth, metric=metric,
                                       sampling=sampling, nicest=nicest, use_fwhm=fwhm))

# Create figure
fig = plt.figure(figsize=[20, 10])
grid = GridSpec(ncols=2, nrows=2, bottom=0.05, top=0.95, left=0.05, right=0.95, hspace=0.1, wspace=0.1)

# Make plots for PNICER
ax1 = plt.subplot(grid[0], projection=wcsaxes.WCS(nicer[0].fits_header))
ax1.imshow((pnicer[1].map / pnicer[0].map) - 1, origin="lower", interpolation="nearest", cmap=cmap, vmin=-0.5, vmax=0.5)
ax2 = plt.subplot(grid[2], projection=wcsaxes.WCS(nicer[0].fits_header))
ax2.imshow((pnicer[2].map / pnicer[0].map) - 1, origin="lower", interpolation="nearest", cmap=cmap, vmin=-0.5, vmax=0.5)

# Make plots for NICER
ax3 = plt.subplot(grid[1], projection=wcsaxes.WCS(nicer[0].fits_header))
ax3.imshow((nicer[1].map / nicer[0].map) - 1, origin="lower", interpolation="nearest", cmap=cmap, vmin=-0.5, vmax=0.5)
ax4 = plt.subplot(grid[3], projection=wcsaxes.WCS(nicer[0].fits_header))
ax4.imshow((nicer[2].map / nicer[0].map) - 1, origin="lower", interpolation="nearest", cmap=cmap, vmin=-0.5, vmax=0.5)

plt.show()


