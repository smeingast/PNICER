# ----------------------------------------------------------------------
# Import stuff
import wcsaxes
import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS

from astropy.io import fits
from pnicer import Magnitudes
from matplotlib.pyplot import GridSpec
from matplotlib.ticker import MultipleLocator

# ----------------------------------------------------------------------
# Define file paths
science_path = "/Users/Antares/Dropbox/Data/Orion/VISION/Catalog/VISION_+_Spitzer_s_noYSO.fits"
control_path = "/Users/Antares/Dropbox/Data/Orion/VISION/Catalog/VISION_CF+_Spitzer_s.fits"
results_path = "/Users/Antares/Dropbox/Projects/PNICER/Paper/Results/"

emap_2mass_path = "/Users/Antares/Dropbox/Data/Orion/Other/Orion_Wide_Emap_2MASS.fits"
emap_herschel_path = "/Users/Antares/Dropbox/Data/Orion/Other/Orion_Planck_Herschel_fit_wcs_AK_OriA.fits"

# ----------------------------------------------------------------------
# Load data
skip = 1
cskip = 1
n_features = 3

science_dummy = fits.open(science_path)[1].data
control_dummy = fits.open(control_path)[1].data

# Load coordinates
science_glon = science_dummy["GLON"][::skip]
science_glat = science_dummy["GLAT"][::skip]
control_glon = control_dummy["GLON"][::cskip]
control_glat = control_dummy["GLAT"][::cskip]

# Set feature parameters
features_names = ["J", "H", "Ks", "IRAC1", "IRAC2"]
errors_names = ["J_err", "H_err", "Ks_err", "IRAC1_err", "IRAC2_err"]
features_extinction = [2.5, 1.55, 1.0, 0.636, 0.54]

# Load photometry
science_data = [science_dummy[n][::skip] for n in features_names[:n_features]]
control_data = [control_dummy[n][::cskip] for n in features_names[:n_features]]

# Load measurement errors
science_error = [science_dummy[n][::skip] for n in errors_names[:n_features]]
control_error = [control_dummy[n][::cskip] for n in errors_names[:n_features]]
features_extinction = features_extinction[:n_features]
features_names = features_names[:n_features]


# ----------------------------------------------------------------------
# Read external extinction maps
emap_2mass = fits.open(emap_2mass_path)[0].data
emap_2mass_header = fits.open(emap_2mass_path)[0].header

emap_herschel = fits.open(emap_herschel_path)[0].data
emap_herschel_header = fits.open(emap_herschel_path)[0].header


# ----------------------------------------------------------------------
# Initialize data
science = Magnitudes(mag=science_data, err=science_error, extvec=features_extinction,
                     lon=science_glon, lat=science_glat, names=features_names)
control = Magnitudes(mag=control_data, err=control_error, extvec=features_extinction,
                     lon=control_glon, lat=control_glat, names=features_names)

# science_color = science.mag2color()
# control_color = control.mag2color()


# ----------------------------------------------------------------------
# Get NICER and PNICER extinctions
# ext_pnicer = science_color.pnicer(control=control_color)
ext_pnicer = science.pnicer(control=control, add_colors=True)
ext_nicer = science.nicer(control=control)

# Save extinction data
# ext_pnicer.save_fits(path="/Users/Antares/Desktop/pnicer_table.fits")
# ext_nicer.save_fits(path="/Users/Antares/Desktop/nicer_table.fits")


# ----------------------------------------------------------------------
# Build extinction maps
bandwidth, metric, sampling, nicest, fwhm = 2/60, "gaussian", 2, True, True
map1 = ext_pnicer.build_map(bandwidth=bandwidth, metric=metric, sampling=sampling, nicest=nicest, use_fwhm=fwhm)
map1.save_fits(path="/Users/Antares/Desktop/pnicer_map.fits")
map2 = ext_nicer.build_map(bandwidth=bandwidth, metric=metric, sampling=sampling, nicest=nicest, use_fwhm=fwhm)
map2.save_fits(path="/Users/Antares/Desktop/nicer_map.fits")


# ----------------------------------------------------------------------
# Create figure
fig = plt.figure(figsize=[15, 4])
grid = GridSpec(ncols=2, nrows=2, bottom=0.05, top=0.95, left=0.05, right=0.9, hspace=0.05, wspace=0.05)

# Add colorbar axes
cax = fig.add_axes([0.91, 0.05, 0.02, 0.89])

# Set defaults
vmin, vmax = 0, 2
edges = [[216.2, 206.7], [-20.5, -18.05]]

# Plot 1st results
ax1 = plt.subplot(grid[0], projection=wcsaxes.WCS(map1.fits_header))
# Extinction map
im1 = ax1.imshow(map1.map, origin="lower", interpolation="nearest", cmap="binary", vmin=vmin, vmax=vmax)

# Plot 2nd results
ax2 = plt.subplot(grid[1], projection=wcsaxes.WCS(map2.fits_header))
# Extinction map
im2 = ax2.imshow(map2.map, origin="lower", interpolation="nearest", cmap="binary", vmin=vmin, vmax=vmax)

# Plot 2MASS emap
ax3 = plt.subplot(grid[2], projection=wcsaxes.WCS(emap_2mass_header))
# Extinction map
im3 = ax3.imshow(emap_2mass, origin="lower", interpolation="nearest", cmap="binary", vmin=vmin, vmax=vmax)

# Plot Herschel emap
ax4 = plt.subplot(grid[3], projection=wcsaxes.WCS(emap_herschel_header))
# Extinction map
im4 = ax4.imshow(emap_herschel, origin="lower", interpolation="nearest", cmap="binary", vmin=vmin, vmax=vmax)

# Plot colorbar
plt.colorbar(im1, cax=cax, ticks=MultipleLocator(0.2), label="$A_K$")

# Set common properties
for ax, header in zip([ax1, ax2, ax3, ax4], [map1.fits_header, map2.fits_header,
                                             emap_2mass_header, emap_herschel_header]):

    # Set limits
    wcs = WCS(header)
    pix = np.floor(wcs.wcs_world2pix(edges[0], edges[1], 0)).astype(np.int)
    ax.set_xlim(pix[0][0], pix[0][1])
    ax.set_ylim(pix[1][0], pix[1][1])

    lon = ax.coords[0]
    lat = ax.coords[1]

    # Set axis labels
    if ax in [ax3, ax4]:
        lon.set_axislabel("Galactic longitude (°)")
    else:
        lon.set_ticklabel_position("")

    if ax in [ax1, ax3]:
        lat.set_axislabel("Galactic latitude (°)")
    else:
        lat.set_ticklabel_position("")

    # Ticker
    lon.set_major_formatter("d")
    lon.display_minor_ticks(True)
    lon.set_minor_frequency(4)
    lat.set_major_formatter("d")
    lat.display_minor_ticks(True)
    lat.set_minor_frequency(2)


# ----------------------------------------------------------------------
# Save figure
plt.savefig(results_path + "extinction_maps.pdf", bbox_inches="tight")
