# ----------------------------------------------------------------------
# Import stuff
import brewer2mpl
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import wcsaxes

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import wcs
from astropy import units as u
from pnicer import Magnitudes, mp_kde
from matplotlib.pyplot import GridSpec
from matplotlib.ticker import MultipleLocator
import warnings

# ----------------------------------------------------------------------
# Change defaults
matplotlib.rcParams.update({'font.size': 13})


# ----------------------------------------------------------------------
# Define file paths
science_path = "/Users/Antares/Dropbox/Data/Orion/VISION/Catalog/VISION_+_Spitzer_s_noYSO.fits"
control_path = "/Users/Antares/Dropbox/Data/Orion/VISION/Catalog/VISION_CF+_Spitzer_s.fits"
results_path = "/Users/Antares/Dropbox/Projects/PNICER/Paper/Results/"
extinction_herschel_path = "/Users/Antares/Dropbox/Data/Orion/Other/Orion_Planck_Herschel_fit_wcs_AK_OriA.fits"
irac1_coverage_path = "/Users/Antares/Dropbox/Data/Orion/Other/Spitzer_Orion_IRAC1_coverage.fits"
irac2_coverage_path = "/Users/Antares/Dropbox/Data/Orion/Other/Spitzer_Orion_IRAC2_coverage.fits"


# ----------------------------------------------------------------------
# Load colorbrewer colormap
cmap1 = brewer2mpl.get_map("RdBu", "Diverging", number=5, reverse=False).get_mpl_colormap(N=21, gamma=1)
cmap2 = brewer2mpl.get_map("RdBu", "Diverging", number=5, reverse=False).get_mpl_colormap(N=15, gamma=1)


# ----------------------------------------------------------------------
# Load catalog data
skip = 1
science_dummy = fits.open(science_path)[1].data
control_dummy = fits.open(control_path)[1].data

science_glon = science_dummy["GLON"][::skip]
science_glat = science_dummy["GLAT"][::skip]

control_glon = control_dummy["GLON"][::skip]
control_glat = control_dummy["GLAT"][::skip]

features_names = ["J", "H", "Ks", "IRAC1", "IRAC2"]
errors_names = ["J_err", "H_err", "Ks_err", "IRAC1_err", "IRAC2_err"]
features_extinction = [2.5, 1.55, 1.0, 0.636, 0.54]

n_features = 5

# Photometry
science_data = [science_dummy[n][::skip] for n in features_names[:n_features]]
control_data = [control_dummy[n][::skip] for n in features_names[:n_features]]

# Measurement errors
science_error = [science_dummy[n][::skip] for n in errors_names[:n_features]]
control_error = [control_dummy[n][::skip] for n in errors_names[:n_features]]
features_extinction = features_extinction[:n_features]
features_names = features_names[:n_features]


# ----------------------------------------------------------------------
# Load Hershcel data
herschel_data = fits.open(extinction_herschel_path)[0].data
herschel_wcs = wcs.WCS(fits.open(extinction_herschel_path)[0].header)


# ----------------------------------------------------------------------
# Read coverage of IRAC1 data
coverage_irac1_data = fits.open(irac1_coverage_path)[0].data
coverage_irac1_header = fits.open(irac1_coverage_path)[0].header
coverage_irac1_wcs = wcs.WCS(coverage_irac1_header)

# Dimensions of coverage map
coverage_irac1_naxis1 = coverage_irac1_data.shape[1]
coverage_irac1_naxis2 = coverage_irac1_data.shape[0]


# ----------------------------------------------------------------------
# Read coverage of IRAC2 data
coverage_irac2_data = fits.open(irac2_coverage_path)[0].data
coverage_irac2_header = fits.open(irac2_coverage_path)[0].header
coverage_irac2_wcs = wcs.WCS(coverage_irac2_header)

# Dimensions of coverage map
coverage_irac2_naxis1 = coverage_irac2_data.shape[1]
coverage_irac2_naxis2 = coverage_irac2_data.shape[0]


# ----------------------------------------------------------------------
# Initialize data
science = Magnitudes(mag=science_data, err=science_error, extvec=features_extinction,
                     lon=science_glon, lat=science_glat, names=features_names).mag2color()
# control = Magnitudes(mag=control_data, err=control_error, extvec=features_extinction,
#                      lon=control_glon, lat=control_glat, names=features_names).mag2color()


# ----------------------------------------------------------------------
# Plot paramters
pixsize = 2/60
sampling = 2
kernel = "epanechnikov"


# ----------------------------------------------------------------------
# Get a WCS grid
header, lon_grid, lat_grid = science.build_wcs_grid(frame="galactic", pixsize=pixsize)
xgrid = np.vstack([lon_grid.ravel(), lat_grid.ravel()]).T

# Get aspect ratio
ar = lon_grid.shape[0] / lon_grid.shape[1]

# Convert to equatorial coodinates
grid_coo = SkyCoord(lon_grid*u.degree, lat_grid*u.degree, frame="galactic")
grid_ra = grid_coo.icrs.ra.degree
grid_dec = grid_coo.icrs.dec.degree


# ----------------------------------------------------------------------
# Evaluate densities

# Total density
data = np.vstack([science.lon, science.lat]).T
dens_tot = mp_kde(grid=xgrid, data=data, bandwidth=pixsize*sampling, shape=lon_grid.shape, kernel=kernel,
                  absolute=True, sampling=sampling)

# JH density
data = np.vstack([science.lon[science.features_masks[0]], science.lat[science.features_masks[0]]]).T
dens_j_h = mp_kde(grid=xgrid, data=data, bandwidth=pixsize*sampling, shape=lon_grid.shape, kernel=kernel,
                  absolute=True, sampling=sampling)

# HK density
data = np.vstack([science.lon[science.features_masks[1]], science.lat[science.features_masks[1]]]).T
dens_h_k = mp_kde(grid=xgrid, data=data, bandwidth=pixsize*sampling, shape=lon_grid.shape, kernel=kernel,
                  absolute=True, sampling=sampling)

# ksi1 density
data = np.vstack([science.lon[science.features_masks[2]], science.lat[science.features_masks[2]]]).T
dens_ki1 = mp_kde(grid=xgrid, data=data, bandwidth=pixsize*sampling, shape=lon_grid.shape, kernel=kernel,
                  absolute=True, sampling=sampling)

# i1i2 density
data = np.vstack([science.lon[science.features_masks[3]], science.lat[science.features_masks[3]]]).T
dens_i1i2 = mp_kde(grid=xgrid, data=data, bandwidth=pixsize*sampling, shape=lon_grid.shape, kernel=kernel,
                   absolute=True, sampling=sampling)


# ----------------------------------------------------------------------
# Get grid coverage in IRAC1 data coverage
grid_x_irac1, grid_y_irac1 = np.ceil(coverage_irac1_wcs.wcs_world2pix(grid_ra, grid_dec, 0)).astype(np.int)

# Mask values outside coverage image
grid_x_irac1[(grid_x_irac1 > coverage_irac1_data.shape[1] - 1) | (grid_x_irac1 < 0)] = 0
grid_y_irac1[(grid_y_irac1 > coverage_irac1_data.shape[0] - 1) | (grid_y_irac1 < 0)] = 0

# Retrieve coverage
grid_irac1_coverage = coverage_irac1_data[grid_y_irac1, grid_x_irac1].reshape(dens_ki1.shape)


# ----------------------------------------------------------------------
# Get grid coverage in IRAC2 data coverage
grid_x_irac2, grid_y_irac2 = np.ceil(coverage_irac2_wcs.wcs_world2pix(grid_ra, grid_dec, 0)).astype(np.int)

# Mask values outside coverage image
grid_x_irac2[(grid_x_irac2 > coverage_irac2_data.shape[1] - 1) | (grid_x_irac2 < 0)] = 0
grid_y_irac2[(grid_y_irac2 > coverage_irac2_data.shape[0] - 1) | (grid_y_irac2 < 0)] = 0

# Retrieve coverage
grid_irac2_coverage = coverage_irac2_data[grid_y_irac2, grid_x_irac2].reshape(dens_i1i2.shape)


# ----------------------------------------------------------------------
# Mask KDE maps
dens_ki1[grid_irac1_coverage < 0.0001] = np.nan
dens_i1i2[(grid_irac1_coverage < 0.0001) | (grid_irac2_coverage < 0.0001)] = np.nan


# ----------------------------------------------------------------------
# Create grid for plotting
fig = plt.figure(figsize=[15, 15 * ar])
grid = GridSpec(ncols=2, nrows=2, bottom=0.05, top=0.9, left=0.05, right=0.9, hspace=0.03, wspace=0.)

# Add colorbars
cax1 = fig.add_axes([0.0535, 0.91, 0.4183, 0.03])
cax2 = fig.add_axes([0.4785, 0.91, 0.4183, 0.03])


# ----------------------------------------------------------------------
# Plot density gains
ax_tot = plt.subplot(grid[0], projection=wcsaxes.WCS(header=header))
ax_hk = plt.subplot(grid[2], projection=wcsaxes.WCS(header=header))
ax_ki1 = plt.subplot(grid[1], projection=wcsaxes.WCS(header=header))
ax_i1i2 = plt.subplot(grid[3], projection=wcsaxes.WCS(header=header))

with warnings.catch_warnings():
    # Ignore NaN and 0 division warnings
    warnings.simplefilter("ignore")
    im_tot = ax_tot.imshow((dens_tot / dens_j_h - 1) * 100, origin="lower", interpolation="nearest",
                           cmap=cmap1, vmin=-140, vmax=140)
    ax_hk.imshow((dens_h_k / dens_j_h - 1) * 100, origin="lower", interpolation="nearest",
                 cmap=cmap1, vmin=-1.4, vmax=1.4)
    im_ki1 = ax_ki1.imshow((dens_ki1 / dens_h_k - 1) * 100, origin="lower", interpolation="nearest",
                           cmap=cmap2, vmin=-50, vmax=50)
    ax_i1i2.imshow((dens_i1i2 / dens_ki1 - 1) * 100, origin="lower", interpolation="nearest",
                   cmap=cmap2, vmin=-0.5, vmax=0.5)


# ----------------------------------------------------------------------
# Plot contour
for ax in [ax_tot, ax_hk, ax_ki1, ax_i1i2]:
    ax.contour(herschel_data, levels=[1], colors="black", lw=2, alpha=1, transform=ax.get_transform(herschel_wcs))


# ----------------------------------------------------------------------
# Add colorbars
cbar1 = plt.colorbar(im_tot, cax=cax1, ticks=MultipleLocator(40), orientation="horizontal")
cbar2 = plt.colorbar(im_ki1, cax=cax2, ticks=MultipleLocator(20), orientation="horizontal")

for cb in [cbar1, cbar2]:
    cb.ax.xaxis.set_ticks_position("top")
    cb.set_label("Relative source density gain (%)", labelpad=-45)


# ----------------------------------------------------------------------
# Adjust axes
for ax in [ax_tot, ax_hk, ax_ki1, ax_i1i2]:

    # Zoom
    ax.set_xlim(0.1 * dens_tot.shape[1], dens_tot.shape[1] - 0.1 * dens_tot.shape[1])
    ax.set_ylim(0.1 * dens_tot.shape[0], dens_tot.shape[0] - 0.1 * dens_tot.shape[0])

    # Grab axes
    lon = ax.coords[0]
    lat = ax.coords[1]

    # Set minor ticks
    lon.set_major_formatter("d")
    lon.display_minor_ticks(True)
    lon.set_minor_frequency(4)
    lat.set_major_formatter("d")
    lat.display_minor_ticks(True)
    lat.set_minor_frequency(2)

    # Set labels
    if ax == ax_hk:
        lat.set_axislabel("                                 Galactic Latitude (°)")
    if (ax == ax_tot) | (ax == ax_hk):
        pass
        # lat.set_axislabel("Gal. Latitude (°)")
    else:
        lat.set_ticklabel_position("")
    if (ax == ax_hk) | (ax == ax_i1i2):
        lon.set_axislabel("Galactic Longitude (°)")
    else:
        # Hide tick labels
        lon.set_ticklabel_position("")


# ----------------------------------------------------------------------
# Save figure
plt.savefig(results_path + "source_gain_kde.pdf", bbox_inches="tight")
