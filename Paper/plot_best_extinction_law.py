# ----------------------------------------------------------------------
# Import stuff
import wcsaxes
import brewer2mpl
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from pnicer import Magnitudes
from matplotlib.pyplot import GridSpec
from matplotlib.ticker import MultipleLocator

"""
THIS NEEDS TO BE FIXED IF USED AT SOME POINT!
"""


# ----------------------------------------------------------------------
# Define file paths
science_path = "/Users/Antares/Dropbox/Data/Orion/VISION/Catalog/VISION_+_Spitzer_s.fits"
control_path = "/Users/Antares/Dropbox/Data/Orion/VISION/Catalog/VISION_CF+_Spitzer_s.fits"
results_path = "/Users/Antares/Dropbox/Projects/PNICER/Paper/Results/"


# ----------------------------------------------------------------------
# Load colorbrewer colormap
cmap = brewer2mpl.get_map("YlOrRd", "Sequential", number=9, reverse=False).get_mpl_colormap(N=100, gamma=1)


# ----------------------------------------------------------------------
# Load data
skip = 2

# Set feature parameters
features_names = ["J", "H", "Ks", "IRAC1", "IRAC2"]
errors_names = ["J_err", "H_err", "Ks_err", "IRAC1_err", "IRAC2_err"]
features_extinction = [2.5, 1.55, 1.0, 0.636, 0.54]

# Open files and load data
science_dummy = fits.open(science_path)[1].data
control_dummy = fits.open(control_path)[1].data


# ----------------------------------------------------------------------
# Load data
science_data = [science_dummy[n][::skip] for n in features_names]
control_data = [control_dummy[n][::skip] for n in features_names]

# Define combined filter
scom = np.prod([np.isfinite(x) for x in science_data], axis=0, dtype=bool)
ccom = np.prod([np.isfinite(x) for x in control_data], axis=0, dtype=bool)

# Apply filter
science_data = [s[scom] for s in science_data]
control_data = [c[ccom] for c in control_data]
science_error = [science_dummy[n][::skip][scom] for n in errors_names]
control_error = [control_dummy[n][::skip][ccom] for n in errors_names]

# Load coordinates
science_glon = science_dummy["GLON"][::skip][scom]
science_glat = science_dummy["GLAT"][::skip][scom]
control_glon = control_dummy["GLON"][::skip][ccom]
control_glat = control_dummy["GLAT"][::skip][ccom]


# ----------------------------------------------------------------------
def get_std(in_ext):
    """
    Function to get deviations from incremental features
    :param in_ext:
    :return:
    """

    # Loop over number of features
    nicer, pnicer = [], []
    for n_features in range(3, 6):

        # Load photometry
        sdata = [science_data[n] for n in range(n_features)]
        cdata = [control_data[n] for n in range(n_features)]

        # Load measurement errors
        serror = [science_error[n] for n in range(n_features)]
        cerror = [control_error[n] for n in range(n_features)]

        # Feature extinction and names
        fext = in_ext[:n_features]
        fname = features_names[:n_features]

        # Initialize data
        science = Magnitudes(mag=sdata, err=serror, extvec=fext,  lon=science_glon, lat=science_glat, names=fname)
        control = Magnitudes(mag=cdata, err=cerror, extvec=fext, lon=control_glon, lat=control_glat, names=fname)

        # Get NICER and PNICER extinctions
        # ext_pnicer = science.mag2color().pnicer(control=control.mag2color())
        ext_nicer = science.nicer(control=control)

        # Append extinction measurements
        # pnicer.append(ext_pnicer.extinction)
        nicer.append(ext_nicer.extinction)

    # Return the standard deviations of the differences
    return np.nanstd(nicer[1] - nicer[0]), np.nanstd(nicer[2] - nicer[0])


# ----------------------------------------------------------------------
# Get evaluation grid
xai1 = xai2 = np.arange(0.2, 0.701, 0.05)
aj, ah, ak, ai1, ai2 = np.meshgrid(2.5, 1.55, 1, xai1, xai2)

x1, x2 = [], []
for j, h, k, i1, i2 in zip(aj.ravel(), ah.ravel(), ak.ravel(), ai1.ravel(), ai2.ravel()):

    # Get deviation for current grid point
    a, b = get_std(in_ext=[j, h, k, i1, i2])
    print([j, h, k, i1, i2])
    print(np.around(a, 4), np.around(b, 4))
    x1.append(a)
    x2.append(b)
    print()


# ----------------------------------------------------------------------
# Plot
fig = plt.figure(figsize=[15, 7])
grid = GridSpec(ncols=3, nrows=1, bottom=0.05, top=0.95, left=0.05, right=0.95, hspace=0.1, wspace=0.1,
                width_ratios=[1, 1, 0.05])
ax1 = plt.subplot(grid[0])
ax2 = plt.subplot(grid[1])
cax = plt.subplot(grid[2])

vmin, vmax = 0.04, 0.1
im1 = ax1.imshow(np.array(x1).reshape([xai1.size, xai2.size]), extent=[min(xai2), max(xai2), min(xai1), max(xai1)],
                 cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax, origin="lower")
im2 = ax2.imshow(np.array(x2).reshape([xai1.size, xai2.size]), extent=[min(xai2), max(xai2), min(xai1), max(xai1)],
                 cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax, origin="lower")

# Plot colorbar
plt.colorbar(im1, cax=cax, ticks=MultipleLocator(0.01), label="std")
plt.show()
