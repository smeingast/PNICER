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


# ----------------------------------------------------------------------
# Define file paths
science_path = "/Users/Antares/Dropbox/Data/Orion/VISION/Catalog/VISION_+_Spitzer_s.fits"
control_path = "/Users/Antares/Dropbox/Data/Orion/VISION/Catalog/VISION_CF+_Spitzer_s.fits"
results_path = "/Users/Antares/Dropbox/Projects/PNICER/Paper/Results/"
irac1_coverage_path = "/Users/Antares/Dropbox/Data/Orion/Other/Spitzer_Orion_IRAC1_coverage_s.fits"


# ----------------------------------------------------------------------
# Load colorbrewer colormap
cmap = brewer2mpl.get_map("RdBu", "Diverging", number=11, reverse=True).get_mpl_colormap(N=11, gamma=1)


# ----------------------------------------------------------------------
# Read IRAC coverage map
irac1_coverage = fits.open(irac1_coverage_path)[0].data
irac1_coverage_header = fits.open(irac1_coverage_path)[0].header


# ----------------------------------------------------------------------
# Load data
skip = 1

# Set feature parameters
features_names = ["J", "H", "Ks", "IRAC1", "IRAC2"]
errors_names = ["J_err", "H_err", "Ks_err", "IRAC1_err", "IRAC2_err"]
features_extinction = [2.5, 1.55, 1.0, 0.636, 0.54]

# Open files and load data
science_dummy = fits.open(science_path)[1].data
control_dummy = fits.open(control_path)[1].data

# Load coordinates
science_glon = science_dummy["GLON"][::skip]
science_glat = science_dummy["GLAT"][::skip]
control_glon = control_dummy["GLON"][::skip]
control_glat = control_dummy["GLAT"][::skip]


# ----------------------------------------------------------------------
def build_map(in_ext, out_name, irac1contour=False):
    """
    Function to build incremental extinction map plots
    :param in_ext: Input extinction vector
    :param out_name: Output plot figure name
    :param irac1contour: Whether the IRAC1 coverage should be drawn as contour
    """

    # Set map parameters
    bandwidth, metric, sampling, nicest, fwhm = 5/60, "epanechnikov", 2, False, False

    # Loop over number of features and get extinction map
    nicer, pnicer = [], []
    for n_features in range(3, 6):

        # Load photometry
        sdata = [science_dummy[n][::skip] for n in features_names[:n_features]]
        cdata = [control_dummy[n][::skip] for n in features_names[:n_features]]

        # Load measurement errors
        serror = [science_dummy[n][::skip] for n in errors_names[:n_features]]
        cerror = [control_dummy[n][::skip] for n in errors_names[:n_features]]

        # Feature extinction and names
        fext = in_ext[:n_features]
        fname = features_names[:n_features]

        # Initialize data
        science = Magnitudes(mag=sdata, err=serror, extvec=fext,  lon=science_glon, lat=science_glat, names=fname)
        control = Magnitudes(mag=cdata, err=cerror, extvec=fext, lon=control_glon, lat=control_glat, names=fname)
        # print(science.features_names)

        # Get NICER and PNICER extinctions
        ext_pnicer = science.mag2color().pnicer(control=control.mag2color())
        ext_nicer = science.nicer(control=control)

        # Build extinction maps
        nicer.append(ext_nicer.build_map(bandwidth=bandwidth, metric=metric,
                                         sampling=sampling, nicest=nicest, use_fwhm=fwhm))
        pnicer.append(ext_pnicer.build_map(bandwidth=bandwidth, metric=metric,
                                           sampling=sampling, nicest=nicest, use_fwhm=fwhm))

    # Create figure
    fig = plt.figure(figsize=[14, 4])
    grid = GridSpec(ncols=2, nrows=2, bottom=0.05, top=0.95, left=0.05, right=0.9, hspace=0.05, wspace=0)

    # Add colorbar axes
    cax = fig.add_axes([0.91, 0.05, 0.02, 0.9])

    # Make plots for PNICER
    ratio1 = 100 * ((pnicer[1].map / pnicer[0].map) - 1)
    ratio2 = 100 * ((pnicer[2].map / pnicer[0].map) - 1)
    vmin, vmax = -55, 55
    ax1 = plt.subplot(grid[0], projection=wcsaxes.WCS(nicer[0].fits_header))
    im1 = ax1.imshow(ratio1, origin="lower", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax2 = plt.subplot(grid[2], projection=wcsaxes.WCS(nicer[0].fits_header))
    ax2.imshow(ratio2, origin="lower", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)

    # Make plots for NICER
    ratio3 = 100 * ((nicer[1].map / nicer[0].map) - 1)
    ratio4 = 100 * ((nicer[2].map / nicer[0].map) - 1)
    ax3 = plt.subplot(grid[1], projection=wcsaxes.WCS(nicer[0].fits_header))
    ax3.imshow(ratio3, origin="lower", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax4 = plt.subplot(grid[3], projection=wcsaxes.WCS(nicer[0].fits_header))
    ax4.imshow(ratio4, origin="lower", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)

    # Plot colorbar
    plt.colorbar(im1, cax=cax, ticks=MultipleLocator(20), label="$\Delta A_K (\%)$")

    # Adjust axes
    for ax in [ax1, ax2, ax3, ax4]:

        # Set limits
        ax.set_xlim(0, nicer[0].map.shape[1])
        ax.set_ylim(0, nicer[0].map.shape[0])

        if irac1contour:
            ax.contour(irac1_coverage, levels=[0, 1], transform=ax.get_transform(wcsaxes.WCS(irac1_coverage_header)),
                       colors="black")

        # Grab axes
        lon = ax.coords[0]
        lat = ax.coords[1]

        # Set axis labels
        if ax in [ax2, ax4]:
            lon.set_axislabel("Galactic longitude (°)")
        else:
            lon.set_ticklabel_position("")

        if ax in [ax1, ax2]:
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

        # Annotate
        # ax.annotate("test", xy=(0.04, 0.9), xycoords="axes fraction", ha="left", va="top")

    # Save figure
    plt.savefig(results_path + out_name, bbox_inches="tight")


# ----------------------------------------------------------------------
# Build baseline map
# build_map(in_ext=[2.55, 1.55, 1.0, 0.56, 0.43], out_name="extinction_maps_incremental.pdf", irac1contour=True)
# exit()

# ----------------------------------------------------------------------
# Build incremental maps
i1, i2 = np.meshgrid(np.arange(0.50, 0.63, 0.06), np.arange(0.35, 0.52, 0.08))
# print(i1)
# print()
# print(i2)
# print()
# exit()
extinction = [[2.5, 1.55, 1.0, a, b] for a, b in zip(i1.ravel(), i2.ravel())]

for ext in extinction:

    outname = "extinction_maps_incremental_" + "-".join([str(x) for x in ext]) + ".pdf"
    build_map(in_ext=ext, out_name=outname, irac1contour=True)
