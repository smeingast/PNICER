# ----------------------------------------------------------------------
# Import stuff
import warnings
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from pnicer import Magnitudes
from matplotlib.pyplot import GridSpec
from matplotlib.ticker import MultipleLocator


# ----------------------------------------------------------------------
# Define file paths
science_path = "/Users/Antares/Dropbox/Data/Orion/VISION/Catalog/VISION_+_Spitzer_s_noYSO.fits"
control_path = "/Users/Antares/Dropbox/Data/Orion/VISION/Catalog/VISION_CF+_Spitzer_s.fits"


# ----------------------------------------------------------------------
# Helper function
def get_distance(slope, intercept, x0, y0):
    return np.abs(slope * x0 - y0 + intercept) / np.sqrt(slope**2 + 1)


# ----------------------------------------------------------------------
# Load data
skip = 1
science_dummy = fits.open(science_path)[1].data
control_dummy = fits.open(control_path)[1].data

science_glon = science_dummy["GLON"][::skip]
science_glat = science_dummy["GLAT"][::skip]

control_glon = control_dummy["GLON"]
control_glat = control_dummy["GLAT"]


features_names = ["J", "H", "Ks", "IRAC1", "IRAC2"]
errors_names = ["J_err", "H_err", "Ks_err", "IRAC1_err", "IRAC2_err"]

# Photometry
science_data = [science_dummy[n][::skip] for n in features_names]
science_error = [science_dummy[n][::skip] for n in errors_names]


control_data = [control_dummy[n] for n in features_names]
control_error = [control_dummy[n] for n in errors_names]
# features_extinction = features_extinction
features_names = features_names


# ----------------------------------------------------------------------
# Filter only those with all 5 detections
com = np.isfinite(science_data[0]) & np.isfinite(science_data[1]) & np.isfinite(science_data[2]) & \
    np.isfinite(science_data[3]) & np.isfinite(science_data[4])

science_data = [s[com] for s in science_data]
science_error = [e[com] for e in science_error]
science_glon = science_glon[com]
science_glat = science_glat[com]


# ----------------------------------------------------------------------
# Grid for extinctions
# [2.5, 1.55, 1.0, 0.636, 0.54]
j_range = np.arange(start=2.25, stop=2.35, step=0.02)
h_range = np.arange(start=1.40, stop=1.5, step=0.02)
i1_range = np.arange(start=0.65, stop=0.75, step=0.02)
i2_range = np.arange(start=0.55, stop=0.65, step=0.02)

# Create grid of parameters
jr, hr, i1r, i2r = np.meshgrid(j_range, h_range, i1_range, i2_range)
print(jr.size)


# ----------------------------------------------------------------------
# Define function to get squared sum of orthogonal distances
def get_square_sum(j, h, i1, i2):

    # Initialize data
    science = Magnitudes(mag=science_data, err=science_error, extvec=[j, h, 1, i1, i2],
                         lon=science_glon, lat=science_glat, names=features_names)
    control = Magnitudes(mag=control_data, err=control_error, extvec=[j, h, 1, i1, i2],
                         lon=control_glon, lat=control_glat, names=features_names)

    # Get slopes
    slopes = science.mag2color().extvec.extvec

    # If there is a 0 in the color excess, return NaN
    if 0 in slopes:
        return np.nan, np.array([j, h, 1, i1, i2])

    # Get extinction
    # ext = science.pnicer(control=control,add_colors=True).extinction
    ext = science.nicer(control=control).extinction

    # Get average colors in extinction bins
    step = 0.3
    jh, hks, ksi1, i1i2 = [], [], [], []
    for ak in np.arange(-1, 15.01, step=step):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fil = (ext >= ak) & (ext < ak + step)

            jh_avg = np.nanmean(science.features[0][fil] - science.features[1][fil])
            hk_avg = np.nanmean(science.features[1][fil] - science.features[2][fil])
            ksi1_avg = np.nanmean(science.features[2][fil] - science.features[3][fil])
            i1i2_avg = np.nanmean(science.features[3][fil] - science.features[4][fil])

            jh.append(jh_avg)
            hks.append(hk_avg)
            ksi1.append(ksi1_avg)
            i1i2.append(i1i2_avg)

    # Determine orthogonal distance
    dis_hks_jh = get_distance(slope=slopes[0]/slopes[1], intercept=0, x0=np.array(hks), y0=np.array(jh))
    dis_ksi1_jh = get_distance(slope=slopes[0]/slopes[2], intercept=0, x0=np.array(ksi1), y0=np.array(jh))
    dis_ksi1_hks = get_distance(slope=slopes[1]/slopes[2], intercept=0, x0=np.array(ksi1), y0=np.array(hks))
    dis_i1i2_jh = get_distance(slope=slopes[0]/slopes[3], intercept=0, x0=np.array(i1i2), y0=np.array(jh))
    dis_i1i2_hks = get_distance(slope=slopes[1]/slopes[3], intercept=0, x0=np.array(i1i2), y0=np.array(hks))
    dis_i1i2_ksi1 = get_distance(slope=slopes[2]/slopes[3], intercept=0, x0=np.array(i1i2), y0=np.array(ksi1))

    # Calculate squared sum
    square_sum = np.nansum([dis_hks_jh**2, dis_ksi1_jh**2, dis_ksi1_hks**2,
                            dis_i1i2_jh**2, dis_i1i2_hks**2, dis_i1i2_ksi1**2])

    print(square_sum, np.array([j, h, 1, i1, i2]))
    return square_sum, np.array([j, h, 1, i1, i2])


# ----------------------------------------------------------------------
# Run on grid
# from multiprocessing import Pool
# with Pool(4) as pool:
#     dis = pool.starmap(get_square_sum, zip(jr.ravel(), hr.ravel(), i1r.ravel(), i2r.ravel()))
# # Unpack results
# dis, fext = zip(*dis)
# # Get best feature extinction combination
# features_extinction = fext[np.nanargmin(dis)]

features_extinction = [2.5, 1.55, 1.0, 0.636, 0.54]
# get_square_sum(2.5, 1.55, 0.636, 0.54)
# print(features_extinction)

# ----------------------------------------------------------------------
# Initialize data with best combination
science = Magnitudes(mag=science_data, err=science_error, extvec=features_extinction,
                     lon=science_glon, lat=science_glat, names=features_names)
control = Magnitudes(mag=control_data, err=control_error, extvec=features_extinction,
                     lon=control_glon, lat=control_glat, names=features_names)


# Get extinction
# ext = science.pnicer(control=control,add_colors=True).extinction
ext = science.nicer(control=control).extinction

# Get average colors in extinction bins
step = 0.3
jh, hks, ksi1, i1i2 = [], [], [], []
for ak in np.arange(-1, 15.01, step=step):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fil = (ext >= ak) & (ext < ak + step)

        jh_avg = np.nanmean(science.features[0][fil] - science.features[1][fil])
        hk_avg = np.nanmean(science.features[1][fil] - science.features[2][fil])
        ksi1_avg = np.nanmean(science.features[2][fil] - science.features[3][fil])
        i1i2_avg = np.nanmean(science.features[3][fil] - science.features[4][fil])

        jh.append(jh_avg)
        hks.append(hk_avg)
        ksi1.append(ksi1_avg)
        i1i2.append(i1i2_avg)


# Get slopes
slopes = science.mag2color().extvec.extvec


# ----------------------------------------------------------------------
# Create figure
fig = plt.figure(figsize=[12, 12])
grid = GridSpec(ncols=3, nrows=3, bottom=0.05, top=0.95, left=0.05, right=0.95, hspace=0, wspace=0)


# Add axes
ax_hks_jh = plt.subplot(grid[8])
ax_ksi1_jh = plt.subplot(grid[7])
ax_ksi1_hks = plt.subplot(grid[4])
ax_i1i2_jh = plt.subplot(grid[6])
ax_i1i2_hks = plt.subplot(grid[3])
ax_i1i2_ksi1 = plt.subplot(grid[0])

# Plot
s, alpha = 20, 1
ax_hks_jh.scatter(hks, jh, lw=0, s=s, alpha=alpha)
ax_ksi1_jh.scatter(ksi1, jh, lw=0, s=s, alpha=alpha)
ax_ksi1_hks.scatter(ksi1, hks, lw=0, s=s, alpha=alpha)
ax_i1i2_jh.scatter(i1i2, jh, lw=0, s=s, alpha=alpha)
ax_i1i2_hks.scatter(i1i2, hks, lw=0, s=s, alpha=alpha)
ax_i1i2_ksi1.scatter(i1i2, ksi1, lw=0, s=s, alpha=alpha)

# Plot slopes
x = np.arange(-10, 10, 0.5)
ax_hks_jh.plot(x, slopes[0]/slopes[1] * x)
ax_hks_jh.set_xlabel("$H-K_S$")
ax_ksi1_jh.plot(x, slopes[0]/slopes[2] * x)
ax_ksi1_jh.set_xlabel("$K_S-[3.6]$")
ax_ksi1_hks.plot(x, slopes[1]/slopes[2] * x)
ax_i1i2_jh.plot(x, slopes[0]/slopes[3] * x)
ax_i1i2_jh.set_xlabel("$[4.5]-[3.6]$")
ax_i1i2_jh.set_ylabel("$J-H$")
ax_i1i2_hks.plot(x, slopes[1]/slopes[3] * x)
ax_i1i2_hks.set_ylabel("$H-K_S$")
ax_i1i2_ksi1.plot(x, slopes[2]/slopes[3] * x)
ax_i1i2_ksi1.set_ylabel("$K_S-[3.6]$")

# Adjust axes properties
ax_all = [ax_i1i2_ksi1, ax_i1i2_hks, ax_ksi1_hks, ax_i1i2_jh, ax_ksi1_jh, ax_hks_jh]
for idx in range(len(ax_all)):
    # Limits
    ax_all[idx].set_xlim(-0.5, 4)
    ax_all[idx].set_ylim(-0.5, 6)

    # Ticker
    ax_all[idx].xaxis.set_major_locator(MultipleLocator(1))
    ax_all[idx].xaxis.set_minor_locator(MultipleLocator(0.2))
    ax_all[idx].yaxis.set_major_locator(MultipleLocator(1))
    ax_all[idx].yaxis.set_minor_locator(MultipleLocator(0.2))

    if idx < 3:
        ax_all[idx].axes.xaxis.set_ticklabels([])

    if idx in [2, 4, 5]:
        ax_all[idx].axes.yaxis.set_ticklabels([])


# Save
plt.savefig("/Users/Antares/Dropbox/Projects/PNICER/Paper/Results/extinction_law.pdf", bbox_inches="tight")
