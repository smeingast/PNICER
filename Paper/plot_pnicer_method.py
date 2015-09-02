from __future__ import absolute_import, division, print_function
__author__ = 'Stefan Meingast'


# ----------------------------------------------------------------------
# Import stuff
import brewer2mpl
import numpy as np
import matplotlib.pyplot as plt

from pnicer import Magnitudes, Colors, mp_kde
from astropy.io import fits
from matplotlib.pyplot import GridSpec


# ----------------------------------------------------------------------
# Define file paths
science_path = "/Users/Antares/Dropbox/Data/Orion/VISION/Catalog/VISION_+_Spitzer_s.fits"
control_path = "/Users/Antares/Dropbox/Data/Orion/VISION/Catalog/VISION_CF+_Spitzer_s.fits"
results_path = "/Users/Antares/Dropbox/Projects/PNICER/Paper/Results/"

# Load colormap
cmap = brewer2mpl.get_map('Blues', 'Sequential', number=9, reverse=False).get_mpl_colormap(N=100, gamma=0.5)


# ----------------------------------------------------------------------
# Load data
control_dummy = fits.open(control_path)[1].data

control_glon = control_dummy["GLON"]
control_glat = control_dummy["GLAT"]


# ----------------------------------------------------------------------
# Define features to be used
features_names = ["J", "H", "Ks", "IRAC1"]
errors_names = ["J_err", "H_err", "Ks_err", "IRAC1_err"]
features_extinction = [2.5, 1.55, 1.0, 0.636]

# ----------------------------------------------------------------------
# Load data into lists for PNICER
control_data = [control_dummy[n] for n in features_names]
control_error = [control_dummy[n] for n in errors_names]


# ----------------------------------------------------------------------
# Initialize data
control = Magnitudes(mag=control_data, err=control_error, extvec=features_extinction,
                     lon=control_glon, lat=control_glat, names=features_names)  # .mag2color()
control_rot = control.rotate()

# Define single source to be de-reddened
fake = [np.array([x, y]) for x, y in zip([19.04, 17.16, 16.22, 15.16], [21, 19.57, 18.15, 17.02, 16.51])]
fake_err = [np.array([x, y]) for x, y in zip([0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1])]

fake = Magnitudes(mag=fake, err=fake_err, extvec=features_extinction, names=features_names)  # .mag2color()
fake_rot = fake.rotate()

print(fake.features)

# ----------------------------------------------------------------------
# Run PNICER
ext_fake = fake.pnicer(control=control, sampling=2)
print(ext_fake.extinction)
ext_fake = fake.nicer(control=control)
print(ext_fake.extinction)
exit()

# ----------------------------------------------------------------------
# Get densities for normal data
idx_combinations = [(1, 2), (1, 0), (2, 0)]

# Create grid
grid_bw = 0.05
l, h = -0.5, 2.5 + grid_bw / 2
x, y = np.meshgrid(np.arange(start=l, stop=h, step=grid_bw), np.arange(start=l, stop=h, step=grid_bw))
xgrid = np.vstack([x.ravel(), y.ravel()]).T
edges = (np.min(x), np.max(x), np.min(y), np.max(y))


dens_control_norm, dens_control_rot = [], []
for i2, i1 in idx_combinations:

    data = np.vstack([control.features[i1][control.combined_mask], control.features[i2][control.combined_mask]]).T
    dens_control_norm.append(mp_kde(grid=xgrid, data=data, bandwidth=grid_bw*2, shape=x.shape, kernel="epanechnikov"))

    data = np.vstack([control_rot.features[i1][control_rot.combined_mask], control_rot.features[i2][control_rot.combined_mask]]).T
    dens_control_rot.append(mp_kde(grid=xgrid, data=data, bandwidth=grid_bw*2, shape=x.shape, kernel="epanechnikov"))


# ----------------------------------------------------------------------
# Create figure
fig = plt.figure(figsize=[20, 10])
gs_cn = GridSpec(ncols=2, nrows=2, bottom=0.05, top=0.95, left=0.05, right=0.35, hspace=0, wspace=0)
gs_cr = GridSpec(ncols=2, nrows=2, bottom=0.05, top=0.95, left=0.40, right=0.70, hspace=0, wspace=0)
gs_d = GridSpec(ncols=1, nrows=2, bottom=0.05, top=0.95, left=0.75, right=0.95, hspace=0, wspace=0)

ax_cn_1 = plt.subplot(gs_cn[0])
ax_cn_2 = plt.subplot(gs_cn[2])
ax_cn_3 = plt.subplot(gs_cn[3])
ax_cn_all = [ax_cn_1, ax_cn_2, ax_cn_3]

ax_cr_1 = plt.subplot(gs_cr[0])
ax_cr_2 = plt.subplot(gs_cr[2])
ax_cr_3 = plt.subplot(gs_cr[3])
ax_cr_all = [ax_cr_1, ax_cr_2, ax_cr_3]

ax_d1 = plt.subplot(gs_d[0])
ax_d2 = plt.subplot(gs_d[1])

# Plot standard data
for ax_cn, ax_cr, idx, cidx in zip(ax_cn_all, ax_cr_all, range(3), idx_combinations):

    # Show densities
    ax_cn.imshow(dens_control_norm[idx].T, interpolation="nearest", origin="lower", extent=edges, cmap=cmap)
    ax_cr.imshow(dens_control_rot[idx].T, interpolation="nearest", origin="lower", extent=edges, cmap=cmap)

    # Show extinction arrows
    ax_cn.arrow(1.5, 0, control.extvec.extvec[cidx[0]], control.extvec.extvec[cidx[1]], length_includes_head=True)
    if idx >= 1:
        ax_cr.arrow(2.0, 0, control_rot.extvec.extvec[cidx[0]], control_rot.extvec.extvec[cidx[1]], length_includes_head=True)

    # Show fake source
    ax_cn.scatter(fake.features[cidx[0]], fake.features[cidx[1]])
    ax_cr.scatter(fake_rot.features[cidx[0]], fake_rot.features[cidx[1]])

    ax_cn.scatter(ext_fake.features_dered[cidx[0]], ext_fake.features_dered[cidx[1]])

    for x, y in zip(fake.features[cidx[0]], fake.features[cidx[1]]):
        k = control.extvec.extvec[cidx[1]] / control.extvec.extvec[cidx[0]]
        d = y - k * x
        xn = (-100, 100)
        yn = (k * xn[0] + d, k * xn[1] + d)
        ax_cn.plot(xn, yn)

    # Plot probability density line through fake sources
    if idx >= 1:
        ax_cr.vlines([fake_rot.features[cidx[0]][0], fake_rot.features[cidx[0]][1]], -10, 10)

    for ax in [ax_cn, ax_cr]:
        ax.set_xlim(edges[:2])
        ax.set_ylim(edges[2:])

    # for ax in [ax_cn, ax_sn]:
    #     # Show fake source
    #     ax.scatter(fake.features[cidx[0]], fake.features[cidx[1]])
    #
    # for ax in [ax_cr, ax_sr]:
    #     # Show fake source
    #     ax.scatter(fake_rot.features[cidx[0]], fake_rot.features[cidx[1]])

    for ax in [ax_cn, ax_cr]:

        # Adjust axes
        if idx == 0:
            ax.axes.xaxis.set_ticklabels([])
            ax.set_ylabel("$K_S - IRAC1$")

        if idx == 1:
            ax.set_ylabel("$J - H$")
            ax.set_xlabel("$H - K_S$")

        if idx == 2:
            ax.axes.yaxis.set_ticklabels([])
            ax.set_xlabel("$K_S - IRAC1$")

# Plot PDF



# ax_d1.scatter()

plt.savefig(results_path + "method.pdf", bbox_inches="tight")


