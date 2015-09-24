# ----------------------------------------------------------------------
# Import stuff
import wcsaxes
import warnings
import brewer2mpl
import matplotlib.pyplot as plt

from pnicer import Magnitudes, Extinction, get_covar
from scipy.odr import Model, RealData, ODR
from matplotlib.pyplot import GridSpec
from matplotlib.ticker import MultipleLocator
from MyFunctions import distance_on_unit_sphere
from helper import *


# ----------------------------------------------------------------------
# Define file paths
science_path = "/Users/Antares/Dropbox/Data/Orion/VISION/Catalog/VISION_+_Spitzer_s_noYSO.fits"
control_path = "/Users/Antares/Dropbox/Data/Orion/VISION/Catalog/VISION_CF+_Spitzer_s.fits"
results_path = "/Users/Antares/Dropbox/Projects/PNICER/Paper/Results/"


# ----------------------------------------------------------------------
# Load colorbrewer colormap
cmap0 = brewer2mpl.get_map("Greys", "Sequential", number=9, reverse=False).get_mpl_colormap(N=500, gamma=1)
cmap1 = brewer2mpl.get_map("Spectral", "Diverging", number=11, reverse=True).get_mpl_colormap(N=100, gamma=1)
cmap2 = brewer2mpl.get_map("YlGnBu", "Sequential", number=9, reverse=True).get_mpl_colormap(N=10, gamma=1)
cmap3 = brewer2mpl.get_map("YlOrRd", "Sequential", number=9, reverse=False).get_mpl_colormap(N=10, gamma=1)
viridis = get_viridis()

# ----------------------------------------------------------------------
# Intialize all data
science_all, control_all = pnicer_ini(n_features=5)
science_color_all, control_color_all = science_all.mag2color(), control_all.mag2color()

# Additionally load galaxy classifier
class_sex_science = fits.open(science_path)[1].data["class_sex"]
class_sex_control = fits.open(control_path)[1].data["class_sex"]
class_cog_science = fits.open(science_path)[1].data["class_cog"]
class_cog_control = fits.open(control_path)[1].data["class_cog"]


# ----------------------------------------------------------------------
# Define base bands
base_keys = ("H", "Ks")
# fit_keys = ("J", "IRAC1", "IRAC2")
fit_keys = ("J", "IRAC1")


# ----------------------------------------------------------------------
# Get preliminary extinction
# dummy = science_color.pnicer(control=control_color)
dummy = science_all.nicer(control=control_all)
# dummy.save_fits(path="/Users/Antares/Desktop/test.fits")
ak_all = dummy.extinction


# ----------------------------------------------------------------------
# Make pre-selection of data for all combinations and re-initialize
sd, cd = science_all.dict, control_all.dict
science_fil, control_fil = {}, {}
fig1 = plt.figure(figsize=[12, 5])
grid1 = GridSpec(ncols=3, nrows=1, bottom=0.05, top=0.95, left=0.05, right=0.95, hspace=0.1, wspace=0.2)
axes_sel = [plt.subplot(grid1[i]) for i in range(len(fit_keys))]
for key, sax in zip(fit_keys, axes_sel):

    if key in ["J", "H", "Ks"]:
        lim = 0.1
    else:
        lim = 0.03

    # Construct filter
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sfil = (sd[key] > 0) & (sd[base_keys[0]] > 0) & (sd[base_keys[1]] < 17) & (sd[key + "_err"] < lim) & \
               (class_cog_science == 1) & (class_sex_science > 0.8) & (ak_all > 0.5)
        cfil = (cd[key] > 0) & (cd[base_keys[0]] > 0) & (cd[base_keys[1]] < 17) & (cd[key + "_err"] < 0.1) & \
               (class_cog_control == 1) & (class_sex_control > 0.8)

    # Re-initialize
    science_fil[key], control_fil[key] = pnicer_ini(n_features=5, sfil=sfil, cfil=cfil)

    # Plot selection
    sax.scatter(science_all.dict[base_keys[0]] - science_all.dict[base_keys[1]],
                science_all.dict[base_keys[1]] - science_all.dict[key], lw=0, color="grey", alpha=0.1, s=2)
    sax.scatter(science_fil[key].dict[base_keys[0]] - science_fil[key].dict[base_keys[1]],
                science_fil[key].dict[base_keys[1]] - science_fil[key].dict[key], lw=0, color="red", alpha=0.1, s=2)
    sax.set_xlim(-0.5, 3)
    sax.set_xlabel("$" + base_keys[0] + "-" + base_keys[1] + "$")
    sax.set_ylabel("$" + base_keys[1] + "-" + key + "$")

# Save plot of pre-selection
plt.savefig(results_path + "extinction_law_spatial_selection.png", bbox_inches="tight", dpi=300)
plt.close()


# ----------------------------------------------------------------------
# Define helper functions

def f(vec, val):
    """Linear function y = m*x + b
    :param vec: vector of the parameters
    :param val: array of the current x values
    """
    return vec[0] * val + vec[1]

# Define model
fit_model = Model(f)


def color_average(xcolor, ycolor, xy_ak, ak_step=0.1):

    # Must have equal number of measurements
    assert len(xcolor) == len(ycolor) == len(xy_ak)

    # Get average colors in extinction bins
    avg_x, avg_y, std_x, std_y, avg_ak, avg_n = [], [], [], [], [], []
    for a in np.arange(np.floor(np.nanmin(xy_ak)), np.ceil(np.nanmax(xy_ak)), step=ak_step):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            avg_fil = (xy_ak >= a) & (xy_ak < a + ak_step)

            # Get color for current filter extinction
            xdummy, ydummy = xcolor[avg_fil], ycolor[avg_fil]

            # Do 3-sigma clipping around median
            avg_clip = (np.abs(xdummy - np.median(xdummy)) < 3 * np.std(xdummy)) & \
                       (np.abs(ydummy - np.median(ydummy)) < 3 * np.std(ydummy))

            # We require at least 5 sources
            if np.sum(avg_clip) < 5:
                continue

            # Append averages
            avg_x.append(np.median(xdummy[avg_clip]))
            avg_y.append(np.median(ydummy[avg_clip]))
            std_x.append(np.std(xdummy[avg_clip]))
            std_y.append(np.std(ydummy[avg_clip]))
            avg_ak.append(a + ak_step / 2)
            avg_n.append(np.sum(avg_clip))

    # Convert to arrays
    avg_x, avg_y, avg_ak, avg_n = np.array(avg_x), np.array(avg_y), np.array(avg_ak), np.array(avg_n)

    # Return
    return avg_x, avg_y, std_x, std_y, avg_ak, avg_n


def bin_extinction(xcolor, ycolor, xy_ak, ak_step):

    # Get average colors in extinciton bins
    acol1, acol2, stdc1, stdc2, aak, an = color_average(xcolor=xcolor, ycolor=ycolor, xy_ak=xy_ak, ak_step=ak_step)

    # Fit a line with ODR
    fit_data = RealData(acol1, acol2, sx=stdc1, sy=stdc2)
    beta_odr, ic_odr = ODR(fit_data, fit_model, beta0=[1., 0.]).run().beta

    return beta_odr, ic_odr, acol1, acol2, stdc1, stdc2, aak, an


def lines(pnicer_science, pnicer_control, base_key, fit_key):

    # Type assertion
    assert (isinstance(pnicer_science, Magnitudes)) & (isinstance(pnicer_control, Magnitudes))

    # Get index of features
    base_x = (pnicer_science.features_names.index(base_keys[0]), pnicer_science.features_names.index(base_keys[1]))
    fit_x = pnicer_science.features_names.index(fit_key)

    # Create combined mask
    smask = np.prod(np.vstack([pnicer_science.features_masks[i] for i in base_x + (fit_x,)]), axis=0, dtype=bool)
    cmask = np.prod(np.vstack([pnicer_control.features_masks[i] for i in base_x + (fit_x,)]), axis=0, dtype=bool)

    # Shortcut for data
    x_s_lines = pnicer_science.dict[base_key[0]][smask] - pnicer_science.dict[base_key[1]][smask]
    y_s_lines = pnicer_science.dict[base_key[1]][smask] - pnicer_science.dict[fit_key][smask]
    x_c_lines = pnicer_control.dict[base_key[0]][cmask] - pnicer_control.dict[base_key[1]][cmask]
    y_c_lines = pnicer_control.dict[base_key[1]][cmask] - pnicer_control.dict[fit_key][cmask]

    # Determine (Co-)variance terms of errors
    var_err_science = np.mean(pnicer_science.features_err[base_x[0]][smask]) ** 2 + \
                      np.mean(pnicer_science.features_err[base_x[1]][smask]) ** 2
    cov_err_science = -np.mean(pnicer_science.features_err[base_x[1]][smask]) ** 2
    var_err_control = np.mean(pnicer_control.features_err[base_x[0]][cmask]) ** 2 + \
                      np.mean(pnicer_control.features_err[base_x[1]][cmask]) ** 2
    cov_err_control = -np.mean(pnicer_control.features_err[base_x[1]][cmask]) ** 2

    upper = get_covar(xi=x_s_lines, yi=y_s_lines) - get_covar(xi=x_c_lines, yi=y_c_lines) - \
            cov_err_science + cov_err_control
    lower = np.var(x_s_lines) - var_err_science - np.var(x_c_lines) + var_err_control

    beta_ini = upper / lower

    # Do iterative clipping
    clip_mask = np.arange(pnicer_science.n_data)
    clip_idx = np.arange(pnicer_science.n_data)

    # Get intercept
    fil_dummy = []
    for _ in range(1):

        # Shortcut for data
        xdata_dummy = pnicer_science.dict[base_key[0]][clip_mask] - pnicer_science.dict[base_key[1]][clip_mask]
        ydata_dummy = pnicer_science.dict[base_key[1]][clip_mask] - pnicer_science.dict[fit_key][clip_mask]

        # Get intercept of linear fit throguh median
        ic_dummy = np.median(ydata_dummy) - beta_ini * np.median(xdata_dummy)

        # Get orthogonal distance
        dis_dummy = np.abs(beta_ini * xdata_dummy - ydata_dummy + ic_dummy) / np.sqrt(beta_ini ** 2 + 1)

        # 3 sig filter
        clip_mask = dis_dummy < 3 * np.std(dis_dummy)
        clip_idx = clip_idx[clip_mask]

    # Now append the bad ones
    fil_dummy.extend(np.setdiff1d(np.arange(pnicer_science.n_data), clip_idx))

    # Determine which sources to filter
    uidx_dummy = np.unique(np.array(fil_dummy))
    # And finally get the good index
    g_idx_dummy = np.setdiff1d(np.arange(pnicer_science.n_data), uidx_dummy)

    # Final initialization with cleaned data
    science_data_good = [pnicer_science.features[i][g_idx_dummy] for i in range(pnicer_science.n_features)]
    science_err_good = [pnicer_science.features_err[i][g_idx_dummy] for i in range(pnicer_science.n_features)]

    # Overwrite input
    pnicer_science = Magnitudes(mag=science_data_good, err=science_err_good, extvec=pnicer_science.extvec.extvec,
                                names=pnicer_science.features_names)

    # Create combined mask
    smask = np.prod(np.vstack([pnicer_science.features_masks[i] for i in base_x + (fit_x,)]), axis=0, dtype=bool)
    cmask = np.prod(np.vstack([pnicer_control.features_masks[i] for i in base_x + (fit_x,)]), axis=0, dtype=bool)

    # Shortcut for data
    x_s_lines = pnicer_science.dict[base_key[0]][smask] - pnicer_science.dict[base_key[1]][smask]
    y_s_lines = pnicer_science.dict[base_key[1]][smask] - pnicer_science.dict[fit_key][smask]
    x_c_lines = pnicer_control.dict[base_key[0]][cmask] - pnicer_control.dict[base_key[1]][cmask]
    y_c_lines = pnicer_control.dict[base_key[1]][cmask] - pnicer_control.dict[fit_key][cmask]

    # Determine (Co-)variance terms of errors
    var_err_science = np.mean(pnicer_science.features_err[base_x[0]][smask]) ** 2 + \
                      np.mean(pnicer_science.features_err[base_x[1]][smask]) ** 2
    cov_err_science = -np.mean(pnicer_science.features_err[base_x[1]][smask]) ** 2
    var_err_control = np.mean(pnicer_control.features_err[base_x[0]][cmask]) ** 2 + \
                      np.mean(pnicer_control.features_err[base_x[1]][cmask]) ** 2
    cov_err_control = -np.mean(pnicer_control.features_err[base_x[1]][cmask]) ** 2

    upper = get_covar(xi=x_s_lines, yi=y_s_lines) - get_covar(xi=x_c_lines, yi=y_c_lines) - \
            cov_err_science + cov_err_control
    lower = np.var(x_s_lines) - var_err_science - np.var(x_c_lines) + var_err_control

    # Finally return beta
    return upper / lower, x_s_lines, y_s_lines


# ----------------------------------------------------------------------
# Construct an on-sky grid upon which to evaluate the slopes
header, glon_grid, glat_grid = science_all.build_wcs_grid(frame="galactic", pixsize=15 / 60)
# glon_range, glat_range = all_glon.ravel(), all_glat.ravel()


# ----------------------------------------------------------------------
# Now we loop over each selected feature and do the fitting
# fig2 = plt.figure(figsize=[12, 5])
# grid2 = GridSpec(ncols=3, nrows=1, bottom=0.05, top=0.95, left=0.05, right=0.95, hspace=0.1, wspace=0.2)
# axes_fit = [plt.subplot(grid2[i]) for i in range(len(fit_keys))]

# Now loop over keys
pglon, pglat, beta_binning, beta_lines = {}, {}, {}, {}
for key in fit_keys:

    pglon[key], pglat[key], beta_binning[key], beta_lines[key] = [], [], [], []

    sc, cc = science_fil[key], control_fil[key]
    assert (isinstance(sc, Magnitudes)) & (isinstance(cc, Magnitudes))

    for glon, glat in zip(glon_grid.ravel(), glat_grid.ravel()):

        # Append data
        pglon[key].append(glon)
        pglat[key].append(glat)

        # get sources within given radius
        dis = np.degrees(distance_on_unit_sphere(ra1=np.radians(sc.lon), dec1=np.radians(sc.lat),
                                                 ra2=np.radians(glon), dec2=np.radians(glat)))

        # Additional distance filtering for science field
        dfil = dis < 30 / 60

        # Save number of sources within filter limits
        n = np.sum(dfil)

        # Skip if there are too few
        if np.sum(dfil) < 500:
            beta_binning[key].append(np.nan)
            beta_lines[key].append(np.nan)
            continue

        # Initialize with current distance filter
        fil_data = [sc.features[i][dfil] for i in range(sc.n_features)]
        fil_err = [sc.features_err[i][dfil] for i in range(sc.n_features)]
        fil_sc = Magnitudes(mag=fil_data, err=fil_err, extvec=sc.extvec.extvec, names=sc.features_names)

        # First we do binning in Extinction
        fil_ak = fil_sc.mag2color().pnicer(control=cc.mag2color()).extinction
        color1 = fil_sc.dict[base_keys[0]] - fil_sc.dict[base_keys[1]]
        color2 = fil_sc.dict[base_keys[1]] - fil_sc.dict[key]

        # Skip if base color range is less than 1
        if np.nanmax(color1) - np.nanmin(color1) < 1:
            beta_binning[key].append(np.nan)
            beta_lines[key].append(np.nan)
            continue

        # Do ODR fit to binned data
        bbeta, ic_binning, mcolor1, mcolor2, mstd1, mstd2, mavg, mn = \
            bin_extinction(xcolor=color1, ycolor=color2, xy_ak=fil_ak, ak_step=0.2)

        # No we also perform the LINES fit
        lbeta, lcolor1, lcolor2 = lines(pnicer_science=fil_sc, pnicer_control=cc, base_key=base_keys, fit_key=key)

        # Plot parameters
        x_dummy = np.arange(-1, 5, 1)
        ic_lines = np.median(lcolor2) - lbeta * np.median(lcolor1)

        # Generate plot name
        name = str(np.around(glon, 3)) + "_" + str(np.around(glat, 3)) + "_" + key + ".png"

        # Create figure
        fig_dummy, [ax0, ax1] = plt.subplots(nrows=1, ncols=2, figsize=[10, 5])

        # Plot LINES
        ax0.plot(x_dummy, lbeta * x_dummy + ic_lines, color="black", lw=1, linestyle="dashed", alpha=0.5)
        ax0.scatter(lcolor1, lcolor2, lw=0, s=10, color="blue")

        # Plot binning
        ax1.scatter(mcolor1, mcolor2, lw=0, s=20, c=mavg, zorder=2)
        ax1.errorbar(mcolor1, mcolor2, xerr=mstd1, yerr=mstd2, zorder=1, ecolor="black", elinewidth=0.5, fmt="none")
        ax1.plot(x_dummy, bbeta * x_dummy + ic_binning, color="black", lw=1, linestyle="dashed", alpha=0.5, zorder=3)

        for ax in [ax0, ax1]:
            ax.set_xlabel("$" + base_keys[0] + "-" + base_keys[1] + "$")
            ax.set_ylabel("$" + base_keys[1] + "-" + key + "$")
            ax.set_xlim(-0.5, 3)
            if key == "J":
                ax.set_ylim(-6, 0)
            if key == "IRAC1":
                ax.set_ylim(-0.5, 2)
            if key == "IRAC2":
                ax.set_ylim(-0.5, 2)

        # Save for current position
        plt.savefig("/Users/Antares/Desktop/test/" + name, bbox_inches="tight", dpi=300)
        plt.close()

        # Append data only if fits match reasonably
        if bbeta - lbeta > 0.1:
            beta_binning[key].append(np.nan)
            beta_lines[key].append(np.nan)
        else:
            beta_binning[key].append(bbeta)
            beta_lines[key].append(lbeta)

    # Convert to arrays
    pglon[key], pglat[key] = np.array(pglon[key]), np.array(pglat[key])
    beta_binning[key], beta_lines[key] = np.array(beta_binning[key]), np.array(beta_lines[key])


# Create figure for spatially variable slope plot
fig = plt.figure(figsize=[11, 5])
grid = GridSpec(ncols=3, nrows=3, bottom=0.05, top=0.95, left=0.05, right=0.95, hspace=0.1, wspace=0.1,
                width_ratios=[1, 1, 0.02])

# Add axes
ax0_l = plt.subplot(grid[0], projection=wcsaxes.WCS(header=header))
ax0_b = plt.subplot(grid[1], projection=wcsaxes.WCS(header=header))
cax0 = plt.subplot(grid[2])

ax1_l = plt.subplot(grid[3], projection=wcsaxes.WCS(header=header))
ax1_b = plt.subplot(grid[4], projection=wcsaxes.WCS(header=header))
cax1 = plt.subplot(grid[5])

ax2_l = plt.subplot(grid[6], projection=wcsaxes.WCS(header=header))
ax2_b = plt.subplot(grid[7], projection=wcsaxes.WCS(header=header))
cax2 = plt.subplot(grid[8])

# Plot spatial dependency for everything
for key, axl, axb, cax, crange in zip(fit_keys, [ax0_l, ax1_l, ax2_l],
                              [ax0_b, ax1_b, ax2_b], [cax0, cax1, cax2], [(-2.8, -2.5), (0.55, 0.7), (0.6, 0.9)]):

    # LINES slopes
    im_l = axl.imshow(beta_lines[key].reshape(glat_grid.shape), cmap=cmap1,
                      interpolation="nearest", origin="lower", vmin=crange[0], vmax=crange[1])
    # Binning slopes
    im_b = axb.imshow(beta_binning[key].reshape(glat_grid.shape), cmap=cmap1,
                      interpolation="nearest", origin="lower", vmin=crange[0], vmax=crange[1])
    # Colorbar
    plt.colorbar(im_l, cax=cax, ticks=MultipleLocator(0.05), label=r"$\beta$")

# Save plot
plt.savefig(results_path + "extinction_law_spatial.pdf", bbox_inches="tight", dpi=300)
plt.close()

exit()

# # ----------------------------------------------------------------------
# # Make pre-selection of data
# # pnicer = science_color.pnicer(control=control_color).save_fits(path="/Users/Antares/test.fits")
# # read table
# """ For some reason when I open a pool with PNICER, i can't do plotting down below during the parallelized
# spatial mapping"""
# ext = fits.open("/Users/Antares/test.fits")[1].data["Extinction"]
# # ext = np.full_like(science.features[0], fill_value=1.0)
# for d in [science.dict, control.dict]:
#     # Define filter
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         fil = (d["J"] > 0) & (d["H"] > 0) & (d["Ks"] < 17) & (d["Ks_err"] < 0.1)
#
#         # Test no filtering
#         # fil = np.full_like(d["J"], fill_value=True, dtype=bool)
#
#         if d == science.dict:
#             sfil = fil & (ext > 0.5) & (class_sex_science > 0.8) & (class_cog_science == 1)
#             # sfil = fil.copy()
#         else:
#             cfil = fil & (class_sex_control > 0.8) & (class_cog_control == 1)
#
#
# # ----------------------------------------------------------------------
# # Re-initialize with filtered data
# # noinspection PyUnboundLocalVariable
# science, control = pnicer_ini(skip_science=1, skip_control=1, n_features=3, color=False, sfil=sfil, cfil=cfil)
# science_color, control_color = science.mag2color(), control.mag2color()
#
# # Additionally load galaxy classifier
# class_sex_science = fits.open(science_path)[1].data["class_sex"][sfil]
# class_sex_control = fits.open(control_path)[1].data["class_sex"][cfil]
# class_cog_science = fits.open(science_path)[1].data["class_cog"][sfil]
# class_cog_control = fits.open(control_path)[1].data["class_cog"][cfil]
#
#
# # ----------------------------------------------------------------------
# # Get slopes of CCDs
# base_idx = (1, 2)
# fit_idx, betas, err = science.get_extinction_law(base_index=base_idx, method="OLS", control=control)
#
#
# # ----------------------------------------------------------------------
# # Do iterative clipping
# fil_idx = []
# for fidx, beta in zip(fit_idx, betas):
#
#     mask = np.arange(science.n_data)
#     idx = np.arange(science.n_data)
#
#     # Get intercept
#     for _ in range(1):
#
#         # Shortcut for data
#         xdata_science = science.features[base_idx[0]][mask] - science.features[base_idx[1]][mask]
#         ydata_science = science.features[base_idx[1]][mask] - science.features[fidx][mask]
#
#
#         # Get intercept of linear fir throguh median
#         ic = np.median(ydata_science) - beta * np.median(xdata_science)
#
#         # Get orthogonal distance
#         odis = np.abs(beta * xdata_science - ydata_science + ic) / np.sqrt(beta ** 2 + 1)
#
#         # 3 sig filter
#         mask = odis < 3 * np.std(odis)
#         idx = idx[mask]
#
#     # Now append the bad ones
#     fil_idx.extend(np.setdiff1d(np.arange(science.n_data), idx))
#
# # Determine which sources to filter
# uidx = np.unique(np.array(fil_idx))
# # And finally get the good index
# good_index = np.setdiff1d(np.arange(science.n_data), uidx)
#
#
# # ----------------------------------------------------------------------
# # Final initialization with cleaned data
# science_data_good = [science.features[i][good_index] for i in range(science.n_features)]
# science_err_good = [science.features_err[i][good_index] for i in range(science.n_features)]
#
# science = Magnitudes(mag=science_data_good, err=science_err_good, extvec=science.extvec.extvec,
#                      lon=science.lon[good_index], lat=science.lat[good_index])
# sciencecolor = science.mag2color()
#
#
# # ----------------------------------------------------------------------
# # Get grid
# pixsize = 10 / 60
# header, all_glon, all_glat = science.build_wcs_grid(frame="galactic", pixsize=pixsize)
# grid_shape = all_glat.shape
# glon_range, glat_range = all_glon.ravel(), all_glat.ravel()
#
#
# # ----------------------------------------------------------------------
# # Build preliminary extinction map
# # emap = pnicer.build_map(bandwidth=pixsize * 2, metric="epanechnikov")
#
#
# # ----------------------------------------------------------------------
# # Define function tog et extinction law
# def get_slope(glon_pix, glat_pix, glon_all, glat_all, maxdis):
#
#     # Calculate distance to all other sources from current grid point
#     dis = np.degrees(distance_on_unit_sphere(ra1=np.radians(glon_all), dec1=np.radians(glat_all),
#                                              ra2=np.radians(glon_pix), dec2=np.radians(glat_pix)))
#
#     # Additional distance filtering for science field
#     dfil = dis < maxdis
#
#     # Save number of sources within filter limits
#     n = np.sum(dfil)
#
#     # Skip if there are too few
#     if np.sum(dfil) < 500:
#         return np.nan, np.nan, np.nan, np.nan
#
#     # Initialize with current distance filter
#     csdata = [science.features[i][dfil] for i in range(science.n_features)]
#     cserr = [science.features_err[i][dfil] for i in range(science.n_features)]
#     sc = Magnitudes(mag=csdata, err=cserr, extvec=science.extvec.extvec)
#     xd, yd = sc.features[1] - sc.features[2], sc.features[2] - sc.features[0]
#
#     # Return NaN if color range is too small
#     if np.max(xd) - np.min(xd) < 1:
#         return np.nan, np.nan, np.nan, np.nan
#
#     # Filter Ks-H > 4
#     good = yd > -4
#     csdata = [d[good] for d in csdata]
#     cserr = [e[good] for e in cserr]
#     sc = Magnitudes(mag=csdata, err=cserr, extvec=science.extvec.extvec)
#     xd, yd = sc.features[1] - sc.features[2], sc.features[2] - sc.features[0]
#
#     f, b, e = sc.get_extinction_law(base_index=(1, 2), method="LINES", control=control)
#
#     # Return if error is too large
#     if e[0] > 0.05:
#         return np.nan, np.nan, np.nan, np.nan
#
#     name = str(np.around(glon_pix, 3)) + "_" + str(np.around(glat_pix, 3)) + ".png"
#     myplot(xd, yd, be=b[0], berr=e[0], name=name)
#     time.sleep(0.3)
#
#     # Just return for J band
#     return f[0], b[0], e[0], n
#
#
# def myplot(a, b, be, berr, name):
#
#     # Get intercept
#     interc = np.median(b) - be * np.median(a)
#     _, ax = plt.subplots(nrows=1, ncols=1, figsize=[10, 10])
#     ax.scatter(a, b, lw=0)
#     # ax.plot(np.arange(-1, 5, 1), be * np.arange(-1, 5, 1) + interc, color="black", lw=2, linestyle="dashed")
#     ax.annotate(str(np.around(be, 3)) + " $\pm $ " + str(np.around(berr, 3)),
#                 xy=[0.95, 0.95], xycoords="axes fraction", ha="right", va="top")
#     ax.set_xlim(0, 2.5)
#     ax.set_ylim(-6, 0)
#     ax.set_aspect(1)
#     plt.savefig("/Users/Antares/Desktop/test/" + name, bbox_inches="tight", dpi=300)
#     plt.close()
#
# # ----------------------------------------------------------------------
# # Run parallel
# # with warnings.catch_warnings():
# #     warnings.simplefilter("ignore")
# with Pool(6) as pool:
#     mp = pool.starmap(get_slope, zip(glon_range, glat_range, repeat(science.lon),
#                                      repeat(science.lat), repeat(30 / 60)))
#
# # Unpack results
# idx, slope, slope_err, nsources = list(zip(*mp))
#
# # Convert results to arrays
# slope, slope_err, nsources = np.array(slope), np.array(slope_err), np.array(nsources, dtype=float)
#
#
# # ----------------------------------------------------------------------
# # Plot results
# # plt.imshow(np.array(slope).reshape(grid_shape), cmap="brg", vmin=2.45, vmax=2.55, interpolation="nearest")
# fig = plt.figure(figsize=[11, 13])
# grid = GridSpec(ncols=2, nrows=4, bottom=0.05, top=0.95, left=0.05, right=0.95, hspace=0.1, wspace=0.1,
#                 width_ratios=[1, 0.02])
# # Add axes
# # ax0 = plt.subplot(grid[0], projection=wcsaxes.WCS(header=emap.fits_header))
# # cax0 = plt.subplot(grid[1])
# ax1 = plt.subplot(grid[2], projection=wcsaxes.WCS(header=header))
# cax1 = plt.subplot(grid[3])
# ax2 = plt.subplot(grid[4], projection=wcsaxes.WCS(header=header))
# cax2 = plt.subplot(grid[5])
# ax3 = plt.subplot(grid[6], projection=wcsaxes.WCS(header=header))
# cax3 = plt.subplot(grid[7])
#
# # Plot extinction map
# # im0 = ax0.imshow(emap.map, interpolation="nearest", origin="lower", vmin=0, vmax=2, cmap=cmap0)
# # plt.colorbar(im0, cax=cax0, ticks=MultipleLocator(0.25), label="$A_K$")
#
# # Plot slope
# im1 = ax1.imshow(slope.reshape(grid_shape), cmap=cmap1, interpolation="nearest",
#                  origin="lower", vmin=-2.85, vmax=-2.5)
# plt.colorbar(im1, cax=cax1, ticks=MultipleLocator(0.05), label=r"$\beta$")
#
# # Save slope map
# hdulist = fits.HDUList([fits.PrimaryHDU(),
#                         fits.ImageHDU(data=slope.reshape(grid_shape), header=header),
#                         fits.ImageHDU(data=slope_err.reshape(grid_shape), header=header),
#                         fits.ImageHDU(data=nsources.reshape(grid_shape), header=header)])
# hdulist.writeto("/Users/Antares/Desktop/spatial.fits", clobber=True)
#
# # Plot slope error
# im2 = ax2.imshow(slope_err.reshape(grid_shape), cmap=viridis,
#                  interpolation="nearest", origin="lower", vmin=0, vmax=0.05)
# plt.colorbar(im2, cax=cax2, label=r"$\sigma_{\beta}$")
#
# # Plot number of sources
# # ax2.scatter(glon_range, glat_range, c=slope, lw=0, marker="s", s=300, cmap=cmap, vmin=2.45, vmax=2.55)
# im3 = ax3.imshow(nsources.reshape(grid_shape), cmap=viridis, interpolation="nearest", origin="lower")
# plt.colorbar(im3, cax=cax3, label="#")
#
# # Draw IRAC1 coverage
# irac1_coverage_path = "/Users/Antares/Dropbox/Data/Orion/Other/Spitzer_Orion_IRAC1_coverage_s.fits"
# irac1_coverage = fits.open(irac1_coverage_path)[0].data
# irac1_coverage_header = fits.open(irac1_coverage_path)[0].header
#
# # Adjust axes
# # for ax in [ax0, ax1, ax2]:
# for ax in [ax1, ax2, ax3]:
#     ax.set_xlim(-0.5, grid_shape[1] - 0.5)
#     ax.set_ylim(-0.5, grid_shape[0] - 0.5)
#     # ax.contour(irac1_coverage, levels=[0, 1], transform=ax.get_transform(wcsaxes.WCS(irac1_coverage_header)),
#     #            colors="black")
#
# # Save figure
# plt.savefig(results_path + "extinction_law_spatial.pdf", bbox_inches="tight", dpi=300)
